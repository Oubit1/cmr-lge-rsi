"""Synthetic normal-role dispatch; providers and model execution never go live."""
from __future__ import annotations

import ast
import asyncio
import copy
import importlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents import governance, shadow_analysis
from app import segmentation_cycle as cycle
from app import segmentation_dispatch as dispatch
from app import segmentation_host as host
from app import segmentation_training as training
from app.control_checkpoint import save_journal
from app.database import Database
from app.state_machine import RunState
from contracts.compat import model_dump, model_schema, model_validate
from contracts.models import SegmentationCyclePlan, SpecialistAssessment, SupervisorAction
from providers.mock import MockProvider
from tests import test_segmentation_cycle as fixtures

PROJECT = Path(__file__).resolve().parents[1]


class SegmentationDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CycleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.plan = self.fixture.root, copy.deepcopy(self.fixture.plan)
        self.root.mkdir()
        vendor = Path(self.fixture.job["vendor"]["root"])
        architecture = vendor / "lge_seg/model.py"
        architecture.parent.mkdir(exist_ok=True)
        architecture.write_bytes(Path(self.plan["architecture"]["path"]).read_bytes())
        self.fixture.job["vendor"]["files"]["lge_seg/model.py"] = training.sha256_file(architecture)
        self.fixture.write_json("job.json", self.fixture.job)
        self.plan["job"] = fixtures.binding(self.fixture.job_path)
        self.plan["architecture"] = fixtures.binding(architecture)
        host.freeze_model_sources(self.fixture.reg, self.plan["parent_id"], vendor_root=vendor)
        self.db = Database(self.root / "runtime/workflow.sqlite")
        self.db.initialize()
        dataset = self.db.register_dataset("synthetic", Path(self.fixture.job["dataset_root"]))
        self.run_id = self.db.create_run(dataset, "synthetic", {})
        self.plan.update(run_id=self.run_id, action_id="synthetic-training",
                         owner="lge_myo_specialist", owner_role="lge_myo_specialist")
        self.fixture.plan = self.plan
        self.config = {"schema": "segmentation-dispatch-v1", "enabled": True,
                       "max_actions": 1, "walltime_seconds": 120,
                       "plan_inputs": [{k: self.plan[k] for k in dispatch.INPUT_FIELDS}]}
        self.payload = {"action_id": self.plan["action_id"], "action_class": "segmentation_training",
                        "owner_role": self.plan["owner_role"], "segmentation_cycle_plan": self.plan,
                        "auto_execute": True, "review_approved": True, "max_attempts": 1,
                        "objective": "Bounded segmentation training", "acceptance_metrics": ["frozen evaluation"], "priority": 1}
        self.write(dispatch.CONFIG, self.config)
        self.write("runtime/control/operator_state.json", {"paused": False})
        self.write("runtime/control/run_budget.json", {"schema_version": "run-budget-v1", "active": True,
                   "status": "RUNNING", "started_at": time.time(), "deadline": time.time() + 900,
                   "used": {"analyses": 1, "requests": 2}, "limits": {"analyses": 4, "requests": 40}})
        self.approve()
        network = patch("httpx.Client.send", side_effect=AssertionError("provider network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        async_network = patch("httpx.AsyncClient.send", side_effect=AssertionError("provider network forbidden"))
        async_network.start()
        self.addCleanup(async_network.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_journal(path, value)

    def message(self, kind, body, role="independent_reviewer"):
        self.db.post_agent_message(self.run_id, thread_id=f"run-{self.run_id}:{self.payload['action_id']}",
            sender_role=role, recipient_role=self.payload["owner_role"], message_type=kind, payload=body)

    def approve(self, implementation=True):
        self.db.save_maintenance_action(self.run_id, self.payload, status="APPROVED")
        self.message("assigned_action", self.payload, "supervisor")
        self.message("action_review", {"decision": "accept_experiment", "required_validation_metrics": [],
                     "segmentation_plan_sha256": cycle.implementation_sha256(self.plan)})
        if implementation:
            self.message("implementation_review", {"decision": "accept_experiment", "required_validation_metrics": [],
                         "implementation_sha256": cycle.implementation_sha256(self.plan)})

    def run_dispatch(self):
        return dispatch.dispatch(self.root, self.db, self.run_id, self.payload["action_id"],
                                 providers_path=PROJECT / "configs/providers.yaml", agents_path=PROJECT / "configs/agents.yaml")

    def row(self):
        return self.db.get_maintenance_action(self.run_id, self.payload["action_id"])

    def claim_path(self):
        key = cycle._hash({"run_id": self.run_id, "action_id": self.payload["action_id"]})
        return self.root / "runtime/segmentation_dispatch" / (key + ".json")

    def test_schema_preserves_exact_cycle_plan_and_forbids_self_approval(self):
        typed = model_validate(SegmentationCyclePlan, self.plan)
        self.assertEqual(model_dump(typed), self.plan)
        action = model_validate(SupervisorAction, self.payload)
        self.assertEqual(model_dump(action)["segmentation_cycle_plan"], self.plan)
        self.assertIn("segmentation_training", json.dumps(model_schema(SupervisorAction)))
        for extra in ("approved", "command", "metrics", "approval_path"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                model_validate(SegmentationCyclePlan, dict(self.plan, **{extra: True}))
        with self.assertRaises(ValueError):
            model_validate(SegmentationCyclePlan, dict(self.plan, run_id=True))

    def test_each_target_belongs_to_its_existing_specialist(self):
        for target, owner in governance.SEGMENTATION_OWNERS.items():
            plan = dict(self.plan, target=target, owner_role=owner, owner=owner)
            action = model_validate(SupervisorAction, dict(self.payload, owner_role=owner, segmentation_cycle_plan=plan))
            self.assertEqual(governance.enforce_action_ownership([action]), ([action], []))
            action.owner_role = "workflow_maintainer"
            kept, blocked = governance.enforce_action_ownership([action])
            self.assertEqual(kept, [])
            self.assertEqual(blocked[0]["expected_owner"], owner)

    def test_absent_or_disabled_config_has_no_work_or_review(self):
        for config in (None, {"enabled": False}):
            if config is None:
                (self.root / dispatch.CONFIG).unlink()
            else:
                self.write(dispatch.CONFIG, config)
            with patch.object(cycle, "_preflight") as preflight, patch.object(host, "make_controller") as factory:
                self.assertEqual(self.run_dispatch(), {"status": "DISABLED"})
            preflight.assert_not_called()
            factory.assert_not_called()
        self.assertFalse((self.root / "runtime/segmentation_dispatch").exists())

    def test_readiness_explains_disabled_paused_and_missing_approval_without_paths(self):
        current = dispatch.readiness(self.root)
        self.assertEqual(current["status"], "READY", current)
        self.assertEqual(current["queue"]["approved"], 1)
        self.message("implementation_review", {"decision": "reject"})
        self.assertEqual(dispatch.readiness(self.root)["reason_code"], "APPROVED_PLAN_MISSING")
        self.write("runtime/control/operator_state.json", {"paused": True})
        self.assertEqual(dispatch.readiness(self.root)["status"], "PAUSED")
        self.write(dispatch.CONFIG, {"enabled": False})
        self.assertEqual(dispatch.readiness(self.root)["reason_code"], "CONFIG_DISABLED")
        (self.root / dispatch.CONFIG).unlink()
        absent = dispatch.readiness(self.root)
        self.assertEqual(absent["reason_code"], "CONFIG_MISSING")
        self.assertIsNone(absent["queue"]["approved"])
        self.assertNotIn(str(self.root), json.dumps(current))
        self.assertNotIn(self.plan["action_id"], json.dumps(current))

    def test_invalid_optin_config_fails_closed(self):
        for change in ({"max_actions": 0}, {"max_actions": 4}, {"max_actions": True},
                       {"walltime_seconds": 0}, {"walltime_seconds": 21601}, {"command": "anything"}):
            self.write(dispatch.CONFIG, dict(self.config, **change))
            self.assertFalse(dispatch.proposal_context(self.root, self.run_id)["enabled"])
            with patch.object(host, "make_controller") as factory:
                self.assertEqual(self.run_dispatch()["status"], "FAILED")
            factory.assert_not_called()

    def test_pause_missing_or_exhausted_budget_blocks_every_launch(self):
        budget_path = self.root / "runtime/control/run_budget.json"
        original = json.loads(budget_path.read_text())
        mutations = [dict(original, active=False), dict(original, deadline=time.time()-1),
                     dict(original, deadline=float("inf")), dict(original, status="EXHAUSTED"),
                     dict(original, used={"analyses": 4, "requests": 2}),
                     dict(original, used={"analyses": 1, "requests": 40})]
        for budget in mutations:
            self.write("runtime/control/run_budget.json", budget)
            with patch.object(host, "make_controller") as factory:
                self.assertEqual(self.run_dispatch()["status"], "PAUSED")
            factory.assert_not_called()
        budget_path.unlink()
        self.assertEqual(self.run_dispatch()["status"], "PAUSED")
        self.write("runtime/control/run_budget.json", original)
        self.write("runtime/control/operator_state.json", {"paused": True})
        before = budget_path.read_bytes()
        self.assertEqual(self.run_dispatch()["status"], "PAUSED")
        self.assertEqual(budget_path.read_bytes(), before)

    def test_governance_requires_bound_assignment_and_latest_independent_review(self):
        row = self.row()
        self.assertTrue(governance.agent_action_authorized(self.db, self.run_id, row, row["payload"]))
        self.message("action_review", {"decision": "accept_experiment"})
        self.assertEqual(self.run_dispatch()["status"], "AWAITING_APPROVAL")
        self.approve()
        self.message("action_review", {"decision": "reject", "segmentation_plan_sha256": cycle.implementation_sha256(self.plan)})
        self.assertEqual(self.run_dispatch()["status"], "AWAITING_APPROVAL")
        self.approve()
        altered = dict(self.payload, segmentation_cycle_plan=dict(self.plan, policy_id="invented"))
        self.db.save_maintenance_action(self.run_id, altered, status="APPROVED")
        self.assertEqual(self.run_dispatch()["status"], "AWAITING_APPROVAL")

    def test_exact_implementation_review_and_preflight_are_required(self):
        self.message("implementation_review", {"decision": "accept_experiment", "required_validation_metrics": [],
                     "implementation_sha256": "0" * 64})
        with patch.object(cycle, "run_cycle") as runner:
            self.assertEqual(self.run_dispatch()["error_code"], "EXACT_PLAN_REVIEW_REQUIRED")
        runner.assert_not_called()
        self.assertEqual(self.row()["status"], "REJECTED")

    def test_hallucinated_artifact_path_is_rejected_even_if_messages_claim_approval(self):
        self.plan["job"] = {"path": str(self.root / "not-created.json"), "sha256": "a" * 64}
        self.config["plan_inputs"] = [{k: self.plan[k] for k in dispatch.INPUT_FIELDS}]
        self.write(dispatch.CONFIG, self.config)
        self.approve()
        with patch.object(cycle, "run_cycle") as runner:
            self.assertEqual(self.run_dispatch()["status"], "FAILED")
        runner.assert_not_called()
        self.assertEqual(self.row()["status"], "REJECTED")

    def test_host_receives_same_plan_and_checks_pause_during_work(self):
        budget = (self.root / "runtime/control/run_budget.json").read_bytes()
        def execute(root, plan, *, controller, execute):
            self.assertTrue(execute)
            self.assertEqual(plan, self.plan)
            request = cycle.ControlRequest(root, cycle.plan_sha256(plan), self.run_id, plan["action_id"], "training_poll")
            self.assertEqual(controller.control(request).action, cycle.Control.RUN)
            self.write("runtime/control/operator_state.json", {"paused": True})
            self.assertEqual(controller.control(request).action, cycle.Control.PAUSE)
            return {"stage": "CANCELLED"}
        with patch.object(cycle, "run_cycle", side_effect=execute):
            self.assertEqual(self.run_dispatch()["status"], "CANCELLED")
        self.assertEqual((self.root / "runtime/control/run_budget.json").read_bytes(), budget)
        self.assertEqual(self.row()["status"], "REJECTED")

    def test_session_action_cap_does_not_reset_on_config_increase(self):
        with patch.object(cycle, "run_cycle", return_value={"stage": "RETAIN_PARENT"}) as runner:
            self.assertEqual(self.run_dispatch()["status"], "RETAIN_PARENT")
            self.plan = dict(self.plan, action_id="second-training")
            self.payload = dict(self.payload, action_id=self.plan["action_id"], segmentation_cycle_plan=self.plan)
            self.config["max_actions"] = 3
            self.write(dispatch.CONFIG, self.config)
            self.approve()
            self.assertEqual(self.run_dispatch()["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(runner.call_count, 1)

    def test_waiting_evidence_resumes_same_cycle_without_new_claim(self):
        with patch.object(cycle, "run_cycle", return_value={"stage": "AWAITING_EVALUATION"}) as runner:
            self.assertEqual(self.run_dispatch()["status"], "AWAITING_EVALUATION")
            self.assertEqual(self.run_dispatch()["status"], "AWAITING_EVALUATION")
        self.assertEqual(runner.call_count, 2)
        ledger = json.loads((self.root / "runtime/segmentation_dispatch/sessions.json").read_text())
        self.assertEqual(len(next(iter(ledger.values()))["actions"]), 1)

    def test_interrupted_claim_and_failed_work_are_not_relaunched(self):
        with patch.object(cycle, "run_cycle", return_value={"stage": "AWAITING_EVALUATION"}):
            self.run_dispatch()
        state = json.loads(self.claim_path().read_text())
        state["status"] = "EXECUTING"
        save_journal(self.claim_path(), state)
        with patch.object(cycle, "run_cycle") as runner:
            self.assertEqual(self.run_dispatch()["error_code"], "INTERRUPTED_NO_RETRY")
            self.assertEqual(self.run_dispatch()["status"], "AWAITING_APPROVAL")
        runner.assert_not_called()
        self.assertEqual(self.row()["status"], "REJECTED")

    def test_overall_deadline_persists_across_waiting_resumes(self):
        with patch.object(cycle, "run_cycle", return_value={"stage": "AWAITING_EVALUATION"}):
            self.run_dispatch()
        state = json.loads(self.claim_path().read_text())
        state["deadline"] = time.time() - 1
        save_journal(self.claim_path(), state)
        with patch.object(cycle, "run_cycle") as runner:
            self.assertEqual(self.run_dispatch()["error_code"], "TIME_LIMIT")
        runner.assert_not_called()
        self.assertEqual(self.row()["status"], "REJECTED")

    def test_review_return_value_is_not_a_stored_approval(self):
        self.plan = dict(self.plan, action_id="unreviewed-training")
        self.payload = dict(self.payload, action_id=self.plan["action_id"], segmentation_cycle_plan=self.plan)
        self.approve(implementation=False)
        with patch("agents.implementation_review.review_implementation", return_value={"decision": "accept_experiment"}) as review, \
                patch.object(cycle, "run_cycle") as runner:
            self.assertEqual(self.run_dispatch()["error_code"], "EXACT_PLAN_REVIEW_REQUIRED")
        review.assert_called_once()
        self.assertEqual(review.call_args.kwargs["implementation"], self.plan)
        runner.assert_not_called()

    def test_real_cycle_trains_once_and_waits_for_independent_evidence_without_activation(self):
        sources = {p: p.read_bytes() for p in Path(self.fixture.job["dataset_root"]).rglob("*") if p.is_file()}
        with patch.object(training, "run_job", side_effect=self.fixture.receipt) as train, \
                patch.object(host.RegistryPredictor, "__call__", side_effect=self.fixture.predict):
            first = self.run_dispatch()
            second = self.run_dispatch()
        self.assertEqual(first["status"], "AWAITING_EVALUATION", first)
        self.assertEqual(second["status"], "AWAITING_EVALUATION", second)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(len(self.fixture.predictions), 2)
        self.assertEqual(second["review_status"], "MANUAL_EVIDENCE_REQUIRED")
        requests = self.db.agent_inbox(self.run_id, "independent_reviewer",
                                      thread_id=f"run-{self.run_id}:{self.plan['action_id']}")
        self.assertEqual(len([r for r in requests if r["message_type"] == "segmentation_evaluation_request"]), 1)
        self.assertEqual(dispatch.readiness(self.root)["reason_code"], "MANUAL_EVIDENCE_REQUIRED")
        from app.segmentation_registry import resolve_active
        self.assertEqual(resolve_active(self.fixture.reg, self.plan["target"])["candidate_id"], self.plan["parent_id"])
        self.assertEqual(sources, {p: p.read_bytes() for p in sources})

    def test_normal_dispatch_stored_evaluation_and_activation_review_roundtrip(self):
        class Reviewer(MockProvider):
            async def generate_structured(self, **kwargs):
                self.assert_schema = kwargs["output_schema"].__name__
                return model_validate(kwargs["output_schema"], {"decision": "accept_experiment",
                    "reasons": ["synthetic independently stored evidence"], "required_validation_metrics": [],
                    "risk_flags": [], "confidence": 0.8})
        self.config["automatic_reviews"] = True
        self.write(dispatch.CONFIG, self.config)
        with patch.object(training, "run_job", side_effect=self.fixture.receipt) as train, \
                patch.object(host.RegistryPredictor, "__call__", side_effect=self.fixture.predict), \
                patch("agents.implementation_review.ProviderRegistry.get", return_value=Reviewer()) as reviewer:
            self.assertEqual(self.run_dispatch()["review_status"], "MANUAL_EVIDENCE_REQUIRED")
            reviewer.assert_not_called()  # Missing exposure is not something a model may infer.
            outcome = json.loads((self.root / "runtime/segmentation_cycles" / self.claim_path().stem / "latest.json").read_text())
            common = {"status": "VERIFIED", "plan_sha256": cycle.plan_sha256(self.plan)}
            self.message("segmentation_reference_provenance", dict(common,
                cohort_sha256=self.plan["cohort"]["sha256"], reference_sha256=outcome["report"]["reference_sha256"]), "independent_evaluator")
            self.message("segmentation_parent_exposure", dict(common, parent_id=self.plan["parent_id"],
                cohort_sha256=self.plan["cohort"]["sha256"], training_evaluation_patient_disjoint="ESTABLISHED"), "independent_evaluator")
            self.message("segmentation_split_evidence", dict(common, dataset_sha256=self.plan["dataset_manifest"]["sha256"],
                cohort_sha256=self.plan["cohort"]["sha256"], validation_test_patient_disjoint="ESTABLISHED",
                training_evaluation_patient_disjoint="ESTABLISHED"), "independent_evaluator")
            result = self.run_dispatch()
        self.assertEqual(result["status"], "ACTIVATED", result)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(reviewer.call_count, 2)
        messages = self.db.agent_inbox(self.run_id, self.plan["owner_role"])
        evidence = [m for m in messages if m["message_type"] == "segmentation_evaluation_evidence"]
        activation = [m for m in messages if m["message_type"] == "segmentation_activation_review"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(len(activation), 1)
        self.assertEqual(evidence[0]["payload"]["status"], "VERIFIED")
        self.assertEqual(activation[0]["payload"]["decision"], "accept_experiment")
        self.assertEqual(evidence[0]["payload"]["candidate_id"], activation[0]["payload"]["candidate_id"])
        requests = self.db.agent_inbox(self.run_id, "independent_reviewer")
        self.assertEqual(len([m for m in requests if m["message_type"] == "segmentation_activation_request"]), 1)

    def test_missing_model_sources_creates_one_request_and_remains_explicit(self):
        with patch.object(host, "make_controller", side_effect=host.SourceUnavailable("synthetic missing source")), \
                patch.object(cycle, "run_cycle") as runner:
            for _ in range(2):
                self.assertEqual(self.run_dispatch()["status"], "AWAITING_MODEL_SOURCES")
        runner.assert_not_called()
        requests = self.db.agent_inbox(self.run_id, "independent_reviewer")
        self.assertEqual(len([m for m in requests if m["message_type"] == "segmentation_source_request"]), 1)
        self.assertEqual(dispatch.readiness(self.root)["reason_code"], "MODEL_SOURCES_REQUIRED")

    def test_supervisor_must_carry_actual_owner_plan_unchanged(self):
        action = model_validate(SupervisorAction, self.payload)
        specialist = {"role": self.plan["owner_role"], "applicable": True, "segmentation_cycle_plan": self.plan}
        context = dispatch.proposal_context(self.root, self.run_id)
        workflow = {"stage": "cmr_paired", "integration_required": True}
        self.assertEqual(shadow_analysis.filter_workflow_actions([action], workflow, context, [specialist]), ([action], []))
        for supplied in ([], [dict(specialist, role="cmr_specialist")], [dict(specialist, applicable=False)]):
            self.assertEqual(shadow_analysis.filter_workflow_actions([action], workflow, context, supplied)[0], [])
        self.assertEqual(shadow_analysis.filter_workflow_actions([action], workflow)[0], [])
        self.assertEqual(shadow_analysis.filter_workflow_actions([action], dict(workflow, protocol_repair_required=True), context, [specialist])[0], [])

    def test_normal_shadow_chain_persists_typed_owner_plan_and_hash_bound_handoff(self):
        plan, payload = copy.deepcopy(self.plan), copy.deepcopy(self.payload)
        class Provider(MockProvider):
            async def generate_structured(self, **kwargs):
                schema = kwargs["output_schema"]
                if schema.__name__ == "SupervisorDecision":
                    return model_validate(schema, {"decision": "experiment", "rationale": ["existing prepared inputs"],
                        "actions": [payload], "rejected_proposals": [], "global_risk_flags": [],
                        "max_parallel_actions": 1, "confidence": 0.8})
                if schema.__name__ == "ReviewDecision":
                    return model_validate(schema, {"decision": "accept_experiment", "reasons": ["synthetic"],
                        "required_validation_metrics": [], "risk_flags": [], "confidence": 0.8})
                output = await super().generate_structured(**kwargs)
                if schema.__name__ == "SpecialistAssessment" and output.role == plan["owner_role"]:
                    output.segmentation_cycle_plan = model_validate(SegmentationCyclePlan, plan)
                return output
        evidence = {"run_id": self.run_id, "issue_counts": {"synthetic": 1}, "modalities": ["cmr"],
                    "workflow_dependency": {"stage": "cmr_paired"}, "iteration_history": [],
                    "segmentation_dispatch": dispatch.proposal_context(self.root, self.run_id)}
        self.db.transition(self.run_id, RunState.EVIDENCE_READY)
        with patch.object(shadow_analysis, "build_evidence_package", return_value=evidence), \
                patch("agents.shadow_analysis.ProviderRegistry.get", return_value=Provider()):
            result = asyncio.run(shadow_analysis.run_shadow_analysis(self.db, self.run_id,
                PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"))
        self.assertEqual(result["final_state"], "EXPERIMENT_PROPOSED")
        row = self.row()
        self.assertEqual(row["payload"]["segmentation_cycle_plan"], plan)
        self.assertEqual(row["action_class"], "segmentation_training")
        self.assertTrue(governance.agent_action_authorized(self.db, self.run_id, row, row["payload"]))

    def test_autopilot_routes_training_away_from_qc_code_executor(self):
        # Only stub the unavailable Windows import; no fcntl operation is used.
        with patch.dict(sys.modules, {"fcntl": Mock()} if os.name == "nt" else {}):
            autopilot = importlib.import_module("scripts.autopilot")
        pilot = autopilot.Autopilot.__new__(autopilot.Autopilot)
        pilot.db, pilot.dataset, pilot.state = self.db, Path(self.fixture.job["dataset_root"]), {}
        pilot.event = Mock()
        pilot.args = Mock(providers=PROJECT / "configs/providers.yaml", agents=PROJECT / "configs/agents.yaml", bounded_run=None)
        with patch.object(autopilot, "ROOT", self.root), \
                patch.object(autopilot, "active_production_blocker", return_value=None), \
                patch.object(autopilot, "dependent_workflow_status", return_value={}), \
                patch.object(autopilot, "enrich_workflow_dependency", return_value={"stage": "cmr_paired"}), \
                patch.object(autopilot, "history", return_value=[]), \
                patch("app.autonomy_control.operator_paused", return_value=False), \
                patch.object(dispatch, "dispatch", return_value={"status": "AWAITING_EVALUATION"}) as route, \
                patch.object(autopilot, "run_llm_maintenance") as qc:
            result = pilot.process_actions(self.run_id)
        route.assert_called_once()
        qc.assert_not_called()
        self.assertEqual(result["results"][0]["segmentation"]["status"], "AWAITING_EVALUATION")

    def test_python39_syntax(self):
        for path in (Path(dispatch.__file__), Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
