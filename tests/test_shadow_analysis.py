from __future__ import annotations

import asyncio
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.shadow_analysis import (
    filter_workflow_actions,
    filter_repeated_actions,
    run_shadow_analysis,
    select_analysis_roles,
    supervisor_output_needs_retry,
)
from agents.evidence import progress_ledger
from agents.evidence import action_builds_stage_consumer
from agents.governance import enforce_action_ownership, enforce_autonomous_executor_contract
from contracts.models import FailureAnalysis
from contracts.models import SupervisorAction, SupervisorDecision
from contracts.compat import model_validate
from app.database import Database
from app.state_machine import RunState
from providers.mock import MockProvider


PROJECT = Path(__file__).resolve().parents[1]


class ShadowAnalysisTests(unittest.TestCase):
    def test_stage_consumer_recognizes_unambiguous_latest_report_alias(self):
        self.assertTrue(action_builds_stage_consumer({
            "objective": "Patch scripts/lge_feedback_loop.py and return latest.json",
            "acceptance_metrics": ["lge_stage_consumer_contract.passed==true"],
        }))
        self.assertFalse(action_builds_stage_consumer({
            "objective": "Write latest.json for an unrelated experiment",
        }))

    def test_experiment_decision_without_actions_requires_one_retry(self):
        decision = SupervisorDecision(
            decision="experiment", rationale=["bounded"], actions=[],
            rejected_proposals=[], global_risk_flags=[],
            max_parallel_actions=1, confidence=0.8,
        )
        self.assertTrue(supervisor_output_needs_retry(decision, [], [], False))
        decision.decision = "stop_no_gain"
        self.assertFalse(supervisor_output_needs_retry(decision, [], [], False))

    def test_executable_action_must_be_owned_by_implementing_role(self):
        action = SupervisorAction(
            action_id="bad-owner", owner_role="cmr_specialist",
            action_class="isolated_code_experiment", objective="experiment",
            acceptance_metrics=["gate"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        kept, blocked = enforce_action_ownership([action])
        self.assertEqual(kept, [])
        self.assertEqual(blocked[0]["expected_owner"], "qc_code_optimizer")

    def test_action_without_autonomous_executor_cannot_claim_auto_execute(self):
        action = SupervisorAction(
            action_id="unsupported-tool", owner_role="efficiency_auditor",
            action_class="known_tool", objective="只读对账",
            acceptance_metrics=["输出报告"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        kept, blocked = enforce_autonomous_executor_contract([action])
        self.assertEqual(kept, [])
        self.assertEqual(blocked[0]["reason"], "no_autonomous_executor")

    def test_manual_known_tool_remains_available_for_real_human_dependency(self):
        action = SupervisorAction(
            action_id="manual-tool", owner_role="efficiency_auditor",
            action_class="known_tool", objective="外部工具核验",
            acceptance_metrics=["输出报告"], priority=1, max_attempts=1,
            auto_execute=False,
        )
        kept, blocked = enforce_autonomous_executor_contract([action])
        self.assertEqual(kept, [action])
        self.assertEqual(blocked, [])

    def test_semantic_duplicate_guard_ignores_changed_run_numbers(self):
        prior = [{"run_id": 8, "actions": [{
            "action_id": "A1-CMR-SHADOW-08",
            "action_class": "isolated_code_experiment",
            "objective": "Run CMR centroid drift shadow audit for run 8 on 50 cases",
        }]}]
        repeated = SupervisorAction(
            action_id="A1-CMR-SHADOW-50", owner_role="cmr_specialist",
            action_class="isolated_code_experiment",
            objective="Run CMR centroid drift shadow audit for run 50 on 50 cases",
            acceptance_metrics=["recall"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        novel = SupervisorAction(
            action_id="A2-LGE-MI", owner_role="lge_mi_specialist",
            action_class="rule_draft",
            objective="Draft LGE infarct containment rule using myocardium support",
            acceptance_metrics=["false repair"], priority=2, max_attempts=1,
            auto_execute=True,
        )
        kept, blocked = filter_repeated_actions([repeated, novel], prior)
        self.assertEqual([item.action_id for item in kept], ["A2-LGE-MI"])
        self.assertEqual(blocked[0]["matches"], "A1-CMR-SHADOW-08")

    def test_progress_ledger_shares_review_blocker_without_patient_data(self):
        history = [{"run_id": 7, "actions": [{
            "action_id": "A7", "action_class": "rule_draft",
            "status": "HUMAN_REQUIRED", "objective": "影子规则验证",
            "review_feedback": {"decision": "needs_human_review",
                                "reasons": ["缺少真实留出统计"],
                                "required_validation_metrics": ["每簇计数"]},
        }]}]
        self.assertEqual(progress_ledger(history)[0]["unmet_metrics"], ["每簇计数"])
        self.assertEqual(progress_ledger(history)[0]["review_decision"],
                         "needs_human_review")

    def test_near_duplicate_continuation_requires_real_prior_feedback(self):
        metric = "Per-cluster counts on held-out cases"
        prior = [{"run_id": 7, "actions": [{
            "action_id": "A7", "action_class": "isolated_code_experiment",
            "status": "HUMAN_REQUIRED", "objective": "影子运行三簇分诊，输出每簇标记，不写标签",
            "review_feedback": {"required_validation_metrics": [metric]},
        }]}]
        def action(**extra):
            return SupervisorAction(
                action_id="A8", owner_role="qc_code_optimizer",
                action_class="isolated_code_experiment",
                objective="影子运行三簇分诊，输出每簇标记与真实留出计数，不写标签",
                acceptance_metrics=[metric], priority=1, max_attempts=1,
                auto_execute=True, **extra)
        kept, blocked = filter_repeated_actions([action()], prior)
        self.assertEqual(kept, [])
        self.assertEqual(len(blocked), 1)
        kept, blocked = filter_repeated_actions([action(
            continuation_of="A7", addresses_metric="invented metric",
            continuation_delta="记录每簇真实留出计数")], prior)
        self.assertEqual(kept, [])
        self.assertEqual(blocked[0]["reason"], "invalid_continuation_reference")
        kept, blocked = filter_repeated_actions([action(
            continuation_of="unknown", addresses_metric=metric,
            continuation_delta="记录每簇真实留出计数")], prior)
        self.assertEqual(kept, [])
        kept, blocked = filter_repeated_actions([action(
            continuation_of="A7", addresses_metric=metric,
            continuation_delta="记录每簇真实留出计数与分母")], prior)
        self.assertEqual(len(kept), 1)
        self.assertEqual(blocked, [])
        identical = action(continuation_of="A7", addresses_metric=metric,
                           continuation_delta="声称有新的计数，但目标没有变化")
        identical.objective = prior[0]["actions"][0]["objective"]
        kept, blocked = filter_repeated_actions([identical], prior)
        self.assertEqual(kept, [])

    def test_promoted_action_can_continue_an_explicit_unmet_metric(self):
        metric = "route_conflicts_unresolved<15 on frozen replay"
        prior = [{"run_id": 800, "actions": [{
            "action_id": "S800", "action_class": "isolated_code_experiment",
            "status": "PROMOTED", "objective": "Dry-run 15 route conflicts",
            "review_feedback": {"required_validation_metrics": [metric]},
        }]}]
        continuation = SupervisorAction(
            action_id="S804", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment",
            objective="Replay the promoted resolver on the current pending queue",
            acceptance_metrics=[metric], priority=1, max_attempts=1,
            auto_execute=True, continuation_of="S800", addresses_metric=metric,
            continuation_delta="Replay against the changed pending queue and report residual conflicts",
        )
        kept, blocked = filter_repeated_actions([continuation], prior)
        self.assertEqual([item.action_id for item in kept], ["S804"])
        self.assertEqual(blocked, [])

    def test_exact_lge_consumer_retry_is_allowed_after_framework_failure(self):
        metric = "lge_stage_consumer_contract.passed==true"
        objective = (
            "Implement scripts/lge_feedback_loop.py and continuously emit "
            "runtime/lge_iterations/latest.json"
        )
        prior = [{"run_id": 876, "actions": [{
            "action_id": "S876", "action_class": "isolated_code_experiment",
            "status": "HUMAN_REQUIRED", "objective": objective,
            "review_feedback": {"required_validation_metrics": [metric]},
        }]}]
        continuation = SupervisorAction(
            action_id="S880", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment", objective=objective,
            acceptance_metrics=[metric], priority=1, max_attempts=1,
            auto_execute=True, continuation_of="S876", addresses_metric=metric,
            continuation_delta="Retry after the framework supplied the missing immutable interface",
        )
        kept, blocked = filter_repeated_actions([continuation], prior)
        self.assertEqual([item.action_id for item in kept], ["S880"])
        self.assertEqual(blocked, [])

    def test_rejected_lge_consumer_can_continue_exact_failed_gate_check(self):
        objective = (
            "Implement scripts/lge_feedback_loop.py as the active LGE consumer and expose "
            "runtime/lge_iterations/latest.json through the framework driver"
        )
        prior = [{"run_id": 882, "actions": [{
            "action_id": "S882", "action_class": "isolated_code_experiment",
            "status": "REJECTED", "objective": objective,
            "validation_result": {"suites": [{
                "suite": "lge_stage_consumer_contract",
                "checks": {"holdout_contract_exact": False, "counts_reconcile": True},
            }]},
        }]}]
        continuation = SupervisorAction(
            action_id="S886", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment", objective=objective,
            acceptance_metrics=["holdout_contract_exact==true"], priority=1,
            max_attempts=1, auto_execute=True, continuation_of="S882",
            addresses_metric="holdout_contract_exact==false",
            continuation_delta="Correct the exact failed machine contract in the named adapter",
        )
        kept, blocked = filter_repeated_actions([continuation], prior)
        self.assertEqual([item.action_id for item in kept], ["S886"])
        self.assertEqual(blocked, [])

    def test_declared_continuation_reaches_independent_reviewer(self):
        metric = "每簇留出计数与分母"
        history = [{"run_id": 7, "actions": [{
            "action_id": "A7", "action_class": "rule_draft",
            "status": "HUMAN_REQUIRED",
            "objective": "起草只读三簇规则，输出每簇标记，不写标签",
            "review_feedback": {"decision": "needs_human_review",
                                "required_validation_metrics": [metric]},
        }]}]

        class ContinuationProvider(MockProvider):
            saw_ledger = False
            reviewed_continuation = False

            async def generate_structured(self, **kwargs):
                schema = kwargs["output_schema"]
                data = kwargs["input_data"]
                if schema.__name__ == "SupervisorDecision":
                    self.saw_ledger = data["evidence_summary"]["progress_ledger"][0][
                        "unmet_metrics"] == [metric]
                    return model_validate(schema, {
                        "decision": "experiment", "rationale": ["补齐验证计数"],
                        "actions": [{"action_id": "A8", "owner_role": "rule_curator",
                                     "action_class": "rule_draft",
                                     "objective": "起草只读三簇规则，输出每簇标记及验证计数，不写标签",
                                     "acceptance_metrics": [metric], "priority": 1,
                                     "max_attempts": 1, "auto_execute": True,
                                     "continuation_of": "A7", "addresses_metric": metric,
                                     "continuation_delta": "新增逐簇真实留出计数及明确分母"}],
                        "rejected_proposals": [], "global_risk_flags": [],
                        "max_parallel_actions": 1, "confidence": 0.8,
                    })
                if schema.__name__ == "ReviewDecision":
                    self.reviewed_continuation = data["supervisor_decision"]["actions"][0][
                        "continuation_of"] == "A7"
                    return model_validate(schema, {
                        "decision": "accept_experiment", "reasons": ["隔离实验"],
                        "required_validation_metrics": [metric], "risk_flags": [],
                        "confidence": 0.8,
                    })
                return await super().generate_structured(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "ANON", "modality": "cmr", "issues": "topology"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "audit",
                                   {"iteration_history": history})
            for state in (RunState.QUEUED, RunState.RUNNING, RunState.QC_FAILED):
                db.transition(run_id, state)
            db.register_artifact(run_id, None, "cmr_boundary_manifest", manifest, {"rows": 1})
            provider = ContinuationProvider()
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "unit-test-salt"}), patch(
                "agents.shadow_analysis.ProviderRegistry.get", return_value=provider
            ):
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"
                ))
            self.assertTrue(provider.saw_ledger)
            self.assertTrue(provider.reviewed_continuation)
            self.assertEqual(result["final_state"], "EXPERIMENT_PROPOSED")

    def test_lge_myo_stage_prevents_cmr_from_running_early(self):
        analysis = FailureAnalysis(
            failure_type="cmr_myo_small_area",
            affected_modalities=["cmr"], evidence_cases=[], hypothesis="topology",
            suggested_experiment="shadow comparison", expected_metrics=[],
            confidence=0.8, requires_human_review=False,
        )
        specialists, maintainers = select_analysis_roles(
            {"modalities": ["cmr", "lge"], "iteration_history": [],
             "workflow_dependency": {"stage": "lge_myo"}}, analysis
        )
        self.assertEqual(specialists, ["lge_myo_specialist"])
        self.assertEqual(maintainers, ["rule_curator"])

    def test_code_owner_can_measure_debt_from_rule_or_archived_shadow(self):
        metric = "report coverage counts"
        for action_class, status in (("rule_draft", "HUMAN_REQUIRED"),
                                     ("isolated_code_experiment", "PROMOTED_SHADOW")):
            prior = [{"actions": [{"action_id": "prior", "action_class": action_class,
                       "status": status, "objective": "collect coverage counts",
                       "review_feedback": {"required_validation_metrics": [metric]}}]}]
            action = SupervisorAction(action_id="next", owner_role="qc_code_optimizer",
                action_class="isolated_code_experiment", objective="collect real coverage counts",
                acceptance_metrics=[metric], priority=1, max_attempts=1,
                continuation_of="prior", addresses_metric=metric,
                continuation_delta="Execute existing module and report actual denominators")
            kept, blocked = filter_repeated_actions([action], prior)
            self.assertEqual(len(kept), 1)
            self.assertEqual(blocked, [])

    def test_lge_mi_stage_runs_only_after_myo_stage(self):
        analysis = FailureAnalysis(
            failure_type="mi_multiple_components",
            affected_modalities=["lge"], evidence_cases=[], hypothesis="fragmentation",
            suggested_experiment="shadow comparison", expected_metrics=[],
            confidence=0.8, requires_human_review=False,
        )
        specialists, maintainers = select_analysis_roles(
            {"modalities": ["cmr", "lge"], "iteration_history": [{}, {}],
             "workflow_dependency": {"stage": "lge_mi"}}, analysis
        )
        self.assertEqual(specialists, ["lge_mi_specialist"])
        self.assertEqual(maintainers, ["rule_curator"])

    def test_cross_modality_failure_keeps_code_optimizer_and_periodic_skill(self):
        analysis = FailureAnalysis(
            failure_type="paired_trajectory",
            affected_modalities=["cmr", "lge"], evidence_cases=[], hypothesis="alignment",
            suggested_experiment="paired shadow comparison", expected_metrics=[],
            confidence=0.8, requires_human_review=False,
        )
        specialists, maintainers = select_analysis_roles(
            {"modalities": ["cmr", "lge"], "iteration_history": [{}, {}, {}],
             "workflow_dependency": {"stage": "paired_selection"}}, analysis
        )
        self.assertEqual(specialists, ["lge_myo_specialist", "lge_mi_specialist"])
        self.assertEqual(maintainers, ["rule_curator"])

    def test_lge_consumer_integration_schedules_only_code_owner(self):
        analysis = FailureAnalysis(
            failure_type="missing production consumer", affected_modalities=["lge"],
            evidence_cases=[], hypothesis="consumer contract is missing",
            suggested_experiment="implement workflow consumer", expected_metrics=[],
            confidence=0.9, requires_human_review=False,
        )
        specialists, maintainers = select_analysis_roles(
            {"modalities": ["lge"], "iteration_history": [],
             "workflow_dependency": {"stage": "lge_myo", "integration_required": True}},
            analysis,
        )
        self.assertEqual(specialists, ["lge_myo_specialist"])
        self.assertEqual(maintainers, ["qc_code_optimizer"])

    def test_cmr_specialist_runs_only_for_persisted_paired_stage(self):
        analysis = FailureAnalysis(
            failure_type="paired_cmr_myo", affected_modalities=["cmr", "lge"],
            evidence_cases=[], hypothesis="paired repair",
            suggested_experiment="paired-only shadow comparison", expected_metrics=[],
            confidence=0.8, requires_human_review=False,
        )
        specialists, _ = select_analysis_roles(
            {"modalities": ["cmr", "lge"], "iteration_history": [],
             "workflow_dependency": {"stage": "cmr_paired"}}, analysis
        )
        self.assertEqual(specialists, ["cmr_specialist"])

    def test_mock_analysis_is_anonymous_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "REAL_NAME", "modality": "lge", "issues": "myo_topology"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "audit", {})
            db.transition(run_id, RunState.QUEUED)
            db.transition(run_id, RunState.RUNNING)
            db.transition(run_id, RunState.QC_FAILED)
            db.register_artifact(run_id, None, "lge_issue_manifest", manifest, {"rows": 1})
            old = os.environ.get("CMR_LGE_CASE_SALT")
            os.environ["CMR_LGE_CASE_SALT"] = "unit-test-salt"
            try:
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"
                ))
            finally:
                if old is None:
                    os.environ.pop("CMR_LGE_CASE_SALT", None)
                else:
                    os.environ["CMR_LGE_CASE_SALT"] = old
            self.assertEqual(result["final_state"], "NEEDS_REVIEW")
            payload = str(result)
            self.assertNotIn("REAL_NAME", payload)
            self.assertIn("case_", payload)

    def test_one_maintenance_role_failure_does_not_cancel_completed_agents(self):
        class FailingMaintainer(MockProvider):
            async def generate_structured(self, **kwargs):
                if (kwargs["output_schema"].__name__ == "MaintenanceProposal"
                        and kwargs["input_data"].get("role") == "rule_curator"):
                    raise TimeoutError("simulated role timeout")
                return await super().generate_structured(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "ANON", "modality": "lge",
                                 "issues": "mi_multiple_components"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            run_id = db.create_run(dataset_id, "audit", {})
            for state in (RunState.QUEUED, RunState.RUNNING, RunState.QC_FAILED):
                db.transition(run_id, state)
            db.register_artifact(run_id, None, "lge_issue_manifest", manifest, {"rows": 1})
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "unit-test-salt"}), patch(
                "agents.shadow_analysis.ProviderRegistry.get",
                return_value=FailingMaintainer(),
            ):
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml",
                    PROJECT / "configs/agents.yaml",
                ))

            statuses = {row["role"]: row["status"]
                        for row in db.run_details(run_id)["agent_outputs"]}
            self.assertEqual(statuses["rule_curator"], "FAILED")
            self.assertEqual(statuses["lge_myo_specialist"], "SUCCEEDED")
            self.assertNotIn("lge_mi_specialist", statuses)
            self.assertEqual(statuses["supervisor"], "SUCCEEDED")
            self.assertEqual(result["final_state"], "NEEDS_REVIEW")

    def test_workflow_guard_blocks_supervisor_action_for_later_stage(self):
        action = model_validate(SupervisorAction, {
            "action_id": "A-CMR-EARLY", "owner_role": "qc_code_optimizer",
            "action_class": "isolated_code_experiment",
            "objective": "CMR myocardium scan", "acceptance_metrics": ["bounded"],
            "priority": 1, "max_attempts": 1, "auto_execute": True,
        })
        allowed, blocked = filter_workflow_actions([action], {"stage": "lge_myo"})
        self.assertEqual(allowed, [])
        self.assertEqual(blocked[0]["reason"], "workflow_stage_mismatch")
        self.assertEqual(blocked[0]["action_stage"], "cmr_paired")

    def test_workflow_guard_requires_consumer_after_bounded_shadow_evidence(self):
        scan = SupervisorAction(
            action_id="A-LGE-MYO-SCAN", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment", objective="LGE myo shadow scan",
            acceptance_metrics=["bounded"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        integration = SupervisorAction(
            action_id="A-LGE-CONSUMER", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment",
            objective=("Build scripts/lge_feedback_loop.py production consumer and "
                       "runtime/lge_iterations/latest.json"),
            acceptance_metrics=["bounded"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        workflow = {"stage": "lge_myo", "integration_required": True,
                    "required_output": "persist audited consumer report"}
        allowed, blocked = filter_workflow_actions([scan, integration], workflow)
        self.assertEqual([item.action_id for item in allowed], ["A-LGE-CONSUMER"])
        self.assertEqual(blocked[0]["reason"], "stage_consumer_required")

        experiment_only = SupervisorAction(
            action_id="A-LGE-EXPERIMENT", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment",
            objective=("Write experiments/lge_consumer_probe.py and "
                       "runtime/lge_iterations/latest.json"),
            acceptance_metrics=["bounded"], priority=1, max_attempts=1,
            auto_execute=True,
        )
        allowed, blocked = filter_workflow_actions([experiment_only], workflow)
        self.assertEqual(allowed, [])
        self.assertEqual(blocked[0]["reason"], "stage_consumer_required")

        stale = SupervisorAction(
            action_id="A-LGE-STALE", owner_role="qc_code_optimizer",
            action_class="isolated_code_experiment",
            objective=("Build scripts/lge_feedback_loop.py and emit "
                       "runtime/lge_iterations/latest.json for 9 non-holdout LGE cases"),
            acceptance_metrics=["pending+work+resolved==9", "holdout_excluded_count==48"],
            priority=1, max_attempts=1, auto_execute=True,
        )
        allowed, blocked = filter_workflow_actions([stale], workflow)
        self.assertEqual(allowed, [])
        self.assertEqual(blocked[0]["reason"], "stale_stage_denominator")

    def test_reviewer_approved_action_is_handed_to_owner_through_messages(self):
        class AcceptingProvider(MockProvider):
            async def generate_structured(self, **kwargs):
                name = kwargs["output_schema"].__name__
                if name == "SupervisorDecision":
                    return model_validate(kwargs["output_schema"], {
                        "decision": "experiment", "rationale": ["bounded"],
                        "actions": [{
                            "action_id": "A-RULE-1", "owner_role": "rule_curator",
                            "action_class": "rule_draft", "objective": "起草只读规则实验",
                            "acceptance_metrics": ["误修率不升高"], "priority": 1,
                            "max_attempts": 1, "auto_execute": True,
                        }],
                        "rejected_proposals": [], "global_risk_flags": [],
                        "max_parallel_actions": 1, "confidence": 0.8,
                    })
                if name == "ReviewDecision":
                    return model_validate(kwargs["output_schema"], {
                        "decision": "accept_experiment", "reasons": ["isolated"],
                        "required_validation_metrics": ["false_repair_rate"],
                        "risk_flags": [], "confidence": 0.8,
                    })
                return await super().generate_structured(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "ANON", "modality": "cmr", "issues": "topology"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "audit", {})
            for state in (RunState.QUEUED, RunState.RUNNING, RunState.QC_FAILED):
                db.transition(run_id, state)
            db.register_artifact(run_id, None, "cmr_boundary_manifest", manifest, {"rows": 1})
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "unit-test-salt"}), patch(
                "agents.shadow_analysis.ProviderRegistry.get", return_value=AcceptingProvider()
            ):
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"
                ))
            self.assertEqual(result["final_state"], "EXPERIMENT_PROPOSED")
            action = db.run_details(run_id)["maintenance_actions"][0]
            self.assertEqual(action["status"], "APPROVED")
            saved_payload = json.loads(action["payload_json"])
            self.assertEqual(saved_payload["workflow_stage"], "lge_myo")
            self.assertIn("LGE myocardium", saved_payload["workflow_dependency_contract"])
            inbox = db.agent_inbox(run_id, "rule_curator", thread_id=f"run-{run_id}:A-RULE-1")
            self.assertEqual(
                [item["message_type"] for item in inbox], ["assigned_action", "action_review"]
            )

    def test_rejected_owner_is_returned_to_supervisor_once_before_review(self):
        class CorrectingProvider(MockProvider):
            supervisor_calls = 0
            reviewed_actions = None

            async def generate_structured(self, **kwargs):
                name = kwargs["output_schema"].__name__
                if name == "SupervisorDecision":
                    self.supervisor_calls += 1
                    owner = "cmr_specialist" if self.supervisor_calls == 1 else "qc_code_optimizer"
                    if self.supervisor_calls == 2:
                        self.assert_feedback(kwargs["input_data"])
                    return model_validate(kwargs["output_schema"], {
                        "decision": "experiment", "rationale": ["bounded"],
                        "actions": [{"action_id": "A-CODE-1", "owner_role": owner,
                                     "action_class": "isolated_code_experiment",
                                     "objective": "只读代码实验", "acceptance_metrics": ["误修率"],
                                     "priority": 1, "max_attempts": 1, "auto_execute": True}],
                        "rejected_proposals": [], "global_risk_flags": [],
                        "max_parallel_actions": 1, "confidence": 0.8,
                    })
                if name == "ReviewDecision":
                    self.reviewed_actions = kwargs["input_data"]["supervisor_decision"]["actions"]
                    return model_validate(kwargs["output_schema"], {
                        "decision": "accept_experiment", "reasons": ["isolated"],
                        "required_validation_metrics": [], "risk_flags": [], "confidence": 0.8,
                    })
                return await super().generate_structured(**kwargs)

            def assert_feedback(self, data):
                assert data["guard_feedback"][0]["expected_owner"] == "qc_code_optimizer"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "ANON", "modality": "cmr", "issues": "topology"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "audit", {})
            for state in (RunState.QUEUED, RunState.RUNNING, RunState.QC_FAILED):
                db.transition(run_id, state)
            db.register_artifact(run_id, None, "cmr_boundary_manifest", manifest, {"rows": 1})
            provider = CorrectingProvider()
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "unit-test-salt"}), patch(
                "agents.shadow_analysis.ProviderRegistry.get", return_value=provider
            ):
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"
                ))
            self.assertEqual(provider.supervisor_calls, 2)
            self.assertEqual(provider.reviewed_actions[0]["owner_role"], "qc_code_optimizer")
            self.assertEqual(result["final_state"], "EXPERIMENT_PROPOSED")
            self.assertEqual(db.run_details(run_id)["maintenance_actions"][0]["owner_role"],
                             "qc_code_optimizer")

    def test_supervisor_reconsiders_manual_only_isolated_experiment(self):
        class AutonomyProvider(MockProvider):
            calls = 0
            feedback_seen = False

            async def generate_structured(self, **kwargs):
                schema = kwargs["output_schema"]
                if schema.__name__ == "SupervisorDecision":
                    self.calls += 1
                    self.feedback_seen |= "retry_instruction" in kwargs["input_data"]
                    return model_validate(schema, {
                        "decision": "experiment", "rationale": ["隔离验证"],
                        "actions": [{"action_id": "A-CODE-1", "owner_role": "qc_code_optimizer",
                                     "action_class": "isolated_code_experiment",
                                     "objective": "只读质量实验", "acceptance_metrics": ["误修率"],
                                     "priority": 1, "max_attempts": 1,
                                     "auto_execute": self.calls == 2}],
                        "rejected_proposals": [], "global_risk_flags": [],
                        "max_parallel_actions": 1, "confidence": 0.8,
                    })
                if schema.__name__ == "ReviewDecision":
                    return model_validate(schema, {
                        "decision": "accept_experiment", "reasons": ["只读"],
                        "required_validation_metrics": [], "risk_flags": [], "confidence": 0.8,
                    })
                return await super().generate_structured(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manifest = dataset / "issue_manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "modality", "issues"])
                writer.writeheader()
                writer.writerow({"patient": "ANON", "modality": "lge", "issues": "mi_topology"})
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "audit", {})
            for state in (RunState.QUEUED, RunState.RUNNING, RunState.QC_FAILED):
                db.transition(run_id, state)
            db.register_artifact(run_id, None, "lge_issue_manifest", manifest, {"rows": 1})
            provider = AutonomyProvider()
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "unit-test-salt"}), patch(
                "agents.shadow_analysis.ProviderRegistry.get", return_value=provider
            ):
                result = asyncio.run(run_shadow_analysis(
                    db, run_id, PROJECT / "configs/providers.yaml", PROJECT / "configs/agents.yaml"
                ))
            self.assertEqual(provider.calls, 2)
            self.assertTrue(provider.feedback_seen)
            self.assertEqual(result["final_state"], "EXPERIMENT_PROPOSED")
            self.assertEqual(db.run_details(run_id)["maintenance_actions"][0]["status"],
                             "APPROVED")


if __name__ == "__main__":
    unittest.main()
