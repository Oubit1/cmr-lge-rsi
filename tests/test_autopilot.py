from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.database import Database
from app.state_machine import RunState
from agents.governance import agent_action_authorized, implementation_review_approved


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/autopilot.py"
SPEC = importlib.util.spec_from_file_location("autopilot_module", MODULE_PATH)
autopilot = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(autopilot)


class AutopilotTests(unittest.TestCase):
    def test_incomplete_promotion_blocks_every_owner_before_any_execution(self):
        controller = autopilot.Autopilot.__new__(autopilot.Autopilot)
        controller.db = Mock()
        controller.dataset = Path("synthetic")
        controller.state = {}
        controller.event = Mock()
        controller.db.run_details.return_value = {"maintenance_actions": [
            {"action_id": "ready", "status": "APPROVED", "artifact_path": None},
            {"action_id": "broken", "status": "SUCCEEDED", "artifact_path": "synthetic_workspace"},
        ]}
        with patch.object(autopilot, "active_production_blocker", return_value=None), \
             patch.object(autopilot, "dependent_workflow_status", return_value={}), \
             patch.object(autopilot, "enrich_workflow_dependency", return_value={"stage": "lge_myo"}), \
             patch.object(autopilot, "history", return_value=[]), \
             patch.object(autopilot, "reconcile_promotion", return_value={"status": "RECOVERY_REQUIRED"}), \
             patch.object(autopilot, "run_llm_maintenance") as owner:
            result = controller.process_actions(1)
        self.assertTrue(result["recovery_required"])
        owner.assert_not_called()

    def test_recovery_barrier_prevents_next_paid_analysis(self):
        controller = autopilot.Autopilot.__new__(autopilot.Autopilot)
        controller.dataset = Path("synthetic")
        controller.state = {"stagnant_cycles": 0}
        controller.db = Mock()
        controller.event = Mock()
        controller.latest_experiment_run = Mock(return_value={"run_id": 1})
        controller.process_actions = Mock(return_value={"recovery_required": {"status": "RECOVERY_REQUIRED"}})
        controller.analyze_new_snapshot = Mock()
        with patch.object(autopilot, "global_promotion_preflight", return_value={"status": "READY"}), \
             patch.object(autopilot, "issue_fingerprint", return_value=("hash", {})), \
             patch.object(autopilot, "check_iteration_matches_queue", return_value=(True, {})):
            controller.tick()
        controller.analyze_new_snapshot.assert_not_called()

    def test_older_run_promotion_blocks_newer_run_selection(self):
        from app.rsi_runtime import write_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("test", root / "synthetic")
            old_run = db.create_run(dataset_id, "analysis", {})
            new_run = db.create_run(dataset_id, "analysis", {})
            workspace = root / "old_workspace"
            write_record(workspace / "rsi_promotion_journal.json", {
                "run_id": old_run, "action_id": "broken", "artifact": str(root / "runtime/promotions/incomplete")})
            for run, action, artifact in ((old_run, "broken", workspace), (new_run, "ready", None)):
                db.save_maintenance_action(run, {"action_id": action, "owner_role": "qc_code_optimizer",
                                                "action_class": "isolated_code_experiment"}, status="APPROVED")
                if artifact:
                    db.update_maintenance_action(run, action, status="SUCCEEDED", artifact_path=artifact)
            controller = autopilot.Autopilot.__new__(autopilot.Autopilot)
            controller.db, controller.dataset = db, root / "synthetic"
            controller.state, controller.event = {}, Mock()
            controller.latest_experiment_run = Mock(return_value={"run_id": new_run})
            controller.analyze_new_snapshot = Mock()
            controller.process_actions = Mock()
            with patch.object(autopilot, "ROOT", root):
                controller.tick()
            controller.latest_experiment_run.assert_not_called()
            controller.process_actions.assert_not_called()
            controller.analyze_new_snapshot.assert_not_called()

    def test_fingerprint_schema_migration_preserves_pause_without_false_progress(self):
        state = {
            "last_issue_fingerprint": "old-algorithm",
            "paused_issue_fingerprint": "old-algorithm",
            "events": [],
        }
        changed = autopilot.rebase_issue_fingerprint_schema(
            state, "new-algorithm", {"analyzed_manual_edits": 3}
        )
        self.assertTrue(changed)
        self.assertEqual(state["last_issue_fingerprint"], "new-algorithm")
        self.assertEqual(state["paused_issue_fingerprint"], "new-algorithm")
        self.assertEqual(state["issue_fingerprint_schema"],
                         autopilot.ISSUE_FINGERPRINT_SCHEMA)
        self.assertEqual(state["events"][-1]["event"],
                         "ISSUE_FINGERPRINT_SCHEMA_REBASED")
        self.assertTrue(state["events"][-1]["paused_preserved"])
        self.assertFalse(autopilot.rebase_issue_fingerprint_schema(
            state, "another", {"analyzed_manual_edits": 4}
        ))
        self.assertEqual(state["paused_issue_fingerprint"], "new-algorithm")

    def test_production_validation_requires_full_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "orchestrator/app").mkdir(parents=True)
            results = [
                {"command": ["compile"], "returncode": 0, "stdout": "", "stderr": ""},
                {"command": ["full"], "returncode": 1, "stdout": "", "stderr": "failed"},
            ]
            with patch.object(autopilot, "frozen_evaluation_gate", return_value={
                "passed": True, "required": True, "results": [],
            }), patch.object(autopilot, "command", side_effect=results) as run:
                validation = autopilot.validate_workspace(
                    workspace, ["app/check_auto_apply.py"], workspace / "dataset"
                )
            self.assertFalse(validation["passed"])
            self.assertTrue(validation["full_regression_required"])
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[-1].args[0][1:5],
                             ["-m", "unittest", "discover", "-s"])

    def test_full_regression_uses_isolated_skill_compatibility_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill = workspace / "skill"
            (skill / "tools").mkdir(parents=True)
            (skill / "tools/apply_cmr_qc_fast.py").write_text("# snapshot\n")
            self.assertTrue(autopilot.prepare_full_regression_workspace(workspace))
            alias = workspace / "skills/cmr-lge-label-qc-workflow"
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), skill.resolve())

    def test_nonproduction_validation_skips_full_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "orchestrator/experiments").mkdir(parents=True)
            with patch.object(autopilot, "frozen_evaluation_gate", return_value={
                "passed": True, "required": False, "results": [],
            }), patch.object(autopilot, "command", return_value={
                "command": ["compile"], "returncode": 0, "stdout": "", "stderr": "",
            }) as run:
                validation = autopilot.validate_workspace(
                    workspace, ["experiments/probe.py"], workspace / "dataset"
                )
            self.assertTrue(validation["passed"])
            self.assertFalse(validation["full_regression_required"])
            self.assertEqual(run.call_count, 1)

    def test_check_iteration_must_match_live_queue_before_paid_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "runtime/check_iterations/latest.json"
            latest.parent.mkdir(parents=True)
            latest.write_text('{"pending":73,"work":3}')
            current, detail = autopilot.check_iteration_matches_queue(
                root, {"check_待筛": 63, "check_待处理": 3})
            self.assertFalse(current)
            self.assertEqual(detail["expected_pending"], 63)
            latest.write_text('{"pending":63,"work":3}')
            current, _ = autopilot.check_iteration_matches_queue(
                root, {"check_待筛": 63, "check_待处理": 3})
            self.assertTrue(current)

    def test_measured_route_conflict_targets_domain_owned_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "runtime/check_iterations/latest.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"control_gate_status":"PASS","pending_route_conflicts":15}')
            blocker = autopilot.active_production_blocker(root)
            self.assertEqual(blocker["target_path"], "app/check_auto_apply.py")
            payload = {"owner_role": "qc_code_optimizer",
                       "action_class": "isolated_code_experiment",
                       "objective": "patch app/check_auto_apply.py conflict resolver"}
            requirements = autopilot.owner_execution_requirements(payload, blocker)
            self.assertIn("resolve_route_conflict", requirements["required_interface"])
            self.assertIn("holdout exclusion", requirements["required_edit_scope"])

    def test_lge_consumer_action_always_receives_production_interface(self):
        payload = {
            "owner_role": "qc_code_optimizer",
            "action_class": "isolated_code_experiment",
            "workflow_stage": "lge_myo",
            "objective": "Implement scripts/lge_feedback_loop.py production consumer",
        }
        requirements = autopilot.owner_execution_requirements(payload, None)
        self.assertEqual(requirements["required_target_path"],
                         "scripts/lge_feedback_loop.py")
        self.assertEqual(requirements["required_interface"],
                         "run_evaluation(data: dict) -> dict")
        self.assertEqual(requirements["required_gate"],
                         "lge_stage_consumer_contract")

    def test_matching_deferred_blocker_is_quarantined_only_for_same_issue_set(self):
        blocker = {"target_path": "app/check_auto_apply.py"}
        state = {
            "last_issue_fingerprint": "same",
            "deferred_production_blocker": {
                "target_path": "app/check_auto_apply.py",
                "issue_fingerprint": "same",
            },
        }
        self.assertTrue(autopilot.production_blocker_deferred(state, blocker))
        state["last_issue_fingerprint"] = "changed"
        self.assertFalse(autopilot.production_blocker_deferred(state, blocker))

    def test_route_conflict_gate_requires_reduction_and_clean_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "orchestrator/app").mkdir(parents=True)
            dataset = workspace / "dataset"
            dataset.mkdir()
            latest = workspace / "runtime/check_iterations/latest.json"
            latest.parent.mkdir(parents=True)
            latest.write_text('{"pending_route_conflicts":15}')
            preview = {
                "eligible": 3, "holdout_excluded": 58, "validation_excluded": 20,
                "route_conflicts_detected": 15, "route_conflicts_resolved": 3,
                "route_conflicts_unresolved": 12,
                "planned": [{"conflict_resolution": True}],
                "preflight": [{"preflight": "PASS"}],
            }
            command_result = {"returncode": 0, "stdout": __import__("json").dumps(preview),
                              "stderr": ""}
            with patch.object(autopilot, "ROOT", workspace), \
                 patch.object(autopilot, "command", return_value=command_result):
                result = autopilot.frozen_evaluation_gate(
                    workspace, ["app/check_auto_apply.py"], dataset)
            self.assertTrue(result["passed"])
            self.assertTrue(autopilot.named_production_contract_passed({
                "passed": True, "promotion_scope": "production_path",
                "frozen_evaluation_gate": result,
            }))

    def test_measured_spatial_blocker_supersedes_unrelated_shadow_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "runtime/check_iterations/latest.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"control_gate_status":"FAIL","checked_false_routes":3,'
                              '"spatial_boundary_candidates":138}')
            blocker = autopilot.active_production_blocker(root)
            self.assertIsNotNone(blocker)
            self.assertFalse(autopilot.action_addresses_blocker({
                "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment",
                "objective": "another measurement-only area-ratio shadow",
            }, blocker))
            self.assertTrue(autopilot.action_addresses_blocker({
                "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment",
                "objective": "patch scripts/shadow_spatial_boundary.py in snapshot",
            }, blocker))
            self.assertFalse(autopilot.changed_files_include_target(
                ["experiments/spatial_probe.py", "tests/test_spatial_probe.py"],
                blocker["target_path"]))
            self.assertTrue(autopilot.changed_files_include_target(
                ["orchestrator/scripts/shadow_spatial_boundary.py"],
                blocker["target_path"]))
            requirements = autopilot.owner_execution_requirements({
                "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment",
                "objective": "patch scripts/shadow_spatial_boundary.py in snapshot",
            }, blocker)
            self.assertEqual(requirements["required_target_path"],
                             "scripts/shadow_spatial_boundary.py")
            self.assertEqual(requirements["failure_if_omitted"],
                             "ACTION_MISSED_PRODUCTION_TARGET")
            self.assertIn("--dataset", requirements["required_interface"])
            self.assertIn("DATASET ROOT", requirements["required_interface"])
            self.assertIn("checked_false_routes", requirements["required_interface"])
            self.assertIn("ring_fraction", requirements["required_edit_scope"])
            self.assertIn("MINIMAL", requirements["required_edit_scope"])
            self.assertTrue(autopilot.owner_result_ready_for_validation({
                "ready_for_validation": False,
                "changed_files": [
                    "scripts/shadow_spatial_boundary.py",
                    "tests/test_shadow_spatial_boundary.py",
                ],
                "tests_run": ["python -m unittest tests.test_shadow_spatial_boundary"],
            }, requirements))
            self.assertFalse(autopilot.owner_result_ready_for_validation({
                "ready_for_validation": False,
                "changed_files": ["scripts/shadow_spatial_boundary.py"],
                "tests_run": [],
            }, requirements))

    def test_shadow_only_scope_cannot_include_routing_or_traversal(self):
        self.assertTrue(autopilot.shadow_only_files(
            ["orchestrator/experiments/probe.py", "orchestrator/tests/test_probe.py"]))
        self.assertFalse(autopilot.shadow_only_files([]))
        self.assertFalse(autopilot.shadow_only_files(["orchestrator/scripts/rules_v2.py"]))
        self.assertFalse(autopilot.shadow_only_files(["orchestrator/experiments/../app/rule.py"]))

    def test_spatial_production_gate_requires_zero_checked_false_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "orchestrator/scripts").mkdir(parents=True)
            dataset = workspace / "dataset"
            dataset.mkdir()
            passed_report = {
                "label_mutations": 0, "checked_false_routes": 0,
                "control_gate": "PASS", "pending_spatial_boundary_candidates": 12,
                "records": [{"private": "not returned"}],
            }
            with patch.object(autopilot, "command", return_value={"returncode": 0}), \
                 patch.object(autopilot, "read_json", return_value=passed_report):
                result = autopilot.frozen_evaluation_gate(
                    workspace, ["scripts/shadow_spatial_boundary.py"], dataset)
            self.assertTrue(result["passed"])
            self.assertNotIn("records", result["results"][0]["report"])
            failed_report = {**passed_report, "checked_false_routes": 1,
                             "control_gate": "FAIL"}
            with patch.object(autopilot, "command", return_value={"returncode": 0}), \
                 patch.object(autopilot, "read_json", return_value=failed_report):
                result = autopilot.frozen_evaluation_gate(
                    workspace, ["scripts/shadow_spatial_boundary.py"], dataset)
            self.assertFalse(result["passed"])

    def test_reviewed_work_to_trash_changes_progress_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            work = dataset / "labels/check/待处理/cmr/PATIENT/slice.png"
            work.parent.mkdir(parents=True)
            work.write_bytes(b"synthetic")
            before, old_counts = autopilot.issue_fingerprint(dataset)
            trash = dataset / "labels/check/垃圾桶/cmr/PATIENT/slice.png"
            trash.parent.mkdir(parents=True)
            work.rename(trash)
            after, new_counts = autopilot.issue_fingerprint(dataset)
            self.assertNotEqual(before, after)
            self.assertEqual(old_counts["check_待处理"], 1)
            self.assertEqual(new_counts["check_待处理"], 0)
            self.assertEqual(new_counts["check_垃圾桶"], 1)
            self.assertEqual(autopilot.historical_no_gain_cycles(
                [{"event": "NEXT_ITERATION", "issue_counts": old_counts}],
                new_counts, 3), 0)

    def test_framework_fix_resume_audits_and_resets_only_old_stagnation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = object.__new__(autopilot.Autopilot)
            pilot.dataset = root
            pilot.state_path = root / "state.json"
            counts = {"check_待筛": 185}
            pilot.state = {"status": "NO_PROGRESS_PAUSED", "stagnant_cycles": 3,
                           "paused_issue_fingerprint": "same", "last_code_fingerprint": "old",
                           "events": [{"event": "NEXT_ITERATION", "issue_counts": counts}] * 3}
            with patch.object(autopilot, "issue_fingerprint", return_value=("same", counts)), patch.object(
                autopilot.subprocess, "check_output", return_value="new\n"
            ):
                pilot.resume_after_fix()
            self.assertEqual(pilot.state["stagnant_cycles"], 0)
            self.assertNotIn("paused_issue_fingerprint", pilot.state)
            self.assertEqual(pilot.state["last_code_fingerprint"], "new")
            self.assertEqual(autopilot.historical_no_gain_cycles(
                pilot.state["events"], counts, 3), 0)
            self.assertEqual(pilot.state["status"], "FRAMEWORK_FIX_RESUMED")

    def test_framework_fix_resume_survives_graceful_stop_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = object.__new__(autopilot.Autopilot)
            pilot.dataset = root
            pilot.state_path = root / "state.json"
            pilot.state = {
                "status": "CONTROLLER_STOPPED_AFTER_TICK",
                "stagnant_cycles": 3,
                "paused_issue_fingerprint": "same",
                "last_code_fingerprint": "old",
                "events": [
                    {"event": "ESCALATION_REQUIRED"},
                    {"event": "CONTROLLER_STOPPED_AFTER_TICK"},
                ],
            }
            with patch.object(autopilot, "issue_fingerprint",
                              return_value=("same", {"check_待筛": 156})), patch.object(
                autopilot.subprocess, "check_output", return_value="new\n"
            ):
                pilot.resume_after_fix()
            self.assertEqual(pilot.state["status"], "FRAMEWORK_FIX_RESUMED")
            self.assertEqual(pilot.state["stagnant_cycles"], 0)
            self.assertNotIn("paused_issue_fingerprint", pilot.state)
            self.assertEqual(pilot.state["events"][-1]["previous_status"],
                             "ESCALATION_REQUIRED")

    def test_review_only_round_is_accounted_before_new_paid_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            db.save_maintenance_action(run_id, {
                "action_id": "A1", "owner_role": "rule_curator",
                "action_class": "human_review", "objective": "review-only",
                "auto_execute": False,
            }, status="HUMAN_REQUIRED")
            db.transition(run_id, RunState.EVIDENCE_READY)
            db.transition(run_id, RunState.NEEDS_REVIEW,
                          current_step="shadow_analysis_complete")
            pilot = object.__new__(autopilot.Autopilot)
            pilot.db = db
            pilot.dataset = dataset.resolve()
            self.assertEqual(pilot.latest_experiment_run()["run_id"], run_id)

    def test_newest_accepted_manual_experiment_is_not_skipped_for_old_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)
            old = db.create_run(dataset_id, "analysis", {})
            db.transition(old, RunState.EVIDENCE_READY)
            db.transition(old, RunState.NEEDS_REVIEW)
            newest = db.create_run(dataset_id, "analysis", {})
            db.save_maintenance_action(newest, {
                "action_id": "A2", "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment", "objective": "sandbox only",
                "auto_execute": False,
            }, status="HUMAN_REQUIRED")
            db.transition(newest, RunState.EVIDENCE_READY)
            db.transition(newest, RunState.FAILURE_ANALYSIS)
            db.transition(newest, RunState.EXPERIMENT_PROPOSED)
            pilot = object.__new__(autopilot.Autopilot)
            pilot.db = db
            pilot.dataset = dataset.resolve()
            self.assertEqual(pilot.latest_experiment_run()["run_id"], newest)

    def test_operator_released_action_preempts_newer_ordinary_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            dataset_id = db.register_dataset("demo", dataset)

            def add_run(action_id: str, *, human_released: bool) -> int:
                run_id = db.create_run(dataset_id, "analysis", {})
                payload = {
                    "action_id": action_id,
                    "owner_role": "qc_code_optimizer",
                    "action_class": "isolated_code_experiment",
                    "objective": "bounded sandbox task",
                    "acceptance_metrics": ["bounded"],
                    "max_attempts": 2,
                    "auto_execute": True,
                    "review_approved": True,
                }
                db.save_maintenance_action(run_id, payload, status="APPROVED")
                if human_released:
                    db.attach_operator_evidence_retry(
                        run_id, action_id, edit_id="edit-1", pool="cmr_middle_manifest"
                    )
                thread = f"run-{run_id}:{action_id}"
                db.post_agent_message(
                    run_id, thread_id=thread, sender_role="supervisor",
                    recipient_role="qc_code_optimizer", message_type="assigned_action",
                    payload={"action_id": action_id},
                )
                db.post_agent_message(
                    run_id, thread_id=thread, sender_role="independent_reviewer",
                    recipient_role="qc_code_optimizer", message_type="action_review",
                    payload={"decision": "accept_experiment"},
                )
                db.transition(run_id, RunState.EVIDENCE_READY)
                db.transition(run_id, RunState.FAILURE_ANALYSIS)
                db.transition(run_id, RunState.EXPERIMENT_PROPOSED)
                return run_id

            older = add_run("A-human", human_released=True)
            add_run("A-newer", human_released=False)
            pilot = object.__new__(autopilot.Autopilot)
            pilot.db = db
            pilot.dataset = dataset.resolve()
            self.assertEqual(pilot.latest_experiment_run()["run_id"], older)

    def test_only_unconsumed_structured_operator_evidence_bypasses_blocker(self):
        blocker = {"target_path": "app/check_auto_apply.py",
                   "owner_role": "qc_code_optimizer"}
        ordinary = {"human_retry_history": [{"reviewer": "operator"}]}
        permitted = {"operator_evidence_retry": {
            "edit_id": "edit-1", "pool": "cmr_middle_manifest", "consumed": False,
        }}
        consumed = {"operator_evidence_retry": {
            "edit_id": "edit-1", "pool": "cmr_middle_manifest", "consumed": True,
        }}
        self.assertTrue(autopilot.blocker_supersedes_action(ordinary, blocker))
        self.assertFalse(autopilot.blocker_supersedes_action(permitted, blocker))
        self.assertTrue(autopilot.blocker_supersedes_action(consumed, blocker))

    def test_workflow_stage_rejects_early_cmr_and_stale_lge_actions(self):
        cmr = {"action_id": "S841_cmr_small_area_guard_delta_scan_exec"}
        lge_myo = {"action_id": "A-lge", "workflow_stage": "lge_myo"}
        self.assertEqual(autopilot.semantic_action_workflow_stage(cmr), "cmr_paired")
        self.assertFalse(autopilot.action_matches_workflow_stage(cmr, "lge_myo"))
        self.assertTrue(autopilot.action_matches_workflow_stage(lge_myo, "lge_myo"))
        self.assertFalse(autopilot.action_matches_workflow_stage(lge_myo, "lge_mi"))
        self.assertFalse(autopilot.action_matches_workflow_stage(
            {"action_id": "A-cmr", "objective": "CMR scan",
             "workflow_stage": "lge_myo"}, "lge_myo"
        ))
        self.assertTrue(autopilot.action_matches_workflow_stage(
            {"action_id": "generic-control-plane-audit"}, "lge_myo"
        ))
        self.assertTrue(autopilot.action_uses_stale_lge_shadow_contract({
            "objective": "consume 9 non-holdout LGE cases",
            "acceptance_metrics": ["holdout_excluded_count==48"],
        }))
        self.assertFalse(autopilot.action_uses_stale_lge_shadow_contract({
            "objective": "reconcile to the supplied active-stage input length",
            "addresses_metric": "pending+working+resolved==9 on 9 non-holdout volumes",
        }))

    def test_implementation_review_cannot_promote_with_validation_debt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            thread = f"run-{run_id}:A2"
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="independent_reviewer",
                recipient_role="rule_curator", message_type="implementation_review",
                payload={"decision": "accept_experiment",
                         "required_validation_metrics": ["Held-out false-route rate"]},
            )
            self.assertFalse(implementation_review_approved(db, run_id, "A2", "rule_curator"))
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="independent_reviewer",
                recipient_role="rule_curator", message_type="implementation_review",
                payload={"decision": "accept_experiment", "required_validation_metrics": []},
            )
            self.assertTrue(implementation_review_approved(db, run_id, "A2", "rule_curator"))

    def test_explicit_paused_analysis_keeps_fuse_and_never_executes_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state.json"
            pilot = Mock()
            pilot.dataset = root
            pilot.state_path = state_path
            pilot.state = {"status": "ESCALATION_REQUIRED",
                           "paused_issue_fingerprint": "same", "escalated_for": 734}
            pilot.analyze_new_snapshot.return_value = 735
            args = SimpleNamespace(resume_paused_analysis_once=True)
            old_root = autopilot.ROOT
            try:
                autopilot.ROOT = root
                with patch.object(autopilot, "parse_args", return_value=args), \
                     patch.object(autopilot, "Autopilot", return_value=pilot), \
                     patch.object(autopilot, "issue_fingerprint", return_value=("same", {})):
                    autopilot.main()
            finally:
                autopilot.ROOT = old_root
            pilot.analyze_new_snapshot.assert_called_once_with(734)
            pilot.process_actions.assert_not_called()
            self.assertEqual(pilot.state["status"], "ESCALATION_REQUIRED")
            self.assertEqual(pilot.state["paused_issue_fingerprint"], "same")

    def test_paused_analysis_requires_matching_fingerprint_and_source(self):
        state = {"status": "ESCALATION_REQUIRED",
                 "paused_issue_fingerprint": "old", "escalated_for": 734}
        self.assertEqual(autopilot.paused_analysis_source(state, "old"), 734)
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            autopilot.paused_analysis_source(state, "new")
        with self.assertRaisesRegex(ValueError, "missing escalated source"):
            autopilot.paused_analysis_source({**state, "escalated_for": None}, "old")

    def test_history_carries_independent_reviewer_validation_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            action = {
                "action_id": "A2", "owner_role": "rule_curator",
                "action_class": "rule_draft", "objective": "small-area shadow rule",
                "acceptance_metrics": ["shadow concordance"], "priority": 1,
                "max_attempts": 1, "auto_execute": True,
            }
            db.save_maintenance_action(run_id, action, status="PROMOTED")
            db.post_agent_message(
                run_id, thread_id=f"run-{run_id}:A2",
                sender_role="independent_reviewer", recipient_role="rule_curator",
                message_type="implementation_review",
                payload={"decision": "accept_experiment",
                         "reasons": ["Code passed tests; no real-data evaluation"],
                         "required_validation_metrics": ["Held-out shadow concordance"]},
            )
            latest = autopilot.history(db, dataset)[0]["actions"][0]
            self.assertEqual(latest["status"], "PROMOTED")
            self.assertEqual(latest["review_feedback"]["decision"], "accept_experiment")
            self.assertEqual(latest["review_feedback"]["required_validation_metrics"],
                             ["Held-out shadow concordance"])

    def test_history_carries_pre_action_rejection_into_next_backlog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            db.save_maintenance_action(run_id, {
                "action_id": "A2", "owner_role": "rule_curator",
                "action_class": "rule_draft", "objective": "draft",
            }, status="HUMAN_REQUIRED")
            db.post_agent_message(
                run_id, thread_id=f"run-{run_id}:A2",
                sender_role="independent_reviewer", recipient_role="rule_curator",
                message_type="action_review",
                payload={"decision": "needs_human_review",
                         "reasons": ["No held-out evidence"],
                         "required_validation_metrics": ["Held-out false-route rate"]},
            )
            latest = autopilot.history(db, dataset)[0]["actions"][0]
            self.assertEqual(latest["review_feedback"]["decision"], "needs_human_review")
            self.assertEqual(latest["review_feedback"]["required_validation_metrics"],
                             ["Held-out false-route rate"])

    def test_history_carries_failed_frozen_validation_into_next_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            workspace = root / "runtime/maintenance/run_000007/A7"
            workspace.mkdir(parents=True)
            (workspace / "maintenance_manifest.json").write_text(
                json.dumps({"result": {"ready_for_validation": True}}), encoding="utf-8")
            (workspace / "autopilot_validation.json").write_text(json.dumps({
                "passed": False, "targeted_tests": 1, "promotion_scope": "production_path",
                "frozen_evaluation_gate": {"passed": False, "results": [{
                    "suite": "route_conflict_dry_run", "passed": False,
                    "checks": {"at_least_one_conflict_resolved": False},
                    "report": {"route_conflicts_detected": 5,
                               "route_conflicts_resolved": 0,
                               "route_conflicts_unresolved": 5},
                }]},
            }), encoding="utf-8")
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            db.save_maintenance_action(run_id, {
                "action_id": "A7", "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment", "objective": "tie break",
            }, status="APPROVED")
            db.update_maintenance_action(run_id, "A7", status="REJECTED",
                                         artifact_path=workspace)
            old_root = autopilot.ROOT
            try:
                autopilot.ROOT = root
                latest = autopilot.history(db, dataset)[0]["actions"][0]
            finally:
                autopilot.ROOT = old_root
            result = latest["validation_result"]
            self.assertFalse(result["passed"])
            self.assertFalse(result["frozen_gate_passed"])
            self.assertEqual(result["suites"][0]["report"]["route_conflicts_resolved"], 0)
            self.assertEqual(result["suites"][0]["report"]["route_conflicts_unresolved"], 5)

    def test_history_explains_immutable_lge_consumer_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            workspace = root / "runtime/maintenance/run_000001/A1"
            workspace.mkdir(parents=True)
            (workspace / "autopilot_validation.json").write_text(json.dumps({
                "passed": False,
                "frozen_evaluation_gate": {"passed": False, "results": [{
                    "suite": "lge_stage_consumer_contract", "passed": False,
                    "checks": {"holdout_contract_exact": False}, "report": {},
                }]},
            }), encoding="utf-8")
            db.save_maintenance_action(run_id, {
                "action_id": "A1", "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment", "objective": "consumer",
            }, status="APPROVED")
            db.update_maintenance_action(run_id, "A1", status="REJECTED",
                                         artifact_path=workspace)
            old_root = autopilot.ROOT
            try:
                autopilot.ROOT = root
                suite = autopilot.history(db, dataset)[0]["actions"][0][
                    "validation_result"]["suites"][0]
            finally:
                autopilot.ROOT = old_root
            self.assertIn("frozen_unavailable", suite["immutable_contract"])
            self.assertIn("pure_adapter", suite["immutable_contract"])

    def test_history_compacts_large_owner_report_without_case_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            workspace = root / "runtime/maintenance/run_000008/A8"
            workspace.mkdir(parents=True)
            large_report = {
                "volumes_scanned": 13,
                "suppression_candidate_count": 4,
                "component_count_distribution": {"1": 9, "2": 4},
                "cases": {f"case-{index}": {"value": "x" * 1000}
                          for index in range(20)},
                "rollback_command": "must not enter model context",
            }
            (workspace / "maintenance_manifest.json").write_text(json.dumps({
                "result": {"real_evaluation": {
                    "status": "EXECUTED", "input_records": 13,
                    "source_sha256": "source", "report_sha256": "report",
                    "input_modality": "LGE", "input_cohort": "pending",
                    "reference_labels_available": False,
                    "report": large_report,
                }}
            }), encoding="utf-8")
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            db.save_maintenance_action(run_id, {
                "action_id": "A8", "owner_role": "qc_code_optimizer",
                "action_class": "isolated_code_experiment", "objective": "scan",
            }, status="PROMOTED_SHADOW")
            db.update_maintenance_action(run_id, "A8", status="PROMOTED_SHADOW",
                                         artifact_path=workspace)
            old_root = autopilot.ROOT
            try:
                autopilot.ROOT = root
                measured = autopilot.history(db, dataset)[0]["actions"][0]["measured_result"]
            finally:
                autopilot.ROOT = old_root
            summary = measured["owner_report_summary"]
            self.assertEqual(summary["volumes_scanned"], 13)
            self.assertEqual(summary["suppression_candidate_count"], 4)
            self.assertEqual(summary["component_count_distribution"], {"1": 9, "2": 4})
            self.assertTrue(summary["per_case_payload_omitted"])
            self.assertNotIn("cases", summary)
            self.assertNotIn("rollback_command", summary)

    def test_action_requires_supervisor_and_independent_reviewer_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            db = Database(root / "workflow.sqlite")
            db.initialize()
            run_id = db.create_run(db.register_dataset("demo", dataset), "analysis", {})
            action = {
                "action_id": "A1", "owner_role": "rule_curator",
                "action_class": "rule_draft", "objective": "draft",
                "acceptance_metrics": ["gate"], "priority": 1,
                "max_attempts": 1, "auto_execute": True,
                "review_approved": True,
            }
            db.save_maintenance_action(run_id, action, status="APPROVED")
            row = db.run_details(run_id)["maintenance_actions"][0]
            self.assertFalse(agent_action_authorized(db, run_id, row, action))
            thread = f"run-{run_id}:A1"
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="supervisor",
                recipient_role="rule_curator", message_type="assigned_action",
                payload=action,
            )
            self.assertFalse(agent_action_authorized(db, run_id, row, action))
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="independent_reviewer",
                recipient_role="rule_curator", message_type="action_review",
                payload={"decision": "reject"},
            )
            self.assertFalse(agent_action_authorized(db, run_id, row, action))
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="independent_reviewer",
                recipient_role="rule_curator", message_type="action_review",
                payload={"decision": "accept_experiment"},
            )
            self.assertTrue(agent_action_authorized(db, run_id, row, action))
            db.post_agent_message(
                run_id, thread_id=thread, sender_role="independent_reviewer",
                recipient_role="rule_curator", message_type="action_review",
                payload={"decision": "reject"},
            )
            self.assertFalse(agent_action_authorized(db, run_id, row, action))

    def test_recovers_historical_no_gain_cycles(self):
        counts = {"cmr_middle_manifest": 12}
        events = [
            {"event": "NEXT_ITERATION", "issue_counts": counts},
            {"event": "ACTION_STARTED"},
            {"event": "NEXT_ITERATION", "issue_counts": counts},
            {"event": "NEXT_ITERATION", "issue_counts": counts},
        ]
        self.assertEqual(autopilot.historical_no_gain_cycles(events, counts, 3), 3)
        self.assertEqual(
            autopilot.historical_no_gain_cycles(events, {"cmr_middle_manifest": 11}, 3), 0
        )

    def test_once_never_schedules_a_followup_analysis(self):
        self.assertEqual(
            autopilot.next_iteration_decision(
                auto_done=True, stagnant_cycles=0, max_stagnant=3, once=True
            ),
            "one_shot_complete",
        )
        self.assertEqual(
            autopilot.next_iteration_decision(
                auto_done=True, stagnant_cycles=0, max_stagnant=3, once=False
            ),
            "next_iteration",
        )
        self.assertEqual(
            autopilot.next_iteration_decision(
                auto_done=True, stagnant_cycles=3, max_stagnant=3, once=True
            ),
            "escalate",
        )

    def test_production_failure_signature_ignores_workspace_paths(self):
        validation = {
            "passed": False,
            "promotion_scope": "production_path",
            "frozen_evaluation_gate": {
                "results": [{
                    "suite": "spatial_boundary_checked_controls",
                    "passed": False,
                    "checks": {"control_gate_pass": False},
                    "command": {
                        "returncode": 1,
                        "stderr": "Traceback in /tmp/run_772/file.py\nTypeError: mixed values\n",
                    },
                }],
            },
        }
        first = autopilot.production_failure_signature(validation)
        validation["frozen_evaluation_gate"]["results"][0]["command"]["stderr"] = (
            "Traceback in /tmp/run_773/other.py\nTypeError: mixed values\n"
        )
        self.assertEqual(first, autopilot.production_failure_signature(validation))
        validation["frozen_evaluation_gate"]["results"][0]["command"]["stderr"] = (
            "Traceback\nValueError: malformed record\n"
        )
        self.assertNotEqual(first, autopilot.production_failure_signature(validation))

    def test_nonproduction_failure_has_no_repair_signature(self):
        self.assertIsNone(autopilot.production_failure_signature({
            "passed": False,
            "promotion_scope": "shadow_only",
        }))

    def test_named_production_contract_treats_monitoring_debt_as_nonblocking(self):
        validation = {
            "passed": True,
            "promotion_scope": "production_path",
            "frozen_evaluation_gate": {"results": [{
                "suite": "spatial_boundary_checked_controls",
                "passed": True,
                "checks": {
                    "zero_label_mutations": True,
                    "checked_false_routes_eq_0": True,
                    "control_gate_pass": True,
                    "nonempty_pending_coverage": True,
                },
            }]},
        }
        self.assertTrue(autopilot.named_production_contract_passed(validation))
        validation["frozen_evaluation_gate"]["results"][0]["checks"][
            "checked_false_routes_eq_0"
        ] = False
        self.assertFalse(autopilot.named_production_contract_passed(validation))

    def test_named_lge_consumer_contract_is_a_production_contract(self):
        validation = {
            "passed": True,
            "promotion_scope": "production_path",
            "frozen_evaluation_gate": {"results": [{
                "suite": "lge_stage_consumer_contract",
                "passed": True,
                "checks": {
                    "lge_input": True,
                    "stage_counts_present": True,
                    "counts_reconcile": True,
                    "consumer_ready_boolean": True,
                    "report_modality_exact": True,
                    "report_stage_exact": True,
                    "report_source_hash_exact": True,
                    "holdout_contract_exact": True,
                    "no_embedded_case_ids": True,
                    "zero_label_writes": True,
                },
            }]},
        }
        self.assertTrue(autopilot.named_production_contract_passed(validation))
        validation["frozen_evaluation_gate"]["results"][0]["checks"][
            "counts_reconcile"
        ] = False
        self.assertFalse(autopilot.named_production_contract_passed(validation))

    def test_known_tool_never_enters_snapshot_maintenance_executor(self):
        self.assertFalse(autopilot.maintenance_action_executable("known_tool"))
        self.assertFalse(autopilot.maintenance_action_executable("human_review"))
        self.assertTrue(autopilot.maintenance_action_executable("rule_draft"))

    def test_action_attempts_counts_started_request_but_ignores_bare_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_root = autopilot.ROOT
            try:
                autopilot.ROOT = Path(tmp)
                action_root = Path(tmp) / "runtime/maintenance/run_000007/A1_20261021"
                action_root.mkdir(parents=True)
                self.assertEqual(autopilot.action_attempts(7, "A1"), 0)
                (action_root / "maintenance_request.json").write_text("{}\n", encoding="utf-8")
                self.assertEqual(autopilot.action_attempts(7, "A1"), 1)
                (action_root / "maintenance_manifest.json").write_text("{}\n", encoding="utf-8")
                self.assertEqual(autopilot.action_attempts(7, "A1"), 1)
                (action_root / "codex_events.jsonl").write_text(
                    '{"type":"error","message":"usage limit"}\n', encoding="utf-8"
                )
                self.assertEqual(autopilot.action_attempts(7, "A1"), 1)
            finally:
                autopilot.ROOT = original_root

    def test_issue_fingerprint_is_deterministic_and_change_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "labels/check/manifest_backup/cmr_middle_manifest.csv"
            target.parent.mkdir(parents=True)
            target.write_text("patient,issues\na,x\n", encoding="utf-8")
            first, counts = autopilot.issue_fingerprint(root)
            second, _ = autopilot.issue_fingerprint(root)
            self.assertEqual(first, second)
            self.assertEqual(counts["cmr_middle_manifest"], 1)
            target.write_text("patient,issues\na,x\nb,y\n", encoding="utf-8")
            changed, _ = autopilot.issue_fingerprint(root)
            self.assertNotEqual(first, changed)

    def test_applied_manual_edit_changes_fingerprint_but_staged_edit_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            manual = root / "runtime/manual_annotations"
            edit_id = "edit_001"
            edit_dir = manual / "edits" / edit_id
            edit_dir.mkdir(parents=True)
            (manual / "analyzed_training_corpus.jsonl").write_text(
                json.dumps({
                    "edit_id": edit_id,
                    "modality": "cmr",
                    "target": "myo",
                    "operation": "mixed",
                    "suggested_pattern": "ring_gap",
                    "analysis": {"reusable_casebook_pattern": "repair a ring gap"},
                }) + "\n",
                encoding="utf-8",
            )
            provenance = {
                "edit_id": edit_id,
                "status": "STAGED",
                "applied_by": None,
                "applied_hash": None,
            }
            (edit_dir / "provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            before, counts = autopilot.issue_fingerprint(dataset, root)
            self.assertEqual(counts["analyzed_manual_edits"], 0)

            provenance.update({
                "status": "APPLIED",
                "applied_by": "operator",
                "applied_hash": "sha256:confirmed",
            })
            (edit_dir / "provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            after, counts = autopilot.issue_fingerprint(dataset, root)
            self.assertNotEqual(before, after)
            self.assertEqual(counts["analyzed_manual_edits"], 1)

    def test_review_state_moves_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = root / "labels/check/待筛/cmr/CASE/1.png"
            pending.parent.mkdir(parents=True)
            pending.write_bytes(b"review")
            before, counts = autopilot.issue_fingerprint(root)
            self.assertEqual(counts["check_待筛"], 1)
            checked = root / "labels/check/checked/cmr/CASE/1.png"
            checked.parent.mkdir(parents=True)
            pending.rename(checked)
            after, counts = autopilot.issue_fingerprint(root)
            self.assertNotEqual(before, after)
            self.assertEqual(counts["check_checked"], 1)

    def test_resolve_candidate_never_accepts_dataset_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "orchestrator/experiments/demo.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            resolved = autopilot.resolve_candidate(workspace, "experiments/demo.py")
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved[2], "orchestrator")
            self.assertIsNone(autopilot.resolve_candidate(workspace, "labels/demo.nii.gz"))


if __name__ == "__main__":
    unittest.main()
