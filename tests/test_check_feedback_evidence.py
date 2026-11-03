from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.evidence import (approved_manual_edit_records, approved_manual_review_records,
                             check_feedback_summary, check_iteration_evidence,
                             dependent_workflow_status,
                             enrich_workflow_dependency,
                             safe_review_feedback, validation_backlog,
                             build_evidence_package)


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/check_feedback_loop.py"
SPEC = importlib.util.spec_from_file_location("check_feedback_loop_module", MODULE_PATH)
check_feedback_loop = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(check_feedback_loop)


class CheckFeedbackEvidenceTests(unittest.TestCase):
    def test_two_stage_shadows_require_consumer_integration(self):
        base = {"stage": "lge_myo", "lge_myo_consumer_ready": False,
                "lge_mi_consumer_ready": False}
        history = [{"actions": [{
            "action_id": "A-lge-myo-one", "status": "PROMOTED_SHADOW",
            "workflow_stage": "lge_myo",
        }]}, {"actions": [{
            "action_id": "A-lge-myo-two", "status": "PROMOTED_SHADOW",
            "workflow_stage": "lge_myo",
        }]}]
        enriched = enrich_workflow_dependency(base, history)
        self.assertTrue(enriched["integration_required"])
        self.assertEqual(enriched["shadow_evidence_count"], 2)
        self.assertIn("latest.json", enriched["required_output"])

    def test_dependency_stage_advances_only_on_persisted_zeroes_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            self.assertEqual(dependent_workflow_status(dataset, root)["stage"], "lge_myo")
            report = root / "runtime/lge_iterations/latest.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({
                "lge_myo_consumer_ready": True, "lge_myo_pending": 0, "lge_myo_work": 0,
                "lge_mi_consumer_ready": True, "lge_mi_pending": 3, "lge_mi_work": 0,
            }))
            self.assertEqual(dependent_workflow_status(dataset, root)["stage"], "lge_mi")
            report.write_text(json.dumps({
                "lge_myo_consumer_ready": True, "lge_myo_pending": 0, "lge_myo_work": 0,
                "lge_mi_consumer_ready": True, "lge_mi_pending": 0, "lge_mi_work": 0,
                "pairing_ready": False, "paired_slice_count": 0,
            }))
            self.assertEqual(
                dependent_workflow_status(dataset, root)["stage"], "paired_selection"
            )
            report.write_text(json.dumps({
                "lge_myo_consumer_ready": True, "lge_myo_pending": 0, "lge_myo_work": 0,
                "lge_mi_consumer_ready": True, "lge_mi_pending": 0, "lge_mi_work": 0,
                "pairing_ready": True, "paired_slice_count": 42,
            }))
            self.assertEqual(dependent_workflow_status(dataset, root)["stage"], "cmr_paired")

    def test_auto_apply_reloads_promoted_module_before_each_tick(self):
        stale_run = Mock()
        fresh_run = Mock(return_value={"applied": []})
        stale = SimpleNamespace(run=stale_run)
        fresh = SimpleNamespace(run=fresh_run)
        with patch.object(check_feedback_loop.importlib, "import_module",
                          return_value=stale) as imported, patch.object(
            check_feedback_loop.importlib, "reload", return_value=fresh
        ) as reloaded:
            result = check_feedback_loop.run_current_auto_apply(
                Path("dataset"), Path("runtime"), execute=True, limit=7
            )
        self.assertEqual([('app.quality_scoring',), ('app.check_auto_apply',)],
                         [item.args for item in imported.call_args_list])
        self.assertEqual(2, reloaded.call_count)
        reloaded.assert_called_with(stale)
        stale_run.assert_not_called()
        fresh_run.assert_called_once_with(
            Path("dataset"), Path("runtime"), execute=True, limit=7
        )
        self.assertEqual(result, {"applied": []})

    def test_reviewer_feedback_is_bounded_and_drops_identifiers(self):
        feedback = safe_review_feedback({
            "decision": "needs_human_review",
            "reasons": ["No held-out precision reported", "/data1/private/CASE_001", "x" * 300],
            "required_validation_metrics": ["Measure held-out precision", "name@example.org"],
        })
        self.assertEqual(feedback, {
            "decision": "needs_human_review",
            "reasons": ["No held-out precision reported"],
            "required_validation_metrics": ["Measure held-out precision"],
        })

    def test_validation_backlog_does_not_count_code_as_resolved(self):
        history = [{"run_id": 734, "actions": [{
            "action_id": "A2", "owner_role": "rule_curator", "status": "PROMOTED",
            "review_feedback": {"decision": "accept_experiment",
                                "required_validation_metrics": ["Shadow concordance"]},
        }]}]
        self.assertEqual(validation_backlog(history), [{
            "run_id": 734, "action_id": "A2", "owner_role": "rule_curator",
            "status": "PROMOTED", "review_decision": "accept_experiment",
            "required_validation_metrics": ["Shadow concordance"],
        }])

    def test_next_agent_evidence_contains_reviewer_backlog(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"CMR_LGE_CASE_SALT": "test-salt"}
        ):
            history = [{"run_id": 734, "actions": [{
                "action_id": "A2", "owner_role": "rule_curator", "status": "PROMOTED",
                "review_feedback": {"decision": "accept_experiment",
                                    "required_validation_metrics": ["Held-out shadow concordance"]},
            }]}]
            package = build_evidence_package({
                "run_id": 735, "dataset_id": 1, "artifacts": [],
                "config": {"dataset_root": tmp, "iteration_history": history},
            })
            self.assertEqual(package["validation_backlog"][0]["run_id"], 734)
            self.assertIn("Held-out shadow concordance",
                          package["validation_backlog"][0]["required_validation_metrics"])

    def test_evidence_exposes_lge_first_dependent_workstreams(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"CMR_LGE_CASE_SALT": "test-salt"}
        ):
            root = Path(tmp)
            cmr = root / "cmr.csv"
            lge = root / "lge.csv"
            cmr.write_text("patient,issues\nC1,cmr_myo_small_area\n", encoding="utf-8")
            lge.write_text(
                "patient,issues\nL1,myo_topology|mi_multiple_components\n",
                encoding="utf-8",
            )
            package = build_evidence_package({
                "run_id": 1, "dataset_id": 1,
                "artifacts": [
                    {"artifact_type": "cmr_middle_manifest", "path": str(cmr)},
                    {"artifact_type": "lge_middle_manifest", "path": str(lge)},
                ],
                "config": {"dataset_root": tmp, "iteration_history": []},
            })
            streams = {item["key"]: item for item in package["workstreams"]}
            self.assertEqual(set(streams), {"cmr_myo", "lge_myo", "lge_mi"})
            self.assertEqual(streams["cmr_myo"]["specialist_role"], "cmr_specialist")
            self.assertEqual(streams["lge_myo"]["specialist_role"], "lge_myo_specialist")
            self.assertEqual(streams["lge_mi"]["specialist_role"], "lge_mi_specialist")
            self.assertEqual(streams["lge_mi"]["execution_status"],
                             "production_check_loop_missing")
            self.assertEqual(package["workflow_dependency"]["stage"], "lge_myo")
            self.assertIn("temporal-frame", package["workflow_dependency"]["transfer_prior"])
            self.assertIn("Parallelize samples only", package["iteration_instruction"])
            self.assertTrue(streams["lge_myo"]["eligible_now"])
            self.assertFalse(streams["lge_mi"]["eligible_now"])
            self.assertFalse(streams["cmr_myo"]["eligible_now"])

    def test_check_iteration_exposes_only_aggregate_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runtime/check_iterations/latest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"pending": 193, "control_gate_status": "FAIL",
                                        "candidate_quality": {"status": "SCORED", "scored_volumes": 2,
                                                              "patient": "PRIVATE"},
                                        "report_path": "/data/private/patient",
                                        "records": [{"patient": "PRIVATE"}]}))
            self.assertEqual(check_iteration_evidence(root),
                             {"pending": 193, "control_gate_status": "FAIL",
                              "gate_scope": {
                                  "control_gate_status": "Existing spatial-boundary rule for automatic trash routing only.",
                                  "not_global_veto": "FAIL does not prohibit isolated experiments or independently eligible checked candidates.",
                                  "checked_policy": "Changed candidates require validator plus SALVAGE or calibrated improvement; unchanged non-spatial labels may pass at calibrated score>=70. Frozen validation cases stay excluded.",
                                  "policy_source": "app/check_auto_apply.py:eligible_plan",
                                  "holdout_role": "20-case human adjudication is monitoring/feedback, not a per-candidate manual approval requirement under current operator authorization.",
                              },
                              "candidate_quality": {"status": "SCORED", "scored_volumes": 2}})

    def test_check_iteration_promotes_measured_spatial_blocker_to_planning_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runtime/check_iterations/latest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "control_gate_status": "FAIL",
                "checked_false_routes": 3,
                "spatial_boundary_candidates": 138,
            }))
            evidence = check_iteration_evidence(root)
            blocker = evidence["production_blocker"]
            self.assertEqual(blocker["priority"], "RESOLVE_BEFORE_NEW_SHADOWS")
            self.assertEqual(blocker["target_path"], "scripts/shadow_spatial_boundary.py")

    def test_check_iteration_promotes_route_conflict_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runtime/check_iterations/latest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "control_gate_status": "PASS",
                "pending_route_conflicts": 15,
                "pending_holdout": 50,
                "pending_non_holdout": 23,
            }))
            evidence = check_iteration_evidence(root)
            blocker = evidence["production_blocker"]
            self.assertEqual(blocker["target_path"], "app/check_auto_apply.py")
            self.assertEqual(blocker["owner"], "qc_code_optimizer")
            self.assertEqual(blocker["owner"], "qc_code_optimizer")

    def test_deferred_route_blocker_is_not_retried_for_unchanged_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            (runtime / "check_iterations").mkdir(parents=True)
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "control_gate_status": "PASS", "pending_route_conflicts": 5,
            }))
            (runtime / "autopilot_state.json").write_text(json.dumps({
                "last_issue_fingerprint": "same",
                "deferred_production_blocker": {
                    "target_path": "app/check_auto_apply.py",
                    "issue_fingerprint": "same",
                },
            }))
            evidence = check_iteration_evidence(root)
            self.assertNotIn("production_blocker", evidence)
            self.assertEqual(evidence["deferred_production_blocker"]["target_path"],
                             "app/check_auto_apply.py")

    def test_aggregates_deidentified_why_without_raw_edit_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, project = root / "dataset", root / "project"
            png = dataset / "labels/check/待筛/cmr/PRIVATE/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"png")
            corpus = project / "runtime/manual_annotations/analyzed_training_corpus.jsonl"
            corpus.parent.mkdir(parents=True)
            corpus.write_text("\n".join(json.dumps(row) for row in (
                {"edit_id": "approved-1", "modality": "cmr", "target": "myo", "patient": "PRIVATE",
                 "operation": "add",
                 "suggested_pattern": "local_expansion", "analysis": {
                     "likely_edit_intent": "Restore a locally missing myocardial arc",
                     "qc_rule_candidate": "Flag contiguous localized dropout",
                     "reusable_casebook_pattern": "/data/private/patient"}},
                {"edit_id": "staged-2", "modality": "cmr", "target": "myo", "patient": "PRIVATE",
                 "suggested_pattern": "local_expansion", "analysis": {}}
            )))
            for edit_id, status in (("approved-1", "APPLIED"), ("staged-2", "STAGED")):
                provenance = corpus.parent / "edits" / edit_id / "provenance.json"
                provenance.parent.mkdir(parents=True)
                provenance.write_text(json.dumps({
                    "edit_id": edit_id, "status": status,
                    "applied_by": "reviewer" if status == "APPLIED" else None,
                    "applied_hash": "hash" if status == "APPLIED" else None,
                }))
            result = check_feedback_summary(dataset, project)
            self.assertEqual(result["states"]["待筛"], 1)
            self.assertEqual(result["reviewed_edit_patterns"], {"local_expansion": 1})
            self.assertEqual(result["confirmed_edit_operations"], {"add": 1})
            self.assertEqual(result["deidentified_edit_lessons"]["local_expansion"], {
                "likely_edit_intent": "Restore a locally missing myocardial arc",
                "qc_rule_candidate": "Flag contiguous localized dropout"})
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_staged_edit_becomes_learning_evidence_only_after_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "runtime/manual_annotations"
            manual.mkdir(parents=True)
            (manual / "analyzed_training_corpus.jsonl").write_text(json.dumps({
                "edit_id": "case-1", "modality": "cmr", "target": "myo",
                "suggested_pattern": "local_expansion", "analysis": {"qc_rule_candidate": "candidate"},
            }) + "\n")
            provenance = manual / "edits/case-1/provenance.json"
            provenance.parent.mkdir(parents=True)
            provenance.write_text(json.dumps({"edit_id": "case-1", "status": "STAGED"}))
            self.assertEqual(approved_manual_edit_records(root), [])
            provenance.write_text(json.dumps({
                "edit_id": "case-1", "status": "APPLIED", "applied_by": "reviewer",
                "applied_hash": "hash",
            }))
            self.assertEqual(len(approved_manual_edit_records(root)), 1)

    def test_same_state_review_becomes_deidentified_positive_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            backup = dataset / "labels/check/manifest_backup"
            backup.mkdir(parents=True)
            (backup / "cmr_boundary_manifest.csv").write_text(
                "patient,volume,issues\nPRIVATE_PERSON,1.0.nii.gz,cmr_myo_excess_components\n",
                encoding="utf-8",
            )
            history = dataset / "labels/check/boundary_review_history.jsonl"
            history.write_text(json.dumps({
                "timestamp": "2026-10-26T00:00:00Z",
                "relative_key": "PRIVATE_PERSON/1.0.nii.gz",
                "previous_status": "trash", "status": "trash",
                "reviewer": "doctor-private",
            }) + "\n", encoding="utf-8")
            records = approved_manual_review_records(dataset)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["decision"], "confirmed_trash")
            self.assertEqual(records[0]["issues"], ["cmr_myo_excess_components"])
            serialized = json.dumps(records)
            self.assertNotIn("PRIVATE_PERSON", serialized)
            self.assertNotIn("doctor-private", serialized)


if __name__ == "__main__":
    unittest.main()
