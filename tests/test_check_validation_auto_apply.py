from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

from app import check_auto_apply, check_validation


class CheckValidationTests(unittest.TestCase):
    def test_frozen_plan_and_review_report_do_not_change_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runtime = root / "dataset", root / "runtime"
            holdout = []
            rows = []
            for group, (stratum, count) in enumerate((("adequate_area_low_cavity", 13),
                                   ("low_area_low_cavity", 4),
                                   ("low_area_adequate_cavity", 3))):
                for i in range(count):
                    key = f"P{group}{i:02d}/v{i}.nii.gz"
                    holdout.append({"key": key, "stratum": stratum})
                    rows.append({"key": key, "modality": "cmr", "state": "pending",
                                 "shadow_boundary": i % 2 == 0, "candidate_frames": 0})
                    png = dataset / "labels/check/待筛/cmr" / key.replace(".nii.gz", ".png")
                    png.parent.mkdir(parents=True, exist_ok=True)
                    png.write_bytes(b"unchanged")
            split = {"holdout": holdout, "holdout_sha256": check_validation._digest(
                sorted(holdout, key=lambda row: row["key"]))}
            path = runtime / "check_iterations/holdout_v1.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(split))
            inventory = {"shadow_current": True, "shadow_created_at": "frozen",
                         "shadow_control_gate": "FAIL", "rows": rows}
            with patch.object(check_validation, "build_check_inventory", return_value=inventory):
                plan = check_validation.propose_plan(dataset, runtime)
                self.assertEqual(len(plan["items"]), 20)
                for item in plan["items"]:
                    check_validation.submit_decision(dataset, runtime, {
                        "key": item["key"], "route": "checked", "reviewer": "tester",
                        "plan_sha256": plan["plan_sha256"],
                    })
                state = check_validation.review_state(dataset, runtime)
            self.assertTrue(state["report"]["ready_for_independent_review"])
            self.assertEqual(state["report"]["reviewed"], 20)
            self.assertTrue((runtime / "check_validation/report.json").is_file())
            for item in plan["items"]:
                self.assertEqual((dataset / "labels/check/待筛/cmr" /
                                  item["key"].replace(".nii.gz", ".png")).read_bytes(),
                                 b"unchanged")


class AutoApplyTests(unittest.TestCase):
    def test_route_conflict_abstains_then_reports_governed_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runtime = root / "dataset", root / "runtime"
            key = "P001/1.nii.gz"
            png = dataset / "labels/check/待筛/cmr/P001/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"synthetic")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nP001,1.nii.gz\n")
            candidate_root = runtime / "check_iterations/run1"
            candidate_root.mkdir(parents=True)
            (candidate_root / "evaluation.json").write_text(json.dumps({"records": [{
                "key": key, "decision": "REVIEW", "quality_status": "SCORED",
                "quality_after": 75, "quality_delta": 4,
            }]}))
            (candidate_root / "candidate_validation.json").write_text(json.dumps({
                "records": [{"pool": "cmr_boundary_manifest", "index": 0,
                             "passed": True, "candidate": "unused.npz"}],
            }))
            (candidate_root / "spatial_boundary_shadow.json").write_text(json.dumps({
                "control_gate": "PASS", "checked_false_routes": 0,
                "records": [{"key": key, "state": "待筛",
                             "decision": "spatial_boundary_candidate"}],
            }))
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "report_path": str(candidate_root / "evaluation.json"),
            }))
            frozen: list[dict] = []
            (runtime / "check_iterations/holdout_v1.json").write_text(json.dumps({
                "holdout": frozen, "holdout_sha256": check_validation._digest(frozen),
            }))
            with patch.object(check_auto_apply, "build_check_inventory",
                              return_value={"shadow_current": True}):
                abstain = check_auto_apply.eligible_plan(dataset, runtime)
                self.assertEqual(abstain["route_conflicts_detected"], 1)
                self.assertEqual(abstain["route_conflicts_unresolved"], 1)
                self.assertFalse(abstain["selected"])
                with patch.object(check_auto_apply, "resolve_route_conflict",
                                  return_value={"route": "trash", "reason": "test_rule"}):
                    resolved = check_auto_apply.run(dataset, runtime)
            self.assertEqual(resolved["route_conflicts_resolved"], 1)
            self.assertEqual(resolved["route_conflicts_unresolved"], 0)
            self.assertTrue(resolved["planned"][0]["conflict_resolution"])
            self.assertEqual(resolved["preflight"][0]["preflight"], "PASS")

    def test_frozen_route_conflict_never_reaches_domain_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runtime = root / "dataset", root / "runtime"
            key = "P001/1.nii.gz"
            png = dataset / "labels/check/待筛/cmr/P001/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"synthetic")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nP001,1.nii.gz\n")
            candidate_root = runtime / "check_iterations/run1"
            candidate_root.mkdir(parents=True)
            (candidate_root / "evaluation.json").write_text(json.dumps({
                "records": [{"key": key, "decision": "REVIEW"}],
            }))
            (candidate_root / "candidate_validation.json").write_text(json.dumps({
                "records": [{"pool": "cmr_boundary_manifest", "index": 0,
                             "passed": True, "candidate": "unused.npz"}],
            }))
            (candidate_root / "spatial_boundary_shadow.json").write_text(json.dumps({
                "control_gate": "PASS", "checked_false_routes": 0,
                "records": [{"key": key, "state": "待筛",
                             "decision": "spatial_boundary_candidate"}],
            }))
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "report_path": str(candidate_root / "evaluation.json"),
            }))
            frozen = [{"key": key}]
            (runtime / "check_iterations/holdout_v1.json").write_text(json.dumps({
                "holdout": frozen, "holdout_sha256": check_validation._digest(frozen),
            }))
            resolver = Mock(return_value={"route": "trash", "reason": "bad"})
            with patch.object(check_auto_apply, "build_check_inventory",
                              return_value={"shadow_current": True}), \
                 patch.object(check_auto_apply, "resolve_route_conflict", resolver):
                plan = check_auto_apply.eligible_plan(dataset, runtime)
            resolver.assert_not_called()
            self.assertEqual(plan["route_conflicts_detected"], 0)
            self.assertFalse(plan["selected"])

    def test_score_threshold_plans_and_applies_only_non_holdout_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runtime = root / "dataset", root / "runtime"
            patient, volume = "P001", "-36.6493.nii.gz"
            key = f"{patient}/{volume}"
            label = dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
            label.parent.mkdir(parents=True)
            original = np.zeros((8, 8, 1), dtype=np.uint8)
            original[2:6, 2:6, 0] = 1
            nib.save(nib.Nifti1Image(original, np.eye(4)), str(label))
            before = check_auto_apply._sha(label)
            png = dataset / "labels/check/待筛/cmr" / patient / "-36.6493.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"synthetic-png")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nP001,-36.6493.nii.gz\n")
            candidate_root = runtime / "check_iterations/run1"
            candidate_root.mkdir(parents=True)
            fixed = original[..., 0].copy()
            fixed[2, 2] = 0
            npz = candidate_root / "case_k0.npz"
            np.savez(npz, original=original[..., 0], fixed=fixed)
            (candidate_root / "evaluation.json").write_text(json.dumps({"records": [{
                "key": key, "decision": "REVIEW", "quality_status": "SCORED",
                "quality_after": 71, "quality_delta": 3,
            }]}))
            (candidate_root / "candidate_validation.json").write_text(json.dumps({"records": [{
                "pool": "cmr_boundary_manifest", "index": 0,
                "passed": True, "candidate": str(npz),
            }]}))
            (candidate_root / "spatial_boundary_shadow.json").write_text(json.dumps({
                "control_gate": "FAIL", "records": [],
            }))
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "report_path": str(candidate_root / "evaluation.json"),
            }))
            frozen = [{"key": "OTHER/frozen.nii.gz"}]
            (runtime / "check_iterations/holdout_v1.json").write_text(json.dumps({
                "holdout": frozen, "holdout_sha256": check_validation._digest(frozen),
            }))
            with patch.object(check_auto_apply, "build_check_inventory",
                              return_value={"shadow_current": True}):
                preview = check_auto_apply.run(dataset, runtime)
                self.assertEqual(preview["eligible"], 1)
                self.assertEqual(preview["preflight"][0]["preflight"], "PASS")
                self.assertEqual(check_auto_apply._sha(label), before)
                stale = original.copy()
                stale[0, 0, 0] = 1
                nib.save(nib.Nifti1Image(stale, np.eye(4)), str(label))
                with self.assertRaisesRegex(ValueError, "source mask changed"):
                    check_auto_apply.apply_one(dataset, runtime, candidate_root,
                                               preview["planned"][0], dry_run=True)
                self.assertTrue(png.exists())
                nib.save(nib.Nifti1Image(original, np.eye(4)), str(label))
                before = check_auto_apply._sha(label)
                result = check_auto_apply.run(dataset, runtime, execute=True)
            self.assertEqual(len(result["applied"]), 1)
            self.assertFalse(png.exists())
            self.assertTrue((dataset / "labels/check/checked/cmr" / patient / png.name).exists())
            self.assertNotEqual(check_auto_apply._sha(label), before)
            backup = dataset / "label_iteration/auto_check_backups" / before[:16] / patient / volume
            self.assertEqual(check_auto_apply._sha(backup), before)
            audit = (runtime / "check_iterations/auto_apply.jsonl").read_text()
            self.assertIn('"event": "applied"', audit)

    def test_existing_score_pass_moves_png_without_changing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runtime = root / "dataset", root / "runtime"
            patient, volume = "P001", "1.nii.gz"
            key = f"{patient}/{volume}"
            label = dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
            label.parent.mkdir(parents=True)
            raw = np.zeros((8, 8, 1), dtype=np.uint8)
            raw[2:6, 2:6, 0] = 1
            nib.save(nib.Nifti1Image(raw, np.eye(4)), str(label))
            before = check_auto_apply._sha(label)
            png = dataset / "labels/check/待筛/cmr" / patient / "1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"synthetic-png")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nP001,1.nii.gz\n")
            candidate_root = runtime / "check_iterations/run1"
            candidate_root.mkdir(parents=True)
            (candidate_root / "evaluation.json").write_text(json.dumps({"records": [{
                "key": key, "decision": "REVIEW", "quality_status": "SCORED",
                "quality_before": 91, "quality_after": 91, "quality_delta": 0,
            }]}))
            (candidate_root / "candidate_validation.json").write_text(
                json.dumps({"records": []}))
            (candidate_root / "spatial_boundary_shadow.json").write_text(json.dumps({
                "control_gate": "FAIL", "records": [],
            }))
            (runtime / "check_iterations/latest.json").write_text(json.dumps({
                "report_path": str(candidate_root / "evaluation.json"),
            }))
            frozen = [{"key": key}, {"key": "OTHER/frozen.nii.gz"}]
            frozen = sorted(frozen, key=lambda row: row["key"])
            holdout_sha = check_validation._digest(frozen)
            (runtime / "check_iterations/holdout_v1.json").write_text(json.dumps({
                "holdout": frozen, "holdout_sha256": holdout_sha,
            }))
            plan = {"schema_version": "cmr-check-validation-v1",
                    "holdout_sha256": holdout_sha,
                    "items": [{"key": "OTHER/frozen.nii.gz", "stratum": "test",
                               "suggestion": None, "editor_index": None}]}
            plan["plan_sha256"] = check_validation._digest(plan)
            plan_path = runtime / "check_validation/plan.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps(plan))
            with patch.object(check_auto_apply, "build_check_inventory",
                              return_value={"shadow_current": True}):
                preview = check_auto_apply.run(dataset, runtime)
                self.assertEqual(preview["planned"][0]["reason"], "quality_C_existing")
                self.assertFalse(preview["planned"][0]["candidates"])
                self.assertEqual(preview["holdout_excluded"], 2)
                self.assertEqual(preview["validation_excluded"], 1)
                result = check_auto_apply.run(dataset, runtime, execute=True)
            self.assertEqual(result["applied"][0]["label_write"], False)
            self.assertEqual(check_auto_apply._sha(label), before)
            self.assertTrue((dataset / "labels/check/checked/cmr" / patient / png.name).exists())
            audit = (runtime / "check_iterations/auto_apply.jsonl").read_text()
            self.assertIn('"label_write": false', audit)


if __name__ == "__main__":
    unittest.main()
