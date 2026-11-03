import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.runtime_evaluation import evaluation_input, execute_evaluation


class RuntimeEvaluationTests(unittest.TestCase):
    def test_lge_stage_uses_latest_lge_manifest_and_strips_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "artifacts/run_000010/evidence_snapshot"
            latest = root / "artifacts/run_000011/evidence_snapshot"
            old.mkdir(parents=True)
            latest.mkdir(parents=True)
            header = "modality,patient,slice,layer_index,issues,evidence,check\n"
            (old / "lge_issue_manifest.csv").write_text(
                header + "lge,OLD_PERSON,old.nii.gz,1,myo_topology,myo_components=7,/private/old.png\n",
                encoding="utf-8",
            )
            (latest / "lge_issue_manifest.csv").write_text(
                header
                + "lge,PRIVATE_PERSON,a.nii.gz,2,myo_topology;mi_multiple_components,myo_components=2;mi_components=3,/private/a.png\n"
                + "lge,OTHER_PERSON,b.nii.gz,3,mi_fraction_high,mi_myo_ratio=0.4,/private/b.png\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "test"}):
                myo = evaluation_input(root, workflow_stage="lge_myo")
                mi = evaluation_input(root, workflow_stage="lge_mi")
            self.assertEqual(myo["source_run"], "run_000011")
            self.assertEqual(myo["modality"], "lge")
            self.assertEqual(myo["workflow_stage"], "lge_myo")
            self.assertEqual(myo["full_stage_queue"], 1)
            self.assertEqual(myo["schema"], "existing-lge-qc-measurements-v2")
            self.assertEqual(myo["records"][0]["issue_codes"], ["myo_topology"])
            self.assertEqual(myo["records"][0]["issues"], ["myo_topology"])
            self.assertEqual(myo["records"][0]["measurements"]["myo_components"], 2)
            self.assertEqual(myo["records"][0]["myo_components"], 2)
            self.assertEqual(
                myo["field_availability"]["issue_code_counts"], {"myo_topology": 1})
            self.assertEqual(
                myo["field_availability"]["measurement_field_counts"],
                {"mi_components": 1, "myo_components": 1})
            self.assertFalse(myo["field_availability"]["missing_values_imputed"])
            self.assertEqual(mi["full_stage_queue"], 2)
            serialized = json.dumps({"myo": myo, "mi": mi})
            self.assertNotIn("PRIVATE_PERSON", serialized)
            self.assertNotIn("a.nii.gz", serialized)
            self.assertNotIn("/private/", serialized)

    def test_real_measurements_strip_identifiers_and_preserve_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "runtime/check_iterations"
            folder.mkdir(parents=True)
            report = folder / "evaluation.json"
            report.write_text(json.dumps({"records": [{"key": "PRIVATE_PATIENT/scan",
                "before_area": 12, "after_area": 14, "frames": ["private_path"],
                "decision": "REVIEW", "quality_status": "SCORED",
                "quality_flags_before": ["fragmentation_candidate"],
                "quality_subscores_before": {"topology": 88.5},
                "extra": "private"}, {"key": "OTHER_PATIENT/scan", "before_area": 20}]}))
            holdout = [{"key": "PRIVATE_PATIENT/scan", "stratum": "cluster"}]
            holdout_sha = hashlib.sha256(json.dumps(
                holdout, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            (folder / "holdout_v1.json").write_text(json.dumps({
                "holdout": holdout, "holdout_sha256": holdout_sha,
            }))
            (folder / "spatial_boundary_shadow.json").write_text(json.dumps({
                "rule": "rule-v1", "control_gate": "FAIL", "checked_false_routes": 1,
                "pending_spatial_boundary_candidates": 2,
                "records": [{"key": "PRIVATE_PATIENT/scan", "state": "checked",
                             "decision": "spatial_boundary_candidate",
                             "reason": "outside_unique_sax_core", "rank": 1,
                             "core": [2, 6]}],
            }))
            (folder / "latest.json").write_text(json.dumps({"report_path": str(report)}))
            benchmark = root / "artifacts/image_label_quality_v1"
            benchmark.mkdir(parents=True)
            (benchmark / "summary.json").write_text(json.dumps({"groups": {
                "gold:cmr_myo": {"n": 10, "mean_score": 96.0},
                "gold:lge_myo": {"n": 12, "mean_score": 95.0},
            }}))
            (benchmark / "defect_validation.json").write_text(json.dumps({
                "clean_reference": {"below_70": .01},
                "classes": {"fragmentation": {"detected_drop_ge_5": .9}},
            }))
            with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "test"}):
                data = evaluation_input(root)
            self.assertEqual(data["records"][0]["before_area"], 12)
            self.assertEqual(data["records"][0]["quality_subscores_before"]["topology"], 88.5)
            self.assertEqual(data["quality_benchmark"]["clean_reference"]["below_70"], .01)
            self.assertNotIn("gold:lge_myo", data["quality_benchmark"]["cmr_reference_groups"])
            self.assertEqual(data["spatial_control"]["checked_false_routes"], 1)
            self.assertEqual(data["spatial_control"]["records"][0]["core"], [2, 6])
            self.assertEqual(data["full_pending_queue"], 2)
            self.assertEqual(data["holdout_case_ids"], [data["records"][0]["case_id"]])
            self.assertEqual(data["frozen_split"]["frozen_holdout_in_queue"], 1)
            self.assertEqual(data["frozen_split"]["non_holdout_in_queue"], 1)
            self.assertNotIn("before_cavity", data["records"][0])
            self.assertNotIn("private", json.dumps(data).lower())
            self.assertNotIn("PRIVATE_PATIENT", json.dumps(data))
            self.assertFalse(data["reference_labels_available"])

    def test_actual_owner_function_receives_input_without_api_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "orchestrator"
            package = snapshot / "experiments"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            (package / "probe.py").write_text(
                "import os\ndef run_evaluation(data):\n"
                " return {'n':len(data['records']), 'secret':os.getenv('DASHSCOPE_API_KEY')}\n")
            data = {"status": "AVAILABLE", "records": [{"before_area": 3}],
                    "source_sha256": "source"}
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "DO_NOT_FORWARD"}):
                result = execute_evaluation(snapshot, "experiments.probe", data)
            self.assertEqual(result["status"], "EXECUTED")
            self.assertEqual(result["report"], {"n": 1, "secret": None})
            self.assertTrue((Path(tmp)/"real_evaluation/report.json").is_file())
            scripts = snapshot / "scripts"
            scripts.mkdir()
            (scripts / "__init__.py").touch()
            (scripts / "probe.py").write_text(
                "def run_evaluation(data):\n return {'n':len(data['records']), 'kind':'script'}\n")
            script_result = execute_evaluation(snapshot, "scripts.probe", data)
            self.assertEqual(script_result["status"], "EXECUTED")
            self.assertEqual(script_result["report"], {"n": 1, "kind": "script"})
            self.assertEqual(execute_evaluation(snapshot, "../outside", data)["status"], "FAILED")
            self.assertEqual(execute_evaluation(snapshot, "app.probe", data)["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
