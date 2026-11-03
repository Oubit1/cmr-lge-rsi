from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_lge_feedback_consumer import run_once, validate_report


def input_data() -> dict:
    return {
        "status": "AVAILABLE", "modality": "lge", "workflow_stage": "lge_myo",
        "source_sha256": "source-hash", "records": [{"case_id": "case_a"}],
        "holdout_case_ids": [], "frozen_split": {"status": "UNAVAILABLE"},
        "schema": "existing-lge-qc-measurements-v2",
        "field_availability": {"record_count": 1,
                               "issue_code_counts": {"myo_topology": 1}},
    }


def valid_report() -> dict:
    return {
        "source_modality": "lge", "workflow_stage": "lge_myo",
        "input_source_sha256": "source-hash", "lge_myo_consumer_ready": True,
        "lge_myo_pending": 1, "lge_myo_work": 0, "lge_myo_resolved": 0,
        "frozen_split_status": "UNAVAILABLE", "holdout_excluded": 0,
        "label_write": False, "production_writes": 0,
    }


class LgeFeedbackConsumerDriverTests(unittest.TestCase):
    def test_validation_rejects_wrong_provenance(self):
        report = valid_report()
        report["input_source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            validate_report(input_data(), report, "lge_myo")

    def test_validation_rejects_fabricated_holdout(self):
        report = valid_report()
        report["holdout_excluded"] = 48
        with self.assertRaisesRegex(ValueError, "holdout_unavailable"):
            validate_report(input_data(), report, "lge_myo")

    def test_run_once_publishes_atomically_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "scripts/lge_feedback_loop.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_text("def run_evaluation(data): return {}\n", encoding="utf-8")
            output = root / "runtime/lge_iterations/latest.json"
            audit = root / "runtime/lge_iterations/audit.jsonl"
            module = types.SimpleNamespace(run_evaluation=lambda data: valid_report())
            with patch("scripts.run_lge_feedback_consumer.dependent_workflow_status",
                       return_value={"stage": "lge_myo"}), patch(
                "scripts.run_lge_feedback_consumer.evaluation_input",
                return_value=input_data()), patch(
                "scripts.run_lge_feedback_consumer.importlib.import_module",
                return_value=module), patch(
                "scripts.run_lge_feedback_consumer.importlib.reload", return_value=module):
                first = run_once(root, Path("dataset"), output, audit)
                second = run_once(root, Path("dataset"), output, audit)
            self.assertEqual(first["status"], "PUBLISHED")
            self.assertEqual(second["status"], "UNCHANGED")
            published = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(published["consumer_status"], "VALIDATED")
            self.assertEqual(published["active_input_contract"]["record_count"], 1)
            self.assertTrue(
                published["active_input_contract"]["missing_values_must_not_be_imputed"])
            self.assertEqual(len(audit.read_text(encoding="utf-8").splitlines()), 1)

    def test_missing_promoted_consumer_is_normal_waiting_state(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.run_lge_feedback_consumer.dependent_workflow_status",
            return_value={"stage": "lge_myo"}
        ):
            result = run_once(Path(tmp), Path("dataset"))
        self.assertEqual(result["reason"], "consumer_not_promoted")


if __name__ == "__main__":
    unittest.main()
