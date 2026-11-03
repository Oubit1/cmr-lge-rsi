"""Offline tests for the S760 pseudo-holdout area-ratio concordance check."""

import json
import unittest

from experiments.cmr_area_ratio_pseudo_holdout_check import (
    COHORT_LABEL,
    ROLLBACK_COMMAND,
    audit_diff,
    module_hash,
    report_to_json,
    run_evaluation,
)


class TestPseudoHoldoutCheck(unittest.TestCase):
    def _records(self):
        return [
            # low area ratio, low cavity -> low_area_low_cavity
            {"case_id": "c1", "reference_area": 1000.0, "before_area": 100.0,
             "before_cavity": 0.2},
            # adequate area ratio, low cavity -> adequate_area_low_cavity
            {"case_id": "c2", "reference_area": 1000.0, "before_area": 500.0,
             "before_cavity": 0.2},
            # low area ratio, adequate cavity -> low_area_adequate_cavity
            {"case_id": "c3", "reference_area": 1000.0, "before_area": 100.0,
             "before_cavity": 0.9},
            # missing measurement -> unclassified, not imputed
            {"case_id": "c4", "reference_area": 1000.0, "before_area": None,
             "before_cavity": 0.2},
        ]

    def test_counts_and_denominators(self):
        report = run_evaluation({"records": self._records()})
        self.assertEqual(report["n_records"], 4)
        for c, entry in report["cluster_counts"].items():
            self.assertEqual(entry["denominator"], 4)
        self.assertIn("c4", report["unclassified_cases"])
        self.assertIn("c4", report["missing_measurement_cases"])

    def test_concordance_pseudo_holdout(self):
        data = {
            "records": self._records(),
            "statuses": {"c1": "trash", "c2": "checked", "c3": "trash"},
        }
        report = run_evaluation(data)
        conc = report["concordance"]
        self.assertEqual(conc["cohort"], COHORT_LABEL)
        self.assertEqual(conc["flagged_total"], 3)
        self.assertEqual(conc["flagged_concordant_total"], 2)
        self.assertAlmostEqual(conc["flagged_concordance"], 2 / 3)
        self.assertIn("c4", conc["unknown_status_cases"])

    def test_ledger_deltas(self):
        ledger = {
            "low_area_low_cavity": {"checked": 0, "trash": 1},
            "adequate_area_low_cavity": {"checked": 1, "trash": 0},
            "low_area_adequate_cavity": {"checked": 0, "trash": 0},
        }
        report = run_evaluation({"records": self._records(), "ledger": ledger})
        deltas = report["ledger_deltas"]
        self.assertEqual(deltas["status"], "ok")
        self.assertIsNotNone(deltas["low_area_low_cavity"]["trash_delta"])

    def test_ledger_unavailable_when_absent(self):
        report = run_evaluation({"records": self._records()})
        self.assertEqual(
            report["ledger_deltas"]["status"], "unavailable_no_ledger_supplied"
        )

    def test_zero_write_audit(self):
        report = run_evaluation({"records": self._records()})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["rollback_command"], ROLLBACK_COMMAND)
        self.assertEqual(len(module_hash()), 64)

    def test_suppression_and_unavailable(self):
        report = run_evaluation({"records": self._records()})
        for entry in report["suppression_coverage"].values():
            self.assertIsNone(entry["count"])
            self.assertEqual(entry["status"], "unavailable_no_masks_or_geometry")
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])

    def test_audit_diff_between_reports(self):
        prev = run_evaluation({"records": self._records()})
        cur = run_evaluation({"records": self._records()})
        diff = audit_diff(prev, cur)
        self.assertEqual(diff["production_writes"], 0)
        self.assertEqual(diff["changed_cases"], [])

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": self._records()})
        json.loads(report_to_json(report))


if __name__ == "__main__":
    unittest.main()
