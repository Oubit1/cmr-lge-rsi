"""Offline tests for the S760 consecutive-slice area-ratio suppress rule."""

import unittest

from experiments.cmr_area_ratio_consecutive_suppress_rule import (
    MIN_CONSECUTIVE,
    ROUTING_MODE,
    run_evaluation,
)


def _rec(cid, ref, area, cavity, run=None, **flags):
    rec = {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }
    if run is not None:
        rec["consecutive_slices"] = run
    rec.update(flags)
    return rec


class ConsecutiveSuppressRuleTest(unittest.TestCase):
    def test_flagged_when_consecutive_and_not_suppressed(self):
        data = {
            "records": [
                _rec("c1", 1000.0, 100.0, 1.0, run=MIN_CONSECUTIVE),
            ]
        }
        report = run_evaluation(data)
        case = report["cases"]["c1"]
        self.assertTrue(case["consecutive_ok"])
        self.assertFalse(case["suppressed"])
        self.assertEqual(case["routing"], ROUTING_MODE)
        self.assertEqual(report["n_records"], 1)

    def test_suppressed_when_run_below_minimum(self):
        data = {"records": [_rec("c1", 1000.0, 100.0, 1.0, run=1)]}
        report = run_evaluation(data)
        case = report["cases"]["c1"]
        self.assertFalse(case["consecutive_ok"])
        self.assertEqual(case["routing"], "stack_level_review")
        self.assertIn("c1", report["suppressed_cases"])

    def test_suppressed_when_empty_flag_set(self):
        data = {
            "records": [
                _rec("c1", 1000.0, 100.0, 1.0, run=MIN_CONSECUTIVE, empty=True),
            ]
        }
        report = run_evaluation(data)
        case = report["cases"]["c1"]
        self.assertTrue(case["suppressed"])
        self.assertEqual(case["routing"], "stack_level_review")
        self.assertEqual(report["suppression_coverage"]["empty"]["count"], 1)
        self.assertEqual(report["suppression_coverage"]["empty"]["status"], "ok")

    def test_suppression_unavailable_when_flag_absent(self):
        data = {"records": [_rec("c1", 1000.0, 100.0, 1.0, run=MIN_CONSECUTIVE)]}
        report = run_evaluation(data)
        cov = report["suppression_coverage"]["border_touch"]
        self.assertIsNone(cov["count"])
        self.assertEqual(cov["status"], "unavailable_no_masks_or_geometry")

    def test_missing_measurement_not_imputed(self):
        data = {"records": [{"case_id": "c1", "reference_area": 1000.0}]}
        report = run_evaluation(data)
        case = report["cases"]["c1"]
        self.assertIsNone(case["area_ratio"])
        self.assertIsNone(case["myo_cavity_area_ratio"])
        self.assertIn("c1", report["missing_measurement_cases"])

    def test_concordance_uses_pseudo_holdout_status(self):
        data = {
            "records": [
                _rec("c1", 1000.0, 100.0, 1.0, run=MIN_CONSECUTIVE),
                _rec("c2", 1000.0, 900.0, 1.0, run=MIN_CONSECUTIVE),
            ],
            "statuses": {"c1": "trash", "c2": "checked"},
        }
        report = run_evaluation(data)
        conc = report["concordance"]
        self.assertEqual(conc["cohort"], "pseudo_holdout_checked_volumes_not_truth")
        self.assertEqual(conc["flagged_total"], 1)
        self.assertEqual(conc["flagged_concordant_total"], 1)

    def test_zero_write_audit_fields(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertTrue(report["module_hash"])
        self.assertIn("git checkout", report["rollback_command"])

    def test_ledger_deltas_unavailable_without_ledger(self):
        report = run_evaluation({"records": []})
        self.assertEqual(
            report["ledger_deltas"]["status"], "unavailable_no_ledger_supplied"
        )


if __name__ == "__main__":
    unittest.main()
