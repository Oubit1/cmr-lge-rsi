"""Offline tests for the read-only LGE geometry supply scan."""

from __future__ import annotations

import unittest

from experiments.lge_geometry_supply_scan_exec_13 import (
    EXPECTED_NON_HOLDOUT,
    run_evaluation,
)


def _rec(cid, **extra):
    base = {
        "case_id": cid,
        "before_area": 100.0,
        "after_area": 100.0,
        "before_cavity": 0.3,
        "after_cavity": 0.3,
        "quality_before": 80.0,
        "quality_after": 80.0,
        "quality_delta": 0.0,
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {"topology": 80.0},
        "quality_subscores_after": {"topology": 80.0},
    }
    base.update(extra)
    return base


class GeometrySupplyScanTest(unittest.TestCase):
    def test_missing_geometry_reported_unavailable_not_imputed(self):
        records = [_rec(f"case_{i:02d}") for i in range(EXPECTED_NON_HOLDOUT)]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        self.assertEqual(report["volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertFalse(report["geometry_supplied"])
        self.assertEqual(report["component_count_filled"], 0)
        self.assertEqual(report["hole_count_filled"], 0)
        self.assertEqual(report["component_count_fill_rate"], 0.0)
        self.assertEqual(report["hole_count_fill_rate"], 0.0)
        self.assertFalse(report["component_count_fill_rate_meets_threshold"])
        self.assertFalse(report["hole_count_fill_rate_meets_threshold"])
        self.assertEqual(report["unavailable_field_counts"]["component_count"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(report["unavailable_field_counts"]["hole_count"], EXPECTED_NON_HOLDOUT)
        for case in report["cases"].values():
            self.assertIsNone(case["component_count"])
            self.assertIsNone(case["hole_count"])
            self.assertIn("component_count", case["unavailable_fields"])
            self.assertIn("hole_count", case["unavailable_fields"])

    def test_supplied_geometry_fills_counts(self):
        records = [
            _rec(f"case_{i:02d}", myo_components=2, myo_holes=1)
            for i in range(EXPECTED_NON_HOLDOUT)
        ]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        self.assertTrue(report["geometry_supplied"])
        self.assertEqual(report["component_count_filled"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(report["hole_count_filled"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(report["component_count_fill_rate"], 1.0)
        self.assertEqual(report["hole_count_fill_rate"], 1.0)
        self.assertTrue(report["component_count_fill_rate_meets_threshold"])
        self.assertTrue(report["hole_count_fill_rate_meets_threshold"])

    def test_holdout_excluded_and_zero_label_write(self):
        records = [_rec(f"case_{i:02d}") for i in range(EXPECTED_NON_HOLDOUT + 3)]
        holdout = ["case_00", "case_01", "case_02"]
        report = run_evaluation({"records": records, "holdout_case_ids": holdout})
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 3)
        self.assertEqual(report["volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"]["labels"], 0)
        for cid in holdout:
            self.assertNotIn(cid, report["cases"])

    def test_duplicate_case_ids_deduplicated(self):
        records = [_rec("case_dup") for _ in range(3)]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        self.assertEqual(report["volumes_scanned"], 1)
        self.assertEqual(report["duplicate_case_ids"], ["case_dup", "case_dup"])

    def test_concordance_reported_unavailable(self):
        report = run_evaluation({"records": [_rec("case_x")], "holdout_case_ids": []})
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])


if __name__ == "__main__":
    unittest.main()
