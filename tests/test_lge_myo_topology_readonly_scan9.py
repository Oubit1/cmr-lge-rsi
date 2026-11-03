"""Offline tests for the read-only LGE myo topology scan (9 non-holdout)."""

import unittest

from experiments.lge_myo_topology_readonly_scan9 import (
    EXPECTED_NON_HOLDOUT,
    run_evaluation,
)


def _rec(cid, **extra):
    base = {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": 500.0,
        "after_area": 500.0,
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


class TopologyScanTests(unittest.TestCase):
    def test_holdout_excluded_and_denominator(self):
        records = [_rec("case_h%d" % i) for i in range(3)]
        records += [_rec("case_n%d" % i) for i in range(EXPECTED_NON_HOLDOUT)]
        data = {
            "records": records,
            "holdout_case_ids": ["case_h%d" % i for i in range(3)],
        }
        report = run_evaluation(data)
        self.assertEqual(report["volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 3)
        self.assertTrue(report["non_holdout_denominator_matches_expected"])
        for cid in report["cases"]:
            self.assertFalse(cid.startswith("case_h"))

    def test_missing_fields_reported_unavailable_not_imputed(self):
        data = {"records": [_rec("case_a")], "holdout_case_ids": []}
        report = run_evaluation(data)
        case = report["cases"]["case_a"]
        self.assertIsNone(case["component_count"])
        self.assertIsNone(case["secondary_primary_area_ratio"])
        self.assertIsNone(case["cross_slice_continuity_breaks"])
        self.assertIsNone(case["border_contact_proxy"])
        self.assertIn("component_count", case["unavailable_fields"])
        self.assertIn("secondary_primary_area_ratio", case["unavailable_fields"])
        self.assertIn("cross_slice_continuity_breaks", case["unavailable_fields"])
        self.assertIn("border_contact_proxy", case["unavailable_fields"])
        self.assertEqual(report["component_count_fill_rate"], 0.0)

    def test_supplied_fields_are_reported(self):
        rec = _rec(
            "case_b",
            myo_components=3,
            primary_component_area=400.0,
            secondary_component_area=100.0,
            cross_slice_continuity_breaks=2,
            border_contact_proxy=True,
        )
        report = run_evaluation({"records": [rec], "holdout_case_ids": []})
        case = report["cases"]["case_b"]
        self.assertEqual(case["component_count"], 3)
        self.assertAlmostEqual(case["secondary_primary_area_ratio"], 0.25)
        self.assertEqual(case["cross_slice_continuity_breaks"], 2)
        self.assertTrue(case["border_contact_proxy"])
        self.assertEqual(case["unavailable_fields"], [])
        self.assertEqual(report["component_count_fill_rate"], 1.0)

    def test_zero_write_flags(self):
        report = run_evaluation({"records": [_rec("case_c")], "holdout_case_ids": []})
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["holdout_selected"], 0)


if __name__ == "__main__":
    unittest.main()
