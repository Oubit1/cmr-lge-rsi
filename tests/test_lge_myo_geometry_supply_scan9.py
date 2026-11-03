"""Offline unit tests for the read-only LGE myo geometry-supply scan."""

from __future__ import annotations

import unittest

from experiments.lge_myo_geometry_supply_scan9 import (
    EXPECTED_NON_HOLDOUT,
    run_evaluation,
)


def _rec(cid, **kw):
    base = {"case_id": cid}
    base.update(kw)
    return base


class GeometrySupplyScanTest(unittest.TestCase):
    def test_selects_nine_non_holdout_and_excludes_holdout(self):
        records = [_rec("case_%02d" % i) for i in range(9)]
        records += [_rec("hold_%02d" % i) for i in range(48)]
        holdout = ["hold_%02d" % i for i in range(48)]
        report = run_evaluation({"records": records, "holdout_case_ids": holdout})
        self.assertEqual(report["volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 48)
        self.assertTrue(report["non_holdout_denominator_matches_expected"])

    def test_dedupes_by_case_id(self):
        records = [_rec("case_a"), _rec("case_a"), _rec("case_b")]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        self.assertEqual(report["volumes_scanned"], 2)
        self.assertEqual(report["duplicate_case_ids"], ["case_a"])

    def test_missing_geometry_reports_unavailable_not_imputed(self):
        records = [_rec("case_a")]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        case = report["cases"]["case_a"]
        self.assertIsNone(case["component_count"])
        self.assertIsNone(case["ring_violation_count"])
        self.assertIsNone(case["border_contact_proxy"])
        self.assertFalse(case["component_count_available"])
        self.assertIn("component_count", case["unavailable_fields"])
        self.assertTrue(case["unavailable_reason"].startswith("unavailable:"))
        self.assertEqual(report["geometry_fill_rate"], 0.0)

    def test_supplied_geometry_fills_and_computes_ratio(self):
        records = [
            _rec(
                "case_a",
                component_count=2,
                primary_component_area=100.0,
                secondary_component_area=25.0,
                ring_violation_count=1,
                border_contact_proxy=True,
            )
        ]
        report = run_evaluation({"records": records, "holdout_case_ids": []})
        case = report["cases"]["case_a"]
        self.assertEqual(case["component_count"], 2)
        self.assertEqual(case["secondary_primary_area_ratio"], 0.25)
        self.assertEqual(case["ring_violation_count"], 1)
        self.assertTrue(case["border_contact_available"])
        self.assertEqual(case["unavailable_reason"], "geometry_supplied")
        self.assertEqual(report["geometry_fill_rate"], 1.0)
        self.assertEqual(report["fill_rate_delta_vs_s854"], 1.0)

    def test_zero_write_flags(self):
        report = run_evaluation({"records": [_rec("case_a")], "holdout_case_ids": []})
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["routing_mode"], "shadow_triage_only")


if __name__ == "__main__":
    unittest.main()
