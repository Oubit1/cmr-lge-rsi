"""Offline unittest for the LGE MI continuity reference-signal scan."""

from __future__ import annotations

import unittest

from experiments.lge_mi_continuity_reference_scan_exec_13 import (
    AGREEMENT_MIN,
    EXPECTED_NON_HOLDOUT,
    REASON_CONTINUOUS,
    REASON_OVERSEGMENTED,
    REASON_UNDERSEGMENTED,
    run_evaluation,
)


def _rec(cid, ref, before, after=None):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": before,
        "after_area": after if after is not None else before,
        "before_cavity": 0.3,
        "after_cavity": 0.3,
        "quality_before": 80.0,
        "quality_after": 80.0,
        "quality_delta": 0.0,
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {"integrity": 100.0},
        "quality_subscores_after": {"integrity": 100.0},
    }


def _payload(records, holdout=None):
    return {
        "records": records,
        "holdout_case_ids": holdout or [],
        "spatial_control": {"records": []},
    }


class ContinuityReferenceScanTest(unittest.TestCase):
    def test_continuous_case_gets_continuous_reason(self):
        data = _payload([_rec("c1", 1000.0, 800.0)])
        report = run_evaluation(data)
        case = report["cases"]["c1"]
        self.assertTrue(case["continuous_with_reference"])
        self.assertEqual(case["suppression_reason"], REASON_CONTINUOUS)

    def test_undersegmented_case_reason(self):
        data = _payload([_rec("c2", 1000.0, 100.0)])
        report = run_evaluation(data)
        case = report["cases"]["c2"]
        self.assertFalse(case["continuous_with_reference"])
        self.assertEqual(case["suppression_reason"], REASON_UNDERSEGMENTED)

    def test_oversegmented_case_reason(self):
        data = _payload([_rec("c3", 100.0, 1000.0)])
        report = run_evaluation(data)
        case = report["cases"]["c3"]
        self.assertFalse(case["continuous_with_reference"])
        self.assertEqual(case["suppression_reason"], REASON_OVERSEGMENTED)

    def test_no_reason_count_is_zero(self):
        data = _payload([
            _rec("c1", 1000.0, 800.0),
            _rec("c2", 1000.0, 100.0),
            _rec("c3", 100.0, 1000.0),
        ])
        report = run_evaluation(data)
        self.assertEqual(report["no_reason_count"], 0)
        for case in report["cases"].values():
            self.assertTrue(case["suppression_reason"])

    def test_agreement_meets_threshold(self):
        records = [_rec(f"c{i}", 1000.0, 800.0) for i in range(10)]
        records += [_rec("bad1", 1000.0, 100.0), _rec("bad2", 100.0, 1000.0)]
        records += [_rec("ok1", 1000.0, 900.0)]
        report = run_evaluation(_payload(records))
        self.assertGreaterEqual(report["cross_slice_continuity_agreement"], AGREEMENT_MIN)
        self.assertTrue(report["cross_slice_continuity_agreement_meets_threshold"])

    def test_holdout_excluded_and_zero_label_write(self):
        records = [_rec("c1", 1000.0, 800.0), _rec("h1", 1000.0, 800.0)]
        report = run_evaluation(_payload(records, holdout=["h1"]))
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 1)
        self.assertFalse(report["label_write"])
        self.assertNotIn("h1", report["cases"])

    def test_expected_denominator_flag(self):
        records = [_rec(f"c{i}", 1000.0, 800.0) for i in range(EXPECTED_NON_HOLDOUT)]
        report = run_evaluation(_payload(records))
        self.assertTrue(report["non_holdout_denominator_matches_expected"])

    def test_missing_reference_reports_unavailable(self):
        rec = _rec("c1", 1000.0, 800.0)
        rec.pop("reference_area")
        report = run_evaluation(_payload([rec]))
        case = report["cases"]["c1"]
        self.assertFalse(case["continuous_with_reference"])
        self.assertTrue(case["suppression_reason"])
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])


if __name__ == "__main__":
    unittest.main()
