"""Offline unit tests for the S785 low-cavity threshold replay module."""

from __future__ import annotations

import unittest

from experiments.cmr_low_cavity_threshold_replay import (
    ADEQUATE_CLUSTER,
    EXPECTED_PENDING,
    LOW_AREA_CLUSTER,
    NO_TRANSITION_BASELINE_S783,
    _derive_threshold,
    run_evaluation,
)


def _rec(cid, ref, area, cavity):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }


def _make_records(n):
    records = []
    for i in range(n):
        # Alternate between adequate_area_low_cavity (high area_ratio, low cavity)
        # and low_area (low area_ratio, low cavity).
        if i % 2 == 0:
            records.append(_rec(f"case_adeq_{i:02d}", 1000.0, 600.0, 0.2))
        else:
            records.append(_rec(f"case_low_{i:02d}", 1000.0, 200.0, 0.2))
    return records


class ThresholdReplayTest(unittest.TestCase):
    def test_counts_sum_to_pending(self):
        data = {"records": _make_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_pending"])
        self.assertEqual(report["simulated_total"], EXPECTED_PENDING)
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            EXPECTED_PENDING,
        )

    def test_no_transition_reconciliation(self):
        data = {"records": _make_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(report["no_transition_reconciliation_status"], "ok")
        self.assertEqual(
            report["no_transition_reconciliation_delta_vs_s783"],
            report["simulated_no_transition_total"] - NO_TRANSITION_BASELINE_S783,
        )

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        data = {"records": _make_records(EXPECTED_PENDING - 1)}
        report = run_evaluation(data)
        self.assertEqual(
            report["no_transition_reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )
        self.assertIsNone(report["no_transition_reconciliation_delta_vs_s783"])

    def test_threshold_separation(self):
        out = _derive_threshold([0.6, 0.7], [0.2, 0.3])
        self.assertEqual(out["status"], "ok")
        self.assertAlmostEqual(out["threshold"], 0.45)
        self.assertAlmostEqual(out["margin"], 0.3)

    def test_threshold_unavailable_on_overlap(self):
        out = _derive_threshold([0.4], [0.5])
        self.assertEqual(out["status"], "unavailable_clusters_overlap")
        self.assertIsNone(out["threshold"])

    def test_threshold_unavailable_on_empty_cluster(self):
        out = _derive_threshold([], [0.2])
        self.assertEqual(out["status"], "unavailable_empty_cluster")
        self.assertIsNone(out["threshold"])

    def test_unavailable_metrics_never_imputed(self):
        data = {"records": _make_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        for key in ("concordance_vs_independent_labels", "precision", "ci"):
            self.assertIn(key, report["unavailable"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)

    def test_cluster_threshold_axis(self):
        data = {"records": _make_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(report["threshold"]["adequate_cluster"], ADEQUATE_CLUSTER)
        self.assertEqual(report["threshold"]["low_area_cluster"], LOW_AREA_CLUSTER)
        self.assertEqual(report["threshold"]["axis"], "area_ratio")


if __name__ == "__main__":
    unittest.main()
