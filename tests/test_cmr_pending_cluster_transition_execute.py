"""Offline tests for the read-only S783 cluster-transition shadow report."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_cluster_transition_execute import (
    EXPECTED_PENDING,
    UNROUTED_BASELINE,
    run_evaluation,
)


def _record(cid, ref, area, cavity):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }


def _pending_records(n):
    recs = []
    for i in range(n):
        # Alternate a routed (small cavity) and an unrouted (cavity > 0.5) case.
        if i % 4 == 0:
            recs.append(_record(f"case_{i:04d}", 1000.0, 500.0, 0.8))
        else:
            recs.append(_record(f"case_{i:04d}", 1000.0, 500.0, 0.2))
    return recs


class PendingClusterTransitionExecuteTest(unittest.TestCase):
    def test_simulated_total_matches_pending_denominator(self):
        data = {"records": _pending_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertTrue(report["pending_denominator_matches_expected"])
        self.assertEqual(report["simulated_total"], EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_pending"])
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            EXPECTED_PENDING,
        )

    def test_unrouted_reconciliation_against_22(self):
        data = {"records": _pending_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(report["unrouted_reconciliation_status"], "ok")
        self.assertEqual(
            report["unrouted_reconciliation_delta_vs_22"],
            report["unrouted_count"] - UNROUTED_BASELINE,
        )

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        data = {"records": _pending_records(EXPECTED_PENDING - 1)}
        report = run_evaluation(data)
        self.assertFalse(report["pending_denominator_matches_expected"])
        self.assertIsNone(report["unrouted_reconciliation_delta_vs_22"])
        self.assertEqual(
            report["unrouted_reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )

    def test_metrics_reported_unavailable_without_labels(self):
        data = {"records": _pending_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        for key in ("concordance_vs_independent_labels", "precision", "ci"):
            self.assertIn("no independent reference labels supplied", report["unavailable"][key])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)

    def test_suppression_reasons_sum_to_pending(self):
        data = {"records": _pending_records(EXPECTED_PENDING)}
        report = run_evaluation(data)
        self.assertEqual(sum(report["suppression_reasons"].values()), EXPECTED_PENDING)


if __name__ == "__main__":
    unittest.main()
