"""Offline unittest for the read-only S791 pending small-area cluster scan."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_small_area_cluster_scan import (
    EXPECTED_PENDING,
    RESIDUAL_CLUSTER_BASELINE,
    run_evaluation,
)


def _record(case_id, reference_area, before_area, before_cavity):
    return {
        "case_id": case_id,
        "reference_area": reference_area,
        "before_area": before_area,
        "before_cavity": before_cavity,
        "after_area": before_area,
        "after_cavity": before_cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
        "quality_flags_before": [],
        "quality_flags_after": [],
    }


def _synthetic_pending(n=EXPECTED_PENDING):
    records = []
    for i in range(n):
        # Alternate small-area (low ratio) and adequate-area volumes with a
        # range of cavity fractions so all three clusters are exercised.
        if i % 3 == 0:
            records.append(_record(f"case_p{i:03d}", 1000.0, 200.0, 0.2))
        elif i % 3 == 1:
            records.append(_record(f"case_p{i:03d}", 1000.0, 600.0, 0.2))
        else:
            records.append(_record(f"case_p{i:03d}", 1000.0, 600.0, 0.9))
    return records


class PendingSmallAreaClusterScanTest(unittest.TestCase):
    def test_denominator_and_total(self):
        report = run_evaluation({"records": _synthetic_pending()})
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertTrue(report["pending_denominator_matches_expected"])
        self.assertEqual(report["simulated_total"], EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_denominator"])

    def test_reconciliation_against_residual_baseline(self):
        report = run_evaluation({"records": _synthetic_pending()})
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "ok")
        self.assertEqual(recon["residual_cluster_baseline"], RESIDUAL_CLUSTER_BASELINE)
        self.assertEqual(
            recon["simulated_checked"]
            + recon["simulated_trash"]
            + recon["simulated_no_transition"],
            EXPECTED_PENDING,
        )
        self.assertIsNotNone(recon["checked_delta_vs_residual_baseline"])
        self.assertIsNotNone(recon["trash_delta_vs_residual_baseline"])
        self.assertIsNotNone(recon["no_transition_delta_vs_residual_baseline"])

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        report = run_evaluation({"records": _synthetic_pending(10)})
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "unavailable_pending_denominator_mismatch")
        self.assertIsNone(recon["checked_delta_vs_residual_baseline"])
        self.assertIsNone(recon["trash_delta_vs_residual_baseline"])
        self.assertIsNone(recon["no_transition_delta_vs_residual_baseline"])

    def test_suppression_reasons_sum_to_denominator(self):
        report = run_evaluation({"records": _synthetic_pending()})
        reasons = report["suppression_reasons"]
        self.assertEqual(sum(reasons.values()), EXPECTED_PENDING)
        self.assertIn("routed", reasons)
        self.assertIn("unrouted", reasons)
        self.assertIn("spatial_gate_suppressed", reasons)

    def test_unavailable_metrics_not_imputed(self):
        report = run_evaluation({"records": _synthetic_pending()})
        unavailable = report["unavailable"]
        self.assertIn("concordance_vs_independent_labels", unavailable)
        self.assertIn("precision", unavailable)
        self.assertIn("ci", unavailable)
        self.assertIn("topology_flags", unavailable)
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertIsNone(report["topology_deltas"]["holes_delta"])
        self.assertIsNone(report["topology_deltas"]["border_contact_delta"])

    def test_zero_write_contract(self):
        report = run_evaluation({"records": _synthetic_pending()})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])
        self.assertTrue(report["module_hash"])


if __name__ == "__main__":
    unittest.main()
