"""Offline unit tests for the S795 per-cluster transition replay."""

from __future__ import annotations

import unittest

from experiments.cmr_checked_cohort_cluster_replay_v2 import (
    EXPECTED_CHECKED,
    PENDING_BASELINE,
    run_evaluation,
)
from experiments.cmr_low_cavity_discriminator_scan import (
    NO_TRANSITION_BASELINE_S783,
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
    }


def _synthetic_records(n):
    records = []
    for i in range(n):
        # Vary area ratio and cavity so all three clusters are exercised.
        reference_area = 1000.0
        before_area = 100.0 + (i % 7) * 60.0
        before_cavity = 0.2 + (i % 5) * 0.15
        records.append(
            _record(
                f"case_{i:04d}",
                reference_area,
                before_area,
                before_cavity,
            )
        )
    return records


class TestCheckedCohortClusterReplayV2(unittest.TestCase):
    def test_three_cluster_total_sums_to_denominator(self):
        records = _synthetic_records(EXPECTED_CHECKED)
        report = run_evaluation({"records": records})
        self.assertEqual(report["n_records"], EXPECTED_CHECKED)
        self.assertEqual(report["simulated_total"], EXPECTED_CHECKED)
        self.assertTrue(report["simulated_total_matches_denominator"])
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            EXPECTED_CHECKED,
        )

    def test_reconciliation_against_baselines(self):
        records = _synthetic_records(EXPECTED_CHECKED)
        report = run_evaluation({"records": records})
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "ok")
        self.assertEqual(recon["pending_baseline"], PENDING_BASELINE)
        self.assertEqual(
            recon["s783_no_transition_baseline"],
            NO_TRANSITION_BASELINE_S783,
        )
        self.assertEqual(
            recon["no_transition_delta_vs_pending_baseline"],
            report["simulated_no_transition_total"] - PENDING_BASELINE,
        )
        self.assertEqual(
            recon["no_transition_delta_vs_s783"],
            report["simulated_no_transition_total"] - NO_TRANSITION_BASELINE_S783,
        )

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        records = _synthetic_records(EXPECTED_CHECKED - 1)
        report = run_evaluation({"records": records})
        recon = report["reconciliation"]
        self.assertEqual(
            recon["status"], "unavailable_checked_denominator_mismatch"
        )
        self.assertIsNone(recon["no_transition_delta_vs_pending_baseline"])
        self.assertIsNone(recon["no_transition_delta_vs_s783"])

    def test_suppression_reasons_cover_all_records(self):
        records = _synthetic_records(EXPECTED_CHECKED)
        report = run_evaluation({"records": records})
        reasons = report["suppression_reasons"]
        self.assertEqual(sum(reasons.values()), EXPECTED_CHECKED)
        for key in ("routed", "unrouted", "spatial_gate_suppressed"):
            self.assertIn(key, reasons)

    def test_unavailable_metrics_are_not_imputed(self):
        records = _synthetic_records(EXPECTED_CHECKED)
        report = run_evaluation({"records": records})
        unavailable = report["unavailable"]
        self.assertIn("concordance_vs_independent_labels", unavailable)
        self.assertIn("precision", unavailable)
        self.assertIn("ci", unavailable)
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})


if __name__ == "__main__":
    unittest.main()
