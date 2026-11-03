"""Offline unittest for the S792 checked-cohort cluster replay.

Runs without external data: builds a small synthetic checked cohort plus a
spatial_control block and asserts the three-cluster simulated totals, the
reconciliation contract, the unavailable-metric contract and the zero-write
contract.
"""

from __future__ import annotations

import unittest

from experiments.cmr_checked_cohort_cluster_replay import (
    EXPECTED_CHECKED,
    PENDING_BASELINE,
    run_evaluation,
)
from experiments.cmr_low_cavity_discriminator_scan import (
    NO_TRANSITION_BASELINE_S783,
)


def _record(case_id, reference_area, before_area, before_cavity, checked_label=None):
    rec = {
        "case_id": case_id,
        "reference_area": reference_area,
        "before_area": before_area,
        "before_cavity": before_cavity,
        "after_area": before_area,
        "after_cavity": before_cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
    }
    if checked_label is not None:
        rec["checked_label"] = checked_label
    return rec


def _cohort(n):
    records = []
    for i in range(n):
        # Alternate small-area and adequate-area volumes so both clusters are
        # exercised; cavity stays inside the S759 cavity_max band.
        if i % 2 == 0:
            records.append(_record(f"case_{i:04d}", 1000.0, 200.0, 0.3, "trash"))
        else:
            records.append(_record(f"case_{i:04d}", 1000.0, 900.0, 0.3, "checked"))
    return records


class CheckedCohortClusterReplayTest(unittest.TestCase):
    def test_three_cluster_totals_sum_to_denominator(self):
        data = {"records": _cohort(EXPECTED_CHECKED)}
        report = run_evaluation(data)
        total = (
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"]
        )
        self.assertEqual(total, EXPECTED_CHECKED)
        self.assertTrue(report["simulated_total_matches_checked"])
        self.assertEqual(report["n_records"], EXPECTED_CHECKED)

    def test_reconciliation_against_pending_and_s783(self):
        data = {"records": _cohort(EXPECTED_CHECKED)}
        report = run_evaluation(data)
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "ok")
        self.assertEqual(recon["pending_baseline"], PENDING_BASELINE)
        self.assertEqual(
            recon["s783_no_transition_baseline"], NO_TRANSITION_BASELINE_S783
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
        data = {"records": _cohort(EXPECTED_CHECKED - 1)}
        report = run_evaluation(data)
        recon = report["reconciliation"]
        self.assertEqual(recon["status"], "unavailable_checked_denominator_mismatch")
        self.assertIsNone(recon["no_transition_delta_vs_pending_baseline"])
        self.assertIsNone(recon["no_transition_delta_vs_s783"])

    def test_unavailable_metrics_are_not_imputed(self):
        data = {"records": _cohort(EXPECTED_CHECKED)}
        report = run_evaluation(data)
        unavailable = report["unavailable"]
        for key in (
            "concordance_vs_independent_labels",
            "precision",
            "ci",
            "topology_flags",
        ):
            self.assertIn(key, unavailable)
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertEqual(
            report["agreement_vs_checked_labels"]["status"], "monitoring_grade_only"
        )

    def test_zero_write_contract(self):
        data = {"records": _cohort(EXPECTED_CHECKED)}
        report = run_evaluation(data)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        # The audit_diff helper returns a structured zero-write diff, not an
        # empty dict. Assert the actual zero-write contract: no changed cases
        # and zero writes recorded in the diff.
        diff = report["audit_diff"]
        self.assertIsInstance(diff, dict)
        self.assertEqual(diff.get("changed_cases"), [])
        self.assertEqual(diff.get("writes", {}).get("labels", 0), 0)
        self.assertEqual(diff.get("writes", {}).get("code", 0), 0)
        self.assertEqual(diff.get("writes", {}).get("datasets", 0), 0)
        self.assertIn("rollback_command", report)
        self.assertIn("module_hash", report)


if __name__ == "__main__":
    unittest.main()
