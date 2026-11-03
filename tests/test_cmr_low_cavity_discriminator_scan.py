"""Offline unittest for the S784 low-cavity discriminator scan.

Runs without external data: builds a synthetic 95-case pending cohort plus a
small spatial-control block and asserts the count invariants, the S783
reconciliation contract, the unavailable-metric contract and the zero-write
audit diff.
"""

from __future__ import annotations

import unittest

from experiments.cmr_low_cavity_discriminator_scan import (
    EXPECTED_PENDING,
    NO_TRANSITION_BASELINE_S783,
    TOPOLOGY_FIELDS,
    run_evaluation,
)


def _make_records(n: int):
    records = []
    for i in range(n):
        records.append(
            {
                "case_id": f"case_{i:04d}",
                "reference_area": 1000.0,
                "before_area": 400.0 + (i % 5) * 10.0,
                "before_cavity": 0.2 + (i % 4) * 0.1,
                "after_area": 400.0 + (i % 5) * 10.0,
                "after_cavity": 0.2 + (i % 4) * 0.1,
                "quality_before": 70.0,
                "quality_after": 70.0,
                "quality_delta": 0.0,
                "decision": "REVIEW",
                "quality_status": "SCORED",
                "quality_flags_before": [],
                "quality_flags_after": [],
                "quality_subscores_before": {
                    "integrity": 100.0,
                    "topology": 80.0,
                    "temporal": 50.0,
                    "image_support": 100.0,
                },
                "quality_subscores_after": {
                    "integrity": 100.0,
                    "topology": 80.0,
                    "temporal": 50.0,
                    "image_support": 100.0,
                },
            }
        )
    return records


def _make_spatial(records):
    spatial_records = []
    for i, rec in enumerate(records):
        if i % 3 == 0:
            spatial_records.append(
                {
                    "case_id": rec["case_id"],
                    "state": "待筛",
                    "decision": "spatial_boundary_candidate",
                    "reason": "outside_unique_sax_core",
                    "rank": i % 12,
                    "core": [0, 10],
                }
            )
        else:
            spatial_records.append(
                {
                    "case_id": rec["case_id"],
                    "state": "待筛",
                    "decision": "unresolved",
                    "reason": "inside_sax_core",
                    "rank": i % 12,
                    "core": [0, 10],
                }
            )
    return {
        "status": "AVAILABLE",
        "rule": "existing_casebook_sax_core_0.60_min3_single_gap",
        "control_gate": "PASS",
        "checked_false_routes": 0,
        "pending_candidates": len(records),
        "records": spatial_records,
    }


class LowCavityDiscriminatorScanTest(unittest.TestCase):
    def setUp(self):
        self.records = _make_records(EXPECTED_PENDING)
        self.data = {
            "records": self.records,
            "spatial_control": _make_spatial(self.records),
        }
        self.report = run_evaluation(self.data)

    def test_three_cluster_counts_sum_to_pending(self):
        total = (
            self.report["simulated_checked_total"]
            + self.report["simulated_trash_total"]
            + self.report["simulated_no_transition_total"]
        )
        self.assertEqual(total, EXPECTED_PENDING)
        self.assertEqual(self.report["simulated_total"], EXPECTED_PENDING)
        self.assertTrue(self.report["simulated_total_matches_pending"])

    def test_per_cluster_transitions_sum_to_cluster_denominator(self):
        for cluster, block in self.report["simulated_transitions"].items():
            subtotal = block["checked"] + block["trash"] + block["no_transition"]
            self.assertEqual(subtotal, block["denominator"], cluster)

    def test_suppression_reasons_sum_to_pending(self):
        reasons = self.report["suppression_reasons"]
        self.assertEqual(sum(reasons.values()), EXPECTED_PENDING)

    def test_s783_reconciliation_contract(self):
        self.assertEqual(
            self.report["no_transition_baseline_s783"],
            NO_TRANSITION_BASELINE_S783,
        )
        self.assertEqual(self.report["no_transition_reconciliation_status"], "ok")
        expected_delta = (
            self.report["simulated_no_transition_total"]
            - NO_TRANSITION_BASELINE_S783
        )
        self.assertEqual(
            self.report["no_transition_reconciliation_delta_vs_s783"],
            expected_delta,
        )

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        short = _make_records(EXPECTED_PENDING - 1)
        report = run_evaluation(
            {"records": short, "spatial_control": _make_spatial(short)}
        )
        self.assertFalse(report["pending_denominator_matches_expected"])
        self.assertIsNone(report["no_transition_reconciliation_delta_vs_s783"])
        self.assertEqual(
            report["no_transition_reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )

    def test_unavailable_metrics_are_not_imputed(self):
        unavailable = self.report["unavailable"]
        for key in (
            "concordance_vs_independent_labels",
            "precision",
            "ci",
            "topology_flags",
        ):
            self.assertIn(key, unavailable)
            self.assertIsInstance(unavailable[key], str)
        for field in TOPOLOGY_FIELDS:
            block = self.report["topology_unavailable_counts"][field]
            self.assertEqual(block["count"], EXPECTED_PENDING)
            self.assertEqual(block["denominator"], EXPECTED_PENDING)

    def test_zero_write_audit_diff(self):
        diff = self.report["audit_diff"]
        self.assertEqual(diff["production_writes"], 0)
        self.assertEqual(diff["changed_cases"], [])
        self.assertEqual(diff["removed_cases"], [])
        self.assertEqual(self.report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertIn("git checkout", self.report["rollback_command"])
        self.assertTrue(self.report["module_hash"])

    def test_per_case_has_simulated_transition_and_reason(self):
        for cid, case in self.report["cases"].items():
            self.assertIn(
                case["simulated_transition"],
                ("checked", "trash", "no_transition"),
            )
            self.assertIn(
                case["routing"],
                ("routed", "unrouted", "spatial_gate_suppressed"),
            )


if __name__ == "__main__":
    unittest.main()
