"""Offline unit tests for the read-only S781 pending-gate replay."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_gate_replay import (
    EXPECTED_PENDING,
    UNROUTED_BASELINE,
    run_evaluation,
)


def _rec(cid, ref, area, cavity, **extra):
    rec = {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }
    rec.update(extra)
    return rec


def _spatial(cid, decision, reason, rank, core=None):
    rec = {"case_id": cid, "state": "待筛", "decision": decision, "reason": reason, "rank": rank}
    if core is not None:
        rec["core"] = core
    return rec


class PendingGateReplayTest(unittest.TestCase):
    def test_empty_input_is_safe(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        self.assertEqual(report["simulated_total"], 0)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)

    def test_cluster_counts_and_transitions_sum_to_pending(self):
        records = [
            _rec("c1", 1000.0, 900.0, 0.2),   # adequate area, low cavity
            _rec("c2", 1000.0, 300.0, 0.2),   # low area, low cavity
            _rec("c3", 1000.0, 300.0, 0.6),   # low area, high cavity -> unrouted
        ]
        report = run_evaluation({"records": records})
        self.assertEqual(report["n_records"], 3)
        self.assertEqual(report["simulated_total"], report["classified_total"])
        self.assertEqual(report["unrouted_count"], 1)
        self.assertEqual(report["unrouted_cases"], ["c3"])

    def test_spatial_gate_suppresses_candidate(self):
        records = [
            _rec("c1", 1000.0, 900.0, 0.2),
            _rec("c2", 1000.0, 900.0, 0.2),
        ]
        spatial = {
            "records": [
                _spatial("c1", "spatial_boundary_candidate", "outside_unique_sax_core", 1, [0, 1]),
                _spatial("c2", "unresolved", "inside_sax_core", 0, [0, 1]),
            ]
        }
        report = run_evaluation({"records": records, "spatial_control": spatial})
        self.assertEqual(report["spatial_gate"]["candidate_count"], 1)
        self.assertEqual(report["spatial_gate"]["suppressed_count"], 1)
        self.assertTrue(report["cases"]["c1"]["spatial_gate_suppressed"])
        self.assertFalse(report["cases"]["c2"]["spatial_gate_suppressed"])

    def test_topology_flags_unavailable_when_absent(self):
        records = [_rec("c1", 1000.0, 900.0, 0.2)]
        report = run_evaluation({"records": records})
        for field, info in report["cases"]["c1"]["topology"].items():
            self.assertIsNone(info["value"])
            self.assertEqual(info["status"], "unavailable_no_masks_or_geometry")
        self.assertEqual(report["topology_unavailable_counts"]["holes"]["count"], 1)

    def test_topology_flags_read_when_supplied(self):
        records = [_rec("c1", 1000.0, 900.0, 0.2, holes=2, connected_components=1)]
        report = run_evaluation({"records": records})
        self.assertEqual(report["cases"]["c1"]["topology"]["holes"]["value"], 2)
        self.assertEqual(report["cases"]["c1"]["topology"]["holes"]["status"], "ok")
        self.assertEqual(report["topology_unavailable_counts"]["holes"]["count"], 0)

    def test_reconciliation_delta_available_at_expected_denominator(self):
        records = [_rec(f"c{i}", 1000.0, 900.0, 0.2) for i in range(EXPECTED_PENDING)]
        report = run_evaluation({"records": records})
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertEqual(report["unrouted_count"], 0)
        self.assertEqual(
            report["unrouted_reconciliation_delta_vs_22"],
            0 - UNROUTED_BASELINE,
        )
        self.assertEqual(report["unrouted_reconciliation_status"], "ok")

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        records = [_rec("c1", 1000.0, 900.0, 0.2)]
        report = run_evaluation({"records": records})
        self.assertIsNone(report["unrouted_reconciliation_delta_vs_22"])
        self.assertEqual(
            report["unrouted_reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )

    def test_missing_measurement_not_imputed(self):
        records = [{"case_id": "c1", "reference_area": 1000.0}]
        report = run_evaluation({"records": records})
        self.assertIn("c1", report["missing_measurement_cases"])
        self.assertIsNone(report["cases"]["c1"]["area_ratio"])

    def test_unavailable_truth_reported(self):
        report = run_evaluation({"records": []})
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])
        self.assertIn("rollback_command", report)
        self.assertTrue(report["module_hash"])


if __name__ == "__main__":
    unittest.main()
