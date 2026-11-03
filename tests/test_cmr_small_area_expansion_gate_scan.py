"""Offline unit tests for the S789 small-area expansion gate scan."""

from __future__ import annotations

import unittest

from experiments.cmr_small_area_expansion_gate_scan import (
    EXPECTED_PENDING,
    PENDING_BASELINE,
    run_evaluation,
)


def _record(case_id, area, cavity, reference=1000.0):
    return {
        "case_id": case_id,
        "reference_area": reference,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "quality_before": 80.0,
        "quality_after": 80.0,
        "quality_delta": 0.0,
        "decision": "REVIEW",
        "quality_status": "SCORED",
        "quality_flags_before": [],
        "quality_flags_after": [],
        "quality_subscores_before": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
        "quality_subscores_after": {
            "integrity": 100.0,
            "topology": 80.0,
            "temporal": 60.0,
            "image_support": 100.0,
        },
    }


def _spatial(case_id, decision="unresolved", reason="inside_sax_core"):
    return {
        "case_id": case_id,
        "state": "待筛",
        "decision": decision,
        "reason": reason,
        "rank": 0,
        "core": [0, 8],
    }


class ExpansionGateScanTest(unittest.TestCase):
    def test_candidate_total_matches_denominator(self):
        records = [
            _record("c1", 500.0, 0.3),
            _record("c2", 400.0, 0.5),
            _record("c3", 300.0, 0.2),
        ]
        spatial = {
            "records": [
                _spatial("c1"),
                _spatial("c2", decision="spatial_boundary_candidate",
                         reason="outside_unique_sax_core"),
                _spatial("c3"),
            ]
        }
        report = run_evaluation({"records": records, "spatial_control": spatial})
        self.assertEqual(report["n_records"], 3)
        self.assertEqual(report["expansion_candidate_total"], 2)
        self.assertEqual(report["spatial_gate"]["suppressed_count"], 1)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)

    def test_missing_measurement_not_imputed(self):
        records = [
            _record("c1", 500.0, 0.3),
            {"case_id": "c2", "before_area": None, "before_cavity": None},
        ]
        report = run_evaluation({"records": records, "spatial_control": {"records": []}})
        self.assertIn("c2", report["missing_measurement_cases"])
        self.assertEqual(report["expansion_candidate_total"], 1)
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")

    def test_reconciliation_unavailable_on_denominator_mismatch(self):
        records = [_record("c1", 500.0, 0.3)]
        report = run_evaluation({"records": records, "spatial_control": {"records": []}})
        self.assertFalse(report["stratified_denominator_matches_expected"])
        self.assertEqual(report["candidate_reconciliation_status"],
                         "unavailable_stratified_denominator_mismatch")
        self.assertIsNone(report["candidate_reconciliation_delta_vs_pending_baseline"])

    def test_reconciliation_ok_on_expected_denominator(self):
        records = [_record(f"c{i}", 500.0, 0.3) for i in range(EXPECTED_PENDING)]
        report = run_evaluation({"records": records, "spatial_control": {"records": []}})
        self.assertTrue(report["stratified_denominator_matches_expected"])
        self.assertEqual(report["candidate_reconciliation_status"], "ok")
        self.assertEqual(
            report["candidate_reconciliation_delta_vs_pending_baseline"],
            EXPECTED_PENDING - PENDING_BASELINE,
        )
        self.assertEqual(report["expansion_candidate_total"], EXPECTED_PENDING)

    def test_unavailable_metrics_never_imputed(self):
        records = [_record("c1", 500.0, 0.3)]
        report = run_evaluation({"records": records, "spatial_control": {"records": []}})
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])


if __name__ == "__main__":
    unittest.main()
