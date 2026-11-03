"""Offline unit tests for the read-only S830 pending excess-component scan."""

from __future__ import annotations

import unittest

from experiments.cmr_pending_excess_comp_volume_scan_exec_13 import (
    EXPECTED_PENDING,
    run_evaluation,
)


def _record(cid, area, cavity, **extra):
    rec = {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }
    rec.update(extra)
    return rec


def _thirteen_records():
    recs = []
    for i in range(EXPECTED_PENDING):
        recs.append(
            _record(
                f"case_{i:02d}",
                400.0 + i,
                0.3,
                secondary_component_count=1 + (i % 3),
                primary_component_area=400.0 + i,
                secondary_component_area=20.0 + i,
                secondary_adjacent_to_primary=(i % 2 == 0),
                hole_count=i % 2,
                touches_border=(i % 4 == 0),
                centroid_drift=0.1 * i,
            )
        )
    return recs


class PendingExcessCompScanExec13Test(unittest.TestCase):
    def test_denominator_and_zero_write(self):
        report = run_evaluation({"records": _thirteen_records()})
        self.assertEqual(report["non_holdout_volumes_scanned"], EXPECTED_PENDING)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertFalse(report["label_write"])
        self.assertNotEqual(report["target_path"], "app/check_auto_apply.py")
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})

    def test_simulated_total_matches_denominator(self):
        report = run_evaluation({"records": _thirteen_records()})
        self.assertTrue(report["simulated_total_matches_denominator"])
        self.assertEqual(report["simulated_total"], EXPECTED_PENDING)
        for c, v in report["simulated_transitions_per_cluster"].items():
            self.assertGreaterEqual(v, 0, c)

    def test_every_no_transition_has_reason(self):
        report = run_evaluation({"records": _thirteen_records()})
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        for cid, case in report["cases"].items():
            if case["simulated_transition"] == "no_transition":
                self.assertTrue(case["suppression_reason"], cid)

    def test_secondary_fields_reported(self):
        report = run_evaluation({"records": _thirteen_records()})
        first = report["cases"]["case_00"]["secondary_components"]
        self.assertEqual(first["secondary_component_count"], 1)
        self.assertAlmostEqual(first["secondary_primary_area_ratio"], 20.0 / 400.0)
        self.assertTrue(first["secondary_adjacent_to_primary"])
        self.assertEqual(first["hole_count"], 0)
        self.assertTrue(first["touches_border"])
        self.assertAlmostEqual(first["centroid_drift"], 0.0)
        self.assertTrue(first["secondary_fields_available"])

    def test_missing_secondary_fields_unavailable_not_imputed(self):
        recs = [_record("case_x", 400.0, 0.3)]
        report = run_evaluation({"records": recs})
        sec = report["cases"]["case_x"]["secondary_components"]
        self.assertIsNone(sec["secondary_component_count"])
        self.assertIsNone(sec["secondary_primary_area_ratio"])
        self.assertFalse(sec["secondary_fields_available"])
        self.assertEqual(report["secondary_component_unavailable_count"], 1)

    def test_unavailable_metrics_not_imputed(self):
        report = run_evaluation({"records": _thirteen_records()})
        self.assertEqual(
            report["unavailable"]["concordance_vs_independent_labels"],
            "no independent reference labels supplied",
        )
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])

    def test_denominator_mismatch_marks_reconciliation_unavailable(self):
        report = run_evaluation({"records": _thirteen_records()[:5]})
        self.assertFalse(report["pending_denominator_matches_expected"])
        self.assertEqual(
            report["reconciliation_status"],
            "unavailable_pending_denominator_mismatch",
        )
        self.assertIsNone(report["no_transition_reconciliation_delta_vs_s783"])

    def test_empty_input_is_safe(self):
        report = run_evaluation({})
        self.assertEqual(report["non_holdout_volumes_scanned"], 0)
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertFalse(report["label_write"])


if __name__ == "__main__":
    unittest.main()
