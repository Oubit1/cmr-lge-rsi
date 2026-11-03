"""Offline unit tests for the S788 read-only checked-cluster replay."""

from __future__ import annotations

import unittest

from experiments.cmr_checked_cluster_replay import (
    EXPECTED_CHECKED,
    _apply_threshold,
    _reconciliation_table,
    run_evaluation,
)
from experiments.cmr_low_cavity_threshold_replay import (
    ADEQUATE_CLUSTER,
    LOW_AREA_CLUSTER,
)


def _rec(cid, area, cavity, checked_label=None):
    rec = {
        "case_id": cid,
        "reference_area": 1000.0,
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
        "quality_subscores_before": {},
        "quality_subscores_after": {},
    }
    if checked_label is not None:
        rec["checked_label"] = checked_label
    return rec


class ApplyThresholdTest(unittest.TestCase):
    def test_adequate_cluster_above_threshold_is_checked(self):
        self.assertEqual(
            _apply_threshold(ADEQUATE_CLUSTER, 0.9, {"threshold": 0.5}), "checked"
        )

    def test_adequate_cluster_below_threshold_is_no_transition(self):
        self.assertEqual(
            _apply_threshold(ADEQUATE_CLUSTER, 0.4, {"threshold": 0.5}),
            "no_transition",
        )

    def test_low_area_cluster_below_threshold_is_trash(self):
        self.assertEqual(
            _apply_threshold(LOW_AREA_CLUSTER, 0.2, {"threshold": 0.5}), "trash"
        )

    def test_unavailable_threshold_is_no_transition(self):
        self.assertEqual(
            _apply_threshold(ADEQUATE_CLUSTER, 0.9, {"threshold": None}),
            "no_transition",
        )

    def test_missing_cluster_is_no_transition(self):
        self.assertEqual(_apply_threshold(None, 0.9, {"threshold": 0.5}), "no_transition")


class ReconciliationTest(unittest.TestCase):
    def test_matching_denominator_reports_delta(self):
        table = _reconciliation_table(EXPECTED_CHECKED, 5, 10, 5)
        self.assertTrue(table["checked_denominator_matches_expected"])
        self.assertEqual(table["status"], "ok")
        self.assertIsNotNone(table["no_transition_delta_vs_s783"])
        self.assertEqual(table["simulated_total"], EXPECTED_CHECKED)

    def test_mismatched_denominator_is_unavailable(self):
        table = _reconciliation_table(EXPECTED_CHECKED + 1, 5, 10, 6)
        self.assertFalse(table["checked_denominator_matches_expected"])
        self.assertEqual(table["status"], "unavailable_checked_denominator_mismatch")
        self.assertIsNone(table["no_transition_delta_vs_s783"])


class RunEvaluationTest(unittest.TestCase):
    def _data(self):
        records = [
            _rec("c1", 900.0, 0.2, "checked"),
            _rec("c2", 800.0, 0.2, "checked"),
            _rec("c3", 100.0, 0.2, "trash"),
            _rec("c4", 50.0, 0.2, "trash"),
        ]
        return {"records": records}

    def test_counts_sum_to_denominator(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["n_records"], 4)
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            4,
        )
        self.assertTrue(report["simulated_total_matches_checked"])

    def test_agreement_is_monitoring_grade(self):
        report = run_evaluation(self._data())
        agreement = report["agreement_vs_checked_labels"]
        self.assertEqual(agreement["status"], "monitoring_grade_only")
        self.assertEqual(agreement["denominator"], 4)
        self.assertIsNotNone(agreement["rate"])

    def test_unavailable_metrics_are_not_imputed(self):
        report = run_evaluation(self._data())
        unavailable = report["unavailable"]
        self.assertIn("concordance_vs_independent_labels", unavailable)
        self.assertIn("precision", unavailable)
        self.assertIn("ci", unavailable)
        self.assertIn("topology_flags", unavailable)

    def test_zero_writes_and_rollback_recorded(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])
        self.assertTrue(report["module_hash"])

    def test_empty_input_is_safe(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        self.assertEqual(
            report["agreement_vs_checked_labels"]["status"],
            "unavailable_no_checked_labels",
        )
        self.assertEqual(
            report["reconciliation"]["status"],
            "unavailable_checked_denominator_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
