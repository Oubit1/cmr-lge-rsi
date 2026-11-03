"""Offline unit tests for the read-only S790 checked-cohort replay."""

from __future__ import annotations

import unittest

from experiments.cmr_checked_cohort_replay import (
    EXPECTED_CHECKED,
    PENDING_BASELINE,
    _reconciliation_table,
    report_to_json,
    run_evaluation,
)
from experiments.cmr_low_cavity_discriminator_scan import (
    NO_TRANSITION_BASELINE_S783,
)


def _record(case_id, before_area, reference_area, cavity, checked_label=None):
    rec = {
        "case_id": case_id,
        "reference_area": reference_area,
        "before_area": before_area,
        "before_cavity": cavity,
        "after_area": before_area,
        "after_cavity": cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
    }
    if checked_label is not None:
        rec["checked_label"] = checked_label
    return rec


def _cohort(n):
    records = []
    for i in range(n):
        records.append(
            _record(
                "case_%03d" % i,
                before_area=300.0 + i,
                reference_area=600.0 + i,
                cavity=0.3,
                checked_label="no_transition",
            )
        )
    return {"records": records}


class ReconciliationTableTest(unittest.TestCase):
    def test_denominator_match_reports_deltas(self):
        table = _reconciliation_table(EXPECTED_CHECKED, 100, 50, 29)
        self.assertTrue(table["checked_denominator_matches_expected"])
        self.assertEqual(table["status"], "ok")
        self.assertEqual(table["simulated_total"], EXPECTED_CHECKED)
        self.assertEqual(
            table["no_transition_delta_vs_s783"],
            100 - NO_TRANSITION_BASELINE_S783,
        )
        self.assertEqual(
            table["no_transition_delta_vs_pending_baseline"],
            100 - PENDING_BASELINE,
        )

    def test_denominator_mismatch_is_unavailable(self):
        table = _reconciliation_table(20, 10, 5, 5)
        self.assertFalse(table["checked_denominator_matches_expected"])
        self.assertEqual(table["status"], "unavailable_checked_denominator_mismatch")
        self.assertIsNone(table["no_transition_delta_vs_s783"])
        self.assertIsNone(table["no_transition_delta_vs_pending_baseline"])


class RunEvaluationTest(unittest.TestCase):
    def test_counts_sum_to_denominator(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        self.assertEqual(report["n_records"], EXPECTED_CHECKED)
        self.assertTrue(report["simulated_total_matches_checked"])
        self.assertEqual(report["simulated_total"], EXPECTED_CHECKED)
        self.assertEqual(
            report["simulated_checked_total"]
            + report["simulated_trash_total"]
            + report["simulated_no_transition_total"],
            EXPECTED_CHECKED,
        )

    def test_per_cluster_counts_sum_to_denominator(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        total = 0
        for cluster, info in report["simulated_transitions"].items():
            total += (
                info["checked"] + info["trash"] + info["no_transition"]
            )
        self.assertEqual(total, EXPECTED_CHECKED)

    def test_agreement_is_monitoring_grade_only(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        agreement = report["agreement_vs_checked_labels"]
        self.assertEqual(agreement["status"], "monitoring_grade_only")
        self.assertIsNotNone(agreement["rate"])

    def test_unavailable_metrics_are_not_imputed(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])

    def test_zero_writes_and_audit_diff(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])

    def test_report_is_json_serializable(self):
        report = run_evaluation(_cohort(EXPECTED_CHECKED))
        text = report_to_json(report)
        self.assertIn("simulated_transitions", text)

    def test_empty_cohort_reports_unavailable_agreement(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        self.assertEqual(
            report["agreement_vs_checked_labels"]["status"],
            "unavailable_no_checked_labels",
        )
        self.assertIsNone(report["agreement_vs_checked_labels"]["rate"])


if __name__ == "__main__":
    unittest.main()
