"""Offline unit tests for the read-only shadow CMR pending denominator scan.

No external data, images, masks or network access required.
"""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_pending_denom_scan import (
    EXPECTED_PENDING,
    FULL_PENDING_QUEUE,
    _dedup_ratio,
    _select_non_holdout_pending,
    report_to_json,
    run_evaluation,
)


def _record(case_id, area, cavity):
    return {
        "case_id": case_id,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "decision": "REVIEW",
        "quality_status": "SCORED",
    }


def _pending_records(n=13):
    recs = []
    for i in range(n):
        recs.append(_record("case_pending_%02d" % i, 300.0 + i, 0.3))
    return recs


class SelectionTest(unittest.TestCase):
    def test_selects_exactly_thirteen_non_holdout(self):
        recs = _pending_records(13) + [_record("case_holdout_1", 400.0, 0.4)]
        out = _select_non_holdout_pending(recs, ["case_holdout_1"])
        self.assertEqual(len(out["selected"]), EXPECTED_PENDING)
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["holdout_excluded_count"], 1)

    def test_holdout_excluded_by_case_id(self):
        recs = _pending_records(13) + [_record("case_holdout_1", 400.0, 0.4)]
        out = _select_non_holdout_pending(recs, ["case_holdout_1"])
        ids = [r["case_id"] for r in out["selected"]]
        self.assertNotIn("case_holdout_1", ids)

    def test_duplicate_case_ids_deduped(self):
        recs = _pending_records(13) + [_record("case_pending_00", 300.0, 0.3)]
        out = _select_non_holdout_pending(recs, [])
        self.assertEqual(len(out["selected"]), EXPECTED_PENDING)
        self.assertEqual(out["duplicate_case_ids"], ["case_pending_00"])

    def test_dedup_ratio_reconciles_61_vs_13(self):
        ratio = _dedup_ratio(13, FULL_PENDING_QUEUE)
        self.assertEqual(ratio["full_pending_queue"], 61)
        self.assertEqual(ratio["non_holdout_selected"], 13)
        self.assertAlmostEqual(ratio["dedup_ratio_selected_over_queue"], 13 / 61, places=6)
        self.assertEqual(ratio["status"], "ok")


class EvaluationTest(unittest.TestCase):
    def test_run_evaluation_denominator_and_zero_write(self):
        data = {
            "records": _pending_records(13),
            "holdout_case_ids": ["case_holdout_1"],
            "full_pending_queue": FULL_PENDING_QUEUE,
        }
        report = run_evaluation(data)
        self.assertEqual(report["non_holdout_volumes_scanned"], EXPECTED_PENDING)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertFalse(report["label_write"])
        self.assertNotEqual(report["target_path"], "app/check_auto_apply.py")
        self.assertEqual(report["writes"]["labels"], 0)

    def test_per_cluster_totals_match_denominator(self):
        data = {"records": _pending_records(13), "full_pending_queue": 61}
        report = run_evaluation(data)
        total = sum(report["simulated_transitions_per_cluster"].values())
        self.assertEqual(total, EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_denominator"])

    def test_every_no_transition_has_suppression_reason(self):
        data = {"records": _pending_records(13), "full_pending_queue": 61}
        report = run_evaluation(data)
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertEqual(report["no_transition_without_reason"], [])
        for case in report["cases"].values():
            if case["simulated_transition"] == "no_transition":
                self.assertTrue(case["suppression_reason"])

    def test_unavailable_metrics_not_imputed(self):
        data = {"records": _pending_records(13), "full_pending_queue": 61}
        report = run_evaluation(data)
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")

    def test_report_is_json_serializable(self):
        data = {"records": _pending_records(13), "full_pending_queue": 61}
        report = run_evaluation(data)
        text = report_to_json(report)
        self.assertIn("non_holdout_volumes_scanned", text)


if __name__ == "__main__":
    unittest.main()
