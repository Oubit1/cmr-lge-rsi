"""Offline unit tests for the read-only pending denominator reconciliation."""

import unittest

from experiments.cmr_pending_denom_reconcile import (
    _dedup_representative_cases,
    _select_non_holdout,
    run_evaluation,
)


class DedupTests(unittest.TestCase):
    def test_dedup_by_case_and_issue(self):
        records = [
            {"case_id": "a", "issue_type": "x"},
            {"case_id": "a", "issue_type": "x"},
            {"case_id": "a", "issue_type": "y"},
            {"case_id": "b", "issue_type": "x"},
        ]
        out = _dedup_representative_cases(records)
        self.assertEqual(out["input_count"], 4)
        self.assertEqual(out["unique_count"], 3)
        self.assertEqual(out["duplicate_count"], 1)
        self.assertAlmostEqual(out["dedup_ratio"], 0.75)

    def test_dedup_empty(self):
        out = _dedup_representative_cases([])
        self.assertEqual(out["unique_count"], 0)
        self.assertIsNone(out["dedup_ratio"])


class HoldoutTests(unittest.TestCase):
    def test_holdout_excluded(self):
        records = [
            {"case_id": "a", "issue_type": "x"},
            {"case_id": "b", "issue_type": "x"},
        ]
        out = _select_non_holdout(records, ["a"])
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["holdout_excluded_count"], 1)
        self.assertEqual(len(out["selected"]), 1)


class RunEvaluationTests(unittest.TestCase):
    def _data(self):
        records = [
            {"case_id": "h1", "issue_type": "x"},
            {"case_id": "h2", "issue_type": "x"},
            {"case_id": "n1", "issue_type": "x"},
            {"case_id": "n1", "issue_type": "x"},
            {"case_id": "n2", "issue_type": "y"},
        ]
        return {
            "records": records,
            "holdout_case_ids": ["h1", "h2"],
            "full_pending_queue": 61,
            "frozen_split": {
                "frozen_holdout_in_queue": 48,
                "non_holdout_in_queue": 13,
            },
        }

    def test_zero_write_and_denominator(self):
        report = run_evaluation(self._data())
        self.assertFalse(report["label_write"])
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertEqual(report["target_path"], "experiments/cmr_pending_denom_reconcile.py")
        self.assertEqual(report["representative_cases_input_count"], 5)
        self.assertEqual(report["representative_cases_unique_count"], 4)
        self.assertAlmostEqual(report["dedup_ratio"], 0.8)
        self.assertEqual(report["unique_frozen_denominator"], 2)

    def test_gap_attribution(self):
        report = run_evaluation(self._data())
        gap = report["reconciliation"]["gap_13_vs_61"]
        self.assertEqual(gap["status"], "ok")
        self.assertEqual(gap["gap"], 48)
        self.assertTrue(gap["gap_matches_holdout"])

    def test_missing_frozen_split_unavailable(self):
        report = run_evaluation({"records": [], "holdout_case_ids": []})
        gap = report["reconciliation"]["gap_13_vs_61"]
        self.assertEqual(gap["status"], "unavailable_missing_frozen_split")
        self.assertIsNone(report["dedup_ratio"])


if __name__ == "__main__":
    unittest.main()
