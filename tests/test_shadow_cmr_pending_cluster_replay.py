"""Offline unit tests for the read-only per-cluster pending replay scanner.

No external data, no network, no production writes.
"""

import unittest

from experiments.shadow_cmr_pending_cluster_replay import (
    EXPECTED_PENDING,
    ROUTED_REASON,
    run_evaluation,
)


def _checked(cid, area, cavity):
    return {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }


def _pending(cid, area, cavity):
    return {
        "case_id": cid,
        "reference_area": 1000.0,
        "before_area": area,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
    }


class ShadowPendingClusterReplayTest(unittest.TestCase):
    def _data(self):
        checked = [
            _checked("c_%03d" % i, 400.0 + i, 0.3) for i in range(184)
        ]
        pending = [
            _pending("p_%03d" % i, 300.0 + i, 0.4) for i in range(61)
        ]
        return {"checked_records": checked, "records": pending}

    def test_zero_label_write_and_target(self):
        report = run_evaluation(self._data())
        self.assertFalse(report["label_write"])
        self.assertNotEqual(report["target_path"], "app/check_auto_apply.py")
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["writes"]["labels"], 0)

    def test_per_cluster_counts_and_totals(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["n_records"], EXPECTED_PENDING)
        self.assertTrue(report["simulated_total_matches_denominator"])
        for cluster, entry in report["simulated_transitions"].items():
            total = entry["checked"] + entry["trash"] + entry["no_transition"]
            self.assertEqual(total, entry["denominator"])

    def test_every_no_transition_has_reason(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        for case in report["cases"].values():
            if case["simulated_transition"] == "no_transition":
                self.assertTrue(case["suppression_reason"])

    def test_holdout_excluded(self):
        data = self._data()
        data["holdout_case_ids"] = ["p_000", "p_001"]
        report = run_evaluation(data)
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["holdout_excluded_count"], 2)
        self.assertNotIn("p_000", report["cases"])

    def test_unavailable_metrics_not_imputed(self):
        report = run_evaluation(self._data())
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])
        self.assertIsNone(report["topology_deltas"]["components_delta"])

    def test_routed_reason_present(self):
        report = run_evaluation(self._data())
        self.assertIn(ROUTED_REASON, report["suppression_reasons"])


if __name__ == "__main__":
    unittest.main()
