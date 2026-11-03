"""Offline unit tests for the read-only CMR small-area casebook topology-guard
shadow scanner (S837 continuation to the 39-volume adequate_area_low_cavity
pending cluster).

No external data, images, masks or labels are required.
"""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_small_area_casebook_scan import (
    EXPECTED_NON_HOLDOUT,
    GUARD_APPLICABLE_RATE_MIN,
    TOPOLOGY_GUARD_SUPPRESSED,
    _select_non_holdout,
    _topology_guard,
    run_evaluation,
)


def _rec(cid, area, cavity, flags=None):
    return {
        "case_id": cid,
        "before_area": area,
        "reference_area": 1000.0,
        "before_cavity": cavity,
        "after_area": area,
        "after_cavity": cavity,
        "quality_flags_before": list(flags or []),
        "quality_subscores_before": {"image_support": 100.0},
    }


def _cohort(n=39):
    records = []
    for i in range(n):
        # area_ratio ~ 0.3-0.6, cavity low -> adequate_area_low_cavity cluster
        records.append(_rec("case_%03d" % i, 300.0 + i, 0.2 + (i % 5) * 0.05))
    return records


class SelectNonHoldoutTest(unittest.TestCase):
    def test_excludes_holdout_and_dedups(self):
        records = [_rec("a", 300.0, 0.2), _rec("b", 300.0, 0.2), _rec("a", 300.0, 0.2)]
        out = _select_non_holdout(records, ["b"])
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["holdout_excluded_count"], 1)
        self.assertEqual([r["case_id"] for r in out["selected"]], ["a"])
        self.assertEqual(out["duplicate_case_ids"], ["a"])


class TopologyGuardTest(unittest.TestCase):
    def test_missing_measurement_not_applicable(self):
        g = _topology_guard(None)
        self.assertFalse(g["applicable"])
        self.assertFalse(g["suppressed"])

    def test_below_floor_suppressed(self):
        g = _topology_guard(0.3)
        self.assertTrue(g["applicable"])
        self.assertTrue(g["suppressed"])
        self.assertEqual(g["reason"], TOPOLOGY_GUARD_SUPPRESSED)

    def test_above_floor_not_suppressed(self):
        g = _topology_guard(0.8)
        self.assertTrue(g["applicable"])
        self.assertFalse(g["suppressed"])


class RunEvaluationTest(unittest.TestCase):
    def test_holdout_excluded_and_zero_write(self):
        records = _cohort(39)
        holdout = [r["case_id"] for r in records[:5]]
        report = run_evaluation({"records": records, "holdout_case_ids": holdout})
        self.assertEqual(report["holdout_selected"], 0)
        self.assertFalse(report["label_write"])
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertEqual(report["non_holdout_volumes_scanned"], 34)

    def test_guard_applicable_rate_and_reasons(self):
        report = run_evaluation({"records": _cohort(39)})
        self.assertEqual(report["non_holdout_volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertTrue(report["non_holdout_denominator_matches_expected"])
        rate = report["topology_guard"]["applicable_rate"]
        self.assertIsNotNone(rate)
        self.assertGreater(rate, GUARD_APPLICABLE_RATE_MIN)
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertTrue(report["per_cluster_simulated_transition_counts_populated"])
        self.assertTrue(report["simulated_total_matches_denominator"])

    def test_topology_deltas_unavailable(self):
        report = run_evaluation({"records": _cohort(39)})
        self.assertEqual(report["topology_deltas"]["status"], "unavailable_no_masks")
        self.assertIsNone(report["topology_deltas"]["components_delta"])
        self.assertIn("concordance_vs_independent_labels", report["unavailable"])

    def test_missing_measurement_gets_reason(self):
        records = _cohort(39)
        records[0].pop("before_area")
        records[0].pop("reference_area")
        report = run_evaluation({"records": records})
        self.assertEqual(report["no_transition_without_reason_count"], 0)
        self.assertGreaterEqual(report["suppression_reasons"].get("missing_measurement", 0), 1)


if __name__ == "__main__":
    unittest.main()
