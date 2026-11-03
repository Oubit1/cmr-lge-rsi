"""Offline unit tests for the read-only pending cluster route shadow scanner."""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_pending_cluster_route import (
    EXPECTED_NON_HOLDOUT,
    REASON_POPULATED_RATE_MIN,
    _select_non_holdout,
    run_evaluation,
)


def _rec(cid, area, cavity, ref=1000.0):
    return {
        "case_id": cid,
        "reference_area": ref,
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


def _fixture():
    holdout = ["case_h%02d" % i for i in range(3)]
    non_holdout = ["case_n%02d" % i for i in range(EXPECTED_NON_HOLDOUT)]
    records = [_rec(c, 400.0 + 10 * i, 0.3) for i, c in enumerate(non_holdout)]
    records += [_rec(c, 500.0, 0.4) for c in holdout]
    spatial = {
        "status": "AVAILABLE",
        "records": [
            {
                "case_id": non_holdout[0],
                "state": "待筛",
                "decision": "spatial_boundary_candidate",
                "reason": "outside_unique_sax_core",
                "rank": 1,
                "core": [3, 11],
            },
            {
                "case_id": non_holdout[1],
                "state": "待筛",
                "decision": "unresolved",
                "reason": "inside_sax_core",
                "rank": 11,
                "core": [3, 11],
            },
        ],
    }
    return {
        "records": records,
        "holdout_case_ids": holdout,
        "spatial_control": spatial,
        "frozen_split": {
            "status": "AVAILABLE",
            "frozen_holdout_in_queue": len(holdout),
            "non_holdout_in_queue": len(non_holdout),
        },
    }


class SelectNonHoldoutTest(unittest.TestCase):
    def test_excludes_holdout_and_dedups(self):
        records = [_rec("a", 400.0, 0.3), _rec("a", 400.0, 0.3), _rec("b", 400.0, 0.3)]
        out = _select_non_holdout(records, ["b"])
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["holdout_excluded_count"], 1)
        self.assertEqual([r["case_id"] for r in out["selected"]], ["a"])
        self.assertEqual(out["duplicate_case_ids"], ["a"])


class RunEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.report = run_evaluation(_fixture())

    def test_denominator_and_holdout(self):
        self.assertEqual(self.report["non_holdout_volumes_scanned"], EXPECTED_NON_HOLDOUT)
        self.assertEqual(self.report["holdout_selected"], 0)
        self.assertFalse(self.report["label_write"])
        self.assertEqual(self.report["writes"]["labels"], 0)
        self.assertNotEqual(
            self.report["target_path"], "app/check_auto_apply.py"
        )

    def test_reason_populated_rate(self):
        rate = self.report["no_transition_reason_populated_rate"]
        self.assertIsNotNone(rate)
        self.assertGreater(rate, REASON_POPULATED_RATE_MIN)
        self.assertEqual(self.report["no_transition_without_reason_count"], 0)

    def test_per_cluster_counts_non_negative(self):
        for cluster, counts in self.report["simulated_transitions"].items():
            for key in ("checked", "trash", "no_transition"):
                self.assertGreaterEqual(counts[key], 0, cluster)
        self.assertEqual(
            self.report["simulated_total"],
            self.report["non_holdout_volumes_scanned"],
        )

    def test_unavailable_metrics_not_imputed(self):
        unavailable = self.report["unavailable"]
        self.assertIn("concordance_vs_independent_labels", unavailable)
        self.assertIn("precision", unavailable)
        self.assertIn("ci", unavailable)
        self.assertEqual(
            self.report["topology_deltas"]["status"], "unavailable_no_masks"
        )

    def test_spatial_suppression_reason(self):
        cases = self.report["cases"]
        suppressed = [
            c for c in cases.values() if c["spatial_gate_suppressed"]
        ]
        self.assertTrue(suppressed)
        for case in suppressed:
            self.assertEqual(case["suppression_reason"], "spatial_gate_suppressed")


if __name__ == "__main__":
    unittest.main()
