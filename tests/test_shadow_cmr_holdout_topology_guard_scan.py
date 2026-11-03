"""Offline unittest for the read-only holdout topology-guard shadow scanner."""

from __future__ import annotations

import unittest

from experiments.shadow_cmr_holdout_topology_guard_scan import (
    EXPECTED_HOLDOUT,
    GUARD_APPLICABLE_RATE_MIN,
    REASON_POPULATED_RATE_MIN,
    TOPOLOGY_GUARD_SUPPRESSED,
    _select_holdout,
    _topology_guard,
    run_evaluation,
)


def _rec(cid, area, cavity, flags=None):
    return {
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
        "quality_flags_before": list(flags or []),
        "quality_flags_after": list(flags or []),
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


class SelectHoldoutTest(unittest.TestCase):
    def test_selects_only_holdout_and_dedups(self):
        records = [
            _rec("h1", 500.0, 0.3),
            _rec("h1", 500.0, 0.3),
            _rec("n1", 500.0, 0.3),
            _rec("h2", 500.0, 0.3),
        ]
        out = _select_holdout(records, ["h1", "h2"])
        self.assertEqual([r["case_id"] for r in out["selected"]], ["h1", "h2"])
        self.assertEqual(out["holdout_selected"], 0)
        self.assertEqual(out["non_holdout_excluded_count"], 1)
        self.assertEqual(out["duplicate_case_ids"], ["h1"])


class TopologyGuardTest(unittest.TestCase):
    def test_missing_measurement_not_applicable(self):
        g = _topology_guard(None)
        self.assertFalse(g["applicable"])
        self.assertFalse(g["suppressed"])
        self.assertEqual(g["reason"], "missing_measurement")

    def test_below_floor_suppressed(self):
        g = _topology_guard(0.2)
        self.assertTrue(g["applicable"])
        self.assertTrue(g["suppressed"])
        self.assertEqual(g["reason"], TOPOLOGY_GUARD_SUPPRESSED)

    def test_above_floor_routed(self):
        g = _topology_guard(0.9)
        self.assertTrue(g["applicable"])
        self.assertFalse(g["suppressed"])
        self.assertEqual(g["reason"], "routed")


class RunEvaluationTest(unittest.TestCase):
    def _data(self):
        holdout = [f"h{i:02d}" for i in range(EXPECTED_HOLDOUT)]
        records = []
        for i, cid in enumerate(holdout):
            # alternate low/high area ratios so both guard branches are hit
            area = 200.0 if i % 2 == 0 else 900.0
            records.append(_rec(cid, area, 0.3))
        records.append(_rec("non_holdout_1", 500.0, 0.3))
        return {
            "records": records,
            "holdout_case_ids": holdout,
            "spatial_control": {"records": []},
        }

    def test_zero_label_write_and_no_holdout_selected(self):
        report = run_evaluation(self._data())
        self.assertFalse(report["label_write"])
        self.assertEqual(report["holdout_selected"], 0)
        self.assertEqual(report["writes"]["labels"], 0)
        self.assertEqual(report["holdout_volumes_scanned"], EXPECTED_HOLDOUT)
        self.assertTrue(report["holdout_denominator_matches_expected"])

    def test_guard_applicable_rate_meets_threshold(self):
        report = run_evaluation(self._data())
        rate = report["topology_guard"]["applicable_rate"]
        self.assertIsNotNone(rate)
        self.assertGreater(rate, GUARD_APPLICABLE_RATE_MIN)
        self.assertTrue(report["topology_guard"]["applicable_rate_meets_threshold"])

    def test_reason_populated_rate_meets_threshold(self):
        report = run_evaluation(self._data())
        rate = report["no_transition_reason_populated_rate"]
        self.assertIsNotNone(rate)
        self.assertGreater(rate, REASON_POPULATED_RATE_MIN)
        self.assertEqual(report["no_transition_without_reason_count"], 0)

    def test_simulated_total_matches_denominator(self):
        report = run_evaluation(self._data())
        self.assertTrue(report["simulated_total_matches_denominator"])
        self.assertTrue(report["per_cluster_simulated_transition_counts_populated"])

    def test_missing_measurement_reported_not_imputed(self):
        data = self._data()
        data["records"][0]["before_area"] = None
        data["records"][0]["before_cavity"] = None
        report = run_evaluation(data)
        cid = data["holdout_case_ids"][0]
        case = report["cases"][cid]
        self.assertEqual(case["suppression_reason"], "missing_measurement")
        self.assertIsNone(case["area_ratio"])


if __name__ == "__main__":
    unittest.main()
