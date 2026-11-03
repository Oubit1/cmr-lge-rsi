"""Offline tests for the read-only cavity-vs-myo delta-ratio router (S758)."""

import unittest

from experiments.cmr_cavity_cluster_holdout_transition_check import CLUSTERS
from experiments.cmr_cavity_delta_ratio_router import (
    TARGET_CLUSTER,
    _delta_ratio,
    module_hash,
    report_to_json,
    run_evaluation,
)


def _rec(cid, ref, before_area, before_cavity, after_area=None, after_cavity=None):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": before_area,
        "before_cavity": before_cavity,
        "after_area": before_area if after_area is None else after_area,
        "after_cavity": before_cavity if after_cavity is None else after_cavity,
    }


class DeltaRatioRouterTest(unittest.TestCase):
    def test_cluster_counts_and_denominators(self):
        data = {
            "records": [
                _rec("a", 1000.0, 900.0, 0.2),   # adequate area, low cavity
                _rec("b", 1000.0, 300.0, 0.5),   # low area, adequate cavity
                _rec("c", 1000.0, 300.0, 0.2),   # low area, low cavity
            ]
        }
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], 3)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 3)
        self.assertEqual(report["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_low_cavity"]["count"], 1)

    def test_delta_ratio_and_trash_transition(self):
        # adequate area, low cavity, cavity delta > myo delta -> trash transition
        data = {
            "records": [
                _rec("t", 1000.0, 900.0, 0.2, after_area=880.0, after_cavity=0.5),
            ]
        }
        report = run_evaluation(data)
        routing = report["delta_ratio_routing"]
        self.assertEqual(routing["target_cluster"], TARGET_CLUSTER)
        self.assertEqual(routing["cavity_delta_gt_myo_delta_count"], 1)
        self.assertIn("t", routing["cavity_delta_gt_myo_delta_cases"])
        self.assertEqual(routing["simulated_trash_transitions"], 1)
        self.assertEqual(routing["denominator"], 1)

    def test_missing_measurement_not_imputed(self):
        data = {"records": [{"case_id": "m", "reference_area": 1000.0}]}
        report = run_evaluation(data)
        self.assertIn("m", report["missing_measurement_cases"])
        self.assertIsNone(report["cases"]["m"]["area_ratio"])
        self.assertIsNone(report["cases"]["m"]["delta_ratio"]["myo_delta"])

    def test_zero_myo_delta_ratio_is_none(self):
        rec = _rec("z", 1000.0, 500.0, 0.2, after_area=500.0, after_cavity=0.4)
        delta = _delta_ratio(rec)
        self.assertEqual(delta["myo_delta"], 0.0)
        self.assertIsNone(delta["cavity_myo_delta_ratio"])
        self.assertTrue(delta["cavity_delta_gt_myo_delta"])

    def test_zero_production_writes_and_unavailable_metrics(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("concordance", report["unavailable"])
        self.assertIn("precision", report["unavailable"])
        self.assertIn("ci", report["unavailable"])
        self.assertIn("git checkout", report["rollback_command"])
        self.assertEqual(report["module_hash"], module_hash())

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": [_rec("a", 1000.0, 900.0, 0.2)]})
        self.assertIsInstance(report_to_json(report), str)


if __name__ == "__main__":
    unittest.main()
