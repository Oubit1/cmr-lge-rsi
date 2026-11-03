"""Offline tests for the read-only cavity-context cluster holdout transition check."""

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_transition_check import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
    _classify,
    _simulate_transition,
    audit_diff,
    module_hash,
    report_to_json,
    run_evaluation,
)


class ClassifyTests(unittest.TestCase):
    def test_adequate_area_low_cavity(self):
        self.assertEqual(_classify(0.8, 0.2), "adequate_area_low_cavity")

    def test_low_area_adequate_cavity(self):
        self.assertEqual(_classify(0.3, 0.9), "low_area_adequate_cavity")

    def test_low_area_low_cavity(self):
        self.assertEqual(_classify(0.3, 0.2), "low_area_low_cavity")

    def test_outside_clusters(self):
        self.assertIsNone(_classify(0.8, 0.9))

    def test_missing_measurement_not_imputed(self):
        self.assertIsNone(_classify(None, 0.2))
        self.assertIsNone(_classify(0.8, None))

    def test_threshold_boundaries(self):
        # area_ratio == AREA_RATIO_MIN is not low_area; cavity == CAVITY_MAX is not low_cavity
        self.assertEqual(_classify(AREA_RATIO_MIN, CAVITY_MAX), None)
        self.assertEqual(_classify(AREA_RATIO_MIN, CAVITY_MAX - 0.01), "adequate_area_low_cavity")


class TransitionTests(unittest.TestCase):
    def test_low_area_low_cavity_is_trash(self):
        self.assertEqual(_simulate_transition("low_area_low_cavity", 0.3, 0.2), "trash")

    def test_other_clusters_are_checked(self):
        self.assertEqual(_simulate_transition("adequate_area_low_cavity", 0.8, 0.2), "checked")
        self.assertEqual(_simulate_transition("low_area_adequate_cavity", 0.3, 0.9), "checked")

    def test_unclassified_no_transition(self):
        self.assertEqual(_simulate_transition(None, None, None), "no_transition")


class RunEvaluationTests(unittest.TestCase):
    def _data(self):
        return {
            "records": [
                {"case_id": "a", "reference_area": 1000.0, "before_area": 800.0, "before_cavity": 0.2},
                {"case_id": "b", "reference_area": 1000.0, "before_area": 300.0, "before_cavity": 0.9},
                {"case_id": "c", "reference_area": 1000.0, "before_area": 300.0, "before_cavity": 0.2},
                {"case_id": "d", "reference_area": 1000.0, "before_area": 800.0, "before_cavity": 0.9},
                {"case_id": "e", "reference_area": 0.0, "before_area": 100.0, "before_cavity": 0.2},
            ]
        }

    def test_cluster_counts_and_denominators(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["n_records"], 5)
        self.assertEqual(report["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_low_cavity"]["count"], 1)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 5)

    def test_simulated_transitions(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["simulated_transitions"]["low_area_low_cavity"]["trash"], 1)
        self.assertEqual(report["simulated_transitions"]["adequate_area_low_cavity"]["checked"], 1)
        self.assertEqual(report["simulated_transitions"]["low_area_adequate_cavity"]["checked"], 1)

    def test_missing_measurement_not_imputed(self):
        report = run_evaluation(self._data())
        self.assertIn("e", report["missing_measurement_cases"])
        self.assertIn("d", report["unclassified_cases"])

    def test_suppression_and_concordance_unavailable(self):
        report = run_evaluation(self._data())
        for cond in SUPPRESSION_CONDITIONS:
            self.assertIsNone(report["suppression_coverage"][cond]["count"])
        self.assertIn("concordance", report["unavailable"])
        self.assertIn("ci", report["unavailable"])

    def test_zero_writes_and_rollback(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())
        self.assertEqual(len(module_hash()), 64)

    def test_report_is_json_serializable(self):
        report = run_evaluation(self._data())
        self.assertIsInstance(report_to_json(report), str)
        json.loads(report_to_json(report))

    def test_empty_records(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["count"], 0)


class AuditDiffTests(unittest.TestCase):
    def test_baseline_none(self):
        diff = audit_diff(None, {"cases": {"a": 1}})
        self.assertEqual(diff["baseline"], "none")
        self.assertEqual(diff["production_writes"], 0)

    def test_changed_and_added(self):
        prev = {"module_hash": "x", "cases": {"a": 1, "b": 2}}
        cur = {"module_hash": "y", "cases": {"a": 9, "c": 3}}
        diff = audit_diff(prev, cur)
        self.assertEqual(diff["changed_cases"], ["a"])
        self.assertEqual(diff["added_cases"], ["c"])
        self.assertEqual(diff["removed_cases"], ["b"])


if __name__ == "__main__":
    unittest.main()
