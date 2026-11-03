"""Offline tests for the read-only cavity-context cluster shadow counts."""

import unittest

from experiments.cmr_cavity_cluster_shadow_counts import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
    _classify,
    module_hash,
    run_evaluation,
)


class ClassifyTests(unittest.TestCase):
    def test_adequate_area_low_cavity(self):
        self.assertEqual(_classify(0.9, 0.2), "adequate_area_low_cavity")

    def test_low_area_adequate_cavity(self):
        self.assertEqual(_classify(0.2, 0.9), "low_area_adequate_cavity")

    def test_low_area_low_cavity(self):
        self.assertEqual(_classify(0.2, 0.2), "low_area_low_cavity")

    def test_outside_clusters(self):
        self.assertIsNone(_classify(0.9, 0.9))

    def test_missing_measurement_not_imputed(self):
        self.assertIsNone(_classify(None, 0.2))
        self.assertIsNone(_classify(0.9, None))

    def test_boundary_values(self):
        self.assertEqual(_classify(AREA_RATIO_MIN, CAVITY_MAX), None)
        self.assertEqual(_classify(AREA_RATIO_MIN, CAVITY_MAX - 0.01), "adequate_area_low_cavity")


class RunEvaluationTests(unittest.TestCase):
    def _data(self):
        return {
            "records": [
                {"case_id": "a", "reference_area": 1000.0, "before_area": 900.0, "before_cavity": 0.2},
                {"case_id": "b", "reference_area": 1000.0, "before_area": 200.0, "before_cavity": 0.9},
                {"case_id": "c", "reference_area": 1000.0, "before_area": 200.0, "before_cavity": 0.2},
                {"case_id": "d", "reference_area": 1000.0, "before_area": 900.0, "before_cavity": 0.9},
                {"case_id": "e", "reference_area": 1000.0, "before_area": None, "before_cavity": 0.2},
            ]
        }

    def test_cluster_counts_and_denominators(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["n_records"], 5)
        for cluster in CLUSTERS:
            self.assertEqual(report["cluster_counts"][cluster]["denominator"], 5)
        self.assertEqual(report["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_low_cavity"]["count"], 1)

    def test_missing_measurement_not_imputed_as_zero(self):
        report = run_evaluation(self._data())
        self.assertIn("e", report["missing_measurement_cases"])
        self.assertIn("e", report["unclassified_cases"])
        self.assertIsNone(report["cases"]["e"]["cluster"])

    def test_suppression_coverage_unavailable(self):
        report = run_evaluation(self._data())
        for cond in SUPPRESSION_CONDITIONS:
            entry = report["suppression_coverage"][cond]
            self.assertIsNone(entry["count"])
            self.assertEqual(entry["denominator"], 5)
            self.assertEqual(entry["status"], "unavailable_no_masks_or_geometry")

    def test_zero_writes_and_rollback(self):
        report = run_evaluation(self._data())
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])

    def test_concordance_ci_unavailable(self):
        report = run_evaluation(self._data())
        self.assertIn("concordance", report["unavailable"])
        self.assertIn("ci", report["unavailable"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())
        self.assertEqual(len(module_hash()), 64)

    def test_empty_records(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        for cluster in CLUSTERS:
            self.assertEqual(report["cluster_counts"][cluster]["count"], 0)


if __name__ == "__main__":
    unittest.main()
