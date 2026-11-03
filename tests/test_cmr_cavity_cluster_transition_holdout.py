"""Offline tests for the read-only cavity-context cluster transition evaluation."""

from __future__ import annotations

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_transition_check import (
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
)
from experiments.cmr_cavity_cluster_transition_holdout import (
    COHORT_LABEL,
    ROLLBACK_COMMAND,
    module_hash,
    report_to_json,
    run_evaluation,
)


def _rec(cid, ref, area, cavity):
    return {
        "case_id": cid,
        "reference_area": ref,
        "before_area": area,
        "before_cavity": cavity,
    }


class ClusterTransitionHoldoutTest(unittest.TestCase):
    def setUp(self):
        # Synthetic records covering each cluster plus a missing measurement.
        self.data = {
            "records": [
                _rec("c_adequate_low_cav", 1000.0, 900.0, 0.1),
                _rec("c_low_area_adequate_cav", 1000.0, 100.0, 0.9),
                _rec("c_low_area_low_cav", 1000.0, 100.0, 0.1),
                _rec("c_missing", 0.0, 100.0, 0.1),
            ]
        }

    def test_cluster_counts_and_denominators(self):
        report = run_evaluation(self.data)
        self.assertEqual(report["n_records"], 4)
        for c in CLUSTERS:
            self.assertIn(c, report["cluster_counts"])
            self.assertEqual(report["cluster_counts"][c]["denominator"], 4)
        total = sum(report["cluster_counts"][c]["count"] for c in CLUSTERS)
        self.assertEqual(total, 3)
        self.assertIn("c_missing", report["missing_measurement_cases"])

    def test_simulated_transitions_have_denominators(self):
        report = run_evaluation(self.data)
        for c in CLUSTERS:
            entry = report["simulated_transitions"][c]
            self.assertEqual(entry["denominator"], report["cluster_counts"][c]["count"])
            self.assertEqual(
                entry["checked"] + entry["trash"] + entry["no_transition"],
                entry["denominator"],
            )
            self.assertEqual(entry["cohort"], COHORT_LABEL)

    def test_suppression_coverage_unavailable_with_denominator(self):
        report = run_evaluation(self.data)
        for cond in SUPPRESSION_CONDITIONS:
            entry = report["suppression_coverage"][cond]
            self.assertIsNone(entry["count"])
            self.assertEqual(entry["denominator"], 4)
            self.assertEqual(entry["status"], "unavailable_no_masks_or_geometry")

    def test_no_writes_and_rollback_present(self):
        report = run_evaluation(self.data)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["rollback_command"], ROLLBACK_COMMAND)
        self.assertEqual(report["module_hash"], module_hash())

    def test_unavailable_metrics_declared(self):
        report = run_evaluation(self.data)
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, report["unavailable"])

    def test_report_is_json_serializable(self):
        report = run_evaluation(self.data)
        parsed = json.loads(report_to_json(report))
        self.assertEqual(parsed["n_records"], 4)

    def test_empty_input_is_safe(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 0)


if __name__ == "__main__":
    unittest.main()
