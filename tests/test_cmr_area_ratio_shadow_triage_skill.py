"""Offline tests for the S760 read-only area-ratio shadow-triage skill draft."""

import json
import unittest

from experiments.cmr_area_ratio_shadow_triage_skill import (
    COHORT_LABEL,
    MODULE_VERSION,
    ROUTING_MODE,
    TOPOLOGY_INVARIANTS,
    module_hash,
    report_to_json,
    run_evaluation,
)


class TestShadowTriageSkill(unittest.TestCase):
    def _records(self):
        return [
            {
                "case_id": "c_small",
                "reference_area": 1000.0,
                "before_area": 100.0,
                "before_cavity": 0.2,
            },
            {
                "case_id": "c_large",
                "reference_area": 1000.0,
                "before_area": 900.0,
                "before_cavity": 0.2,
            },
            {
                "case_id": "c_missing",
                "reference_area": 1000.0,
                "before_area": None,
                "before_cavity": None,
            },
        ]

    def test_read_only_and_version(self):
        report = run_evaluation({"records": self._records()})
        self.assertEqual(report["module_version"], MODULE_VERSION)
        self.assertEqual(report["routing_mode"], ROUTING_MODE)
        self.assertEqual(report["cohort"], COHORT_LABEL)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertEqual(report["module_hash"], module_hash())

    def test_missing_measurements_not_imputed(self):
        report = run_evaluation({"records": self._records()})
        self.assertIn("c_missing", report["missing_measurement_cases"])
        self.assertIsNone(report["cases"]["c_missing"]["area_ratio"])
        self.assertIsNone(report["cases"]["c_missing"]["myo_cavity_area_ratio"])

    def test_cluster_counts_have_denominators(self):
        report = run_evaluation({"records": self._records()})
        for cluster, info in report["cluster_counts"].items():
            self.assertEqual(info["denominator"], 3)
            self.assertGreaterEqual(info["count"], 0)

    def test_transitions_have_denominators(self):
        report = run_evaluation({"records": self._records()})
        for cluster, info in report["simulated_transitions"].items():
            self.assertEqual(info["denominator"], report["cluster_counts"][cluster]["count"])
            self.assertEqual(info["cohort"], COHORT_LABEL)

    def test_ledger_deltas_unavailable_without_ledger(self):
        report = run_evaluation({"records": self._records()})
        self.assertEqual(
            report["ledger_deltas"]["status"],
            "unavailable_no_ledger_supplied",
        )

    def test_ledger_deltas_computed_when_supplied(self):
        ledger = {c: {"checked": 0, "trash": 0} for c in report_clusters()}
        report = run_evaluation({"records": self._records(), "ledger": ledger})
        self.assertEqual(report["ledger_deltas"]["status"], "ok")
        for c in report_clusters():
            self.assertIn("checked_delta", report["ledger_deltas"][c])

    def test_suppression_unavailable_without_masks(self):
        report = run_evaluation({"records": self._records()})
        for cond, info in report["suppression_coverage"].items():
            self.assertIsNone(info["count"])
            self.assertEqual(info["status"], "unavailable_no_masks_or_geometry")

    def test_suppression_counted_when_supplied(self):
        recs = self._records()
        recs[0]["border_touch"] = True
        report = run_evaluation({"records": recs})
        self.assertEqual(report["suppression_coverage"]["border_touch"]["count"], 1)
        self.assertEqual(report["cases"]["c_small"]["routing"], "stack_level_review")

    def test_topology_invariants_unavailable_without_masks(self):
        report = run_evaluation({"records": self._records()})
        for name in TOPOLOGY_INVARIANTS:
            info = report["topology_invariants"][name]
            self.assertEqual(info["measured"], 0)
            self.assertEqual(info["status"], "unavailable_no_masks_or_geometry")

    def test_topology_invariants_measured_when_supplied(self):
        recs = self._records()
        recs[0]["connected_components_before"] = 1
        recs[0]["connected_components_after"] = 1
        recs[0]["holes_before"] = 0
        recs[0]["holes_after"] = 1
        recs[0]["border_contact_before"] = False
        recs[0]["border_contact_after"] = False
        report = run_evaluation({"records": recs})
        self.assertEqual(report["topology_invariants"]["connected_components"]["preserved"], 1)
        self.assertEqual(report["topology_invariants"]["holes"]["violated"], 1)
        self.assertIn("c_small", report["topology_violation_cases"])

    def test_concordance_pseudo_holdout_labeled(self):
        statuses = {"c_small": "trash", "c_large": "checked"}
        report = run_evaluation({"records": self._records(), "statuses": statuses})
        self.assertEqual(report["concordance"]["cohort"], COHORT_LABEL)
        self.assertIn("pseudo-holdout", report["concordance"]["note"])

    def test_unavailable_metrics_declared(self):
        report = run_evaluation({"records": self._records()})
        for key in (
            "suppression_coverage",
            "topology_invariants",
            "concordance_vs_independent_labels",
            "precision",
            "ci",
        ):
            self.assertIn(key, report["unavailable"])

    def test_report_is_json_serializable(self):
        report = run_evaluation({"records": self._records()})
        text = report_to_json(report)
        self.assertIsInstance(json.loads(text), dict)

    def test_real_supplied_measurements(self):
        data = _real_measurements()
        if data is None:
            self.skipTest("real measurements not embedded")
        report = run_evaluation(data)
        self.assertEqual(report["n_records"], len(data["records"]))
        self.assertEqual(report["writes"]["labels"], 0)
        for cluster, info in report["cluster_counts"].items():
            self.assertEqual(info["denominator"], len(data["records"]))


def report_clusters():
    from experiments.cmr_cavity_cluster_holdout_transition_check import CLUSTERS
    return CLUSTERS


def _real_measurements():
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "s760_measurements.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    unittest.main()
