"""Offline tests for the read-only CMR cavity-context cluster rule gate."""

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_transition_check import CLUSTERS
from experiments.cmr_cavity_cluster_rule_gate import (
    ADDITIVE_EDIT_INVARIANTS,
    ROUTING_MODE,
    audit_diff,
    module_hash,
    report_to_json,
    rule_text,
    run_evaluation,
)


SYNTHETIC = {
    "records": [
        # adequate area, low cavity -> adequate_area_low_cavity
        {"case_id": "c1", "reference_area": 1000.0, "before_area": 900.0, "before_cavity": 0.2},
        # low area, adequate cavity -> low_area_adequate_cavity
        {"case_id": "c2", "reference_area": 1000.0, "before_area": 100.0, "before_cavity": 0.9},
        # low area, low cavity -> low_area_low_cavity
        {"case_id": "c3", "reference_area": 1000.0, "before_area": 100.0, "before_cavity": 0.2},
        # missing measurement -> unclassified
        {"case_id": "c4", "reference_area": None, "before_area": None, "before_cavity": None},
    ]
}


class RuleTextTests(unittest.TestCase):
    def test_three_clauses_present(self):
        text = rule_text()
        self.assertIn("R1_cluster_routing", text)
        self.assertIn("R2_additive_edit_invariants", text)
        self.assertIn("R3_escalation_gate", text)

    def test_r1_routes_all_clusters_to_shadow(self):
        text = rule_text()
        self.assertEqual(text["R1_cluster_routing"]["routing_mode"], ROUTING_MODE)
        self.assertEqual(set(text["R1_cluster_routing"]["clusters"]), set(CLUSTERS))

    def test_r2_invariants_declared(self):
        text = rule_text()
        self.assertEqual(
            set(text["R2_additive_edit_invariants"]["invariants"]),
            set(ADDITIVE_EDIT_INVARIANTS),
        )


class RunEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.report = run_evaluation(SYNTHETIC)

    def test_cluster_counts_with_denominators(self):
        counts = self.report["cluster_counts"]
        self.assertEqual(set(counts), set(CLUSTERS))
        for c in CLUSTERS:
            self.assertEqual(counts[c]["denominator"], 4)
        self.assertEqual(counts["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(counts["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(counts["low_area_low_cavity"]["count"], 1)

    def test_transitions_have_denominators(self):
        trans = self.report["simulated_transitions"]
        for c in CLUSTERS:
            self.assertEqual(trans[c]["denominator"], self.report["cluster_counts"][c]["count"])
            self.assertEqual(
                trans[c]["checked"] + trans[c]["trash"] + trans[c]["no_transition"],
                trans[c]["denominator"],
            )

    def test_routing_is_shadow_only(self):
        for case in self.report["cases"].values():
            if case["cluster"] is not None:
                self.assertEqual(case["routing"], ROUTING_MODE)

    def test_escalation_blocked(self):
        gate = self.report["escalation_gate"]
        self.assertTrue(gate["requires_per_cluster_transition_counts"])
        self.assertFalse(gate["escalation_allowed"])

    def test_suppression_coverage_denominators(self):
        cov = self.report["suppression_coverage"]
        self.assertTrue(cov)
        for cond in cov.values():
            self.assertEqual(cond["denominator"], 4)
            self.assertIsNone(cond["count"])
            self.assertEqual(cond["status"], "unavailable_no_masks_or_geometry")

    def test_unavailable_metrics_not_fabricated(self):
        unavail = self.report["unavailable"]
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, unavail)

    def test_zero_writes_and_rollback(self):
        self.assertEqual(self.report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertIn("git checkout", self.report["rollback_command"])
        self.assertEqual(self.report["audit_diff"]["production_writes"], 0)

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())
        self.assertEqual(self.report["module_hash"], module_hash())

    def test_report_is_json_serializable(self):
        json.loads(report_to_json(self.report))

    def test_audit_diff_detects_changes(self):
        prev = {"module_hash": "x", "cases": {"c1": {"cluster": "a"}}}
        cur = {"module_hash": "y", "cases": {"c1": {"cluster": "b"}, "c2": {}}}
        diff = audit_diff(prev, cur)
        self.assertEqual(diff["changed_cases"], ["c1"])
        self.assertEqual(diff["added_cases"], ["c2"])
        self.assertEqual(diff["production_writes"], 0)


if __name__ == "__main__":
    unittest.main()
