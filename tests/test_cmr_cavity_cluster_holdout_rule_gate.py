"""Offline tests for the read-only cavity-context cluster rule gate."""

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_rule_gate import (
    ADDITIVE_EDIT_INVARIANTS,
    CLUSTERS,
    ROUTING_MODE,
    SUPPRESSION_CONDITIONS,
    audit_diff,
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


class RuleGateTest(unittest.TestCase):
    def setUp(self):
        self.data = {
            "records": [
                _rec("c_adequate_low", 1000.0, 800.0, 0.2),   # adequate_area_low_cavity
                _rec("c_low_adequate", 1000.0, 300.0, 0.8),   # low_area_adequate_cavity
                _rec("c_low_low", 1000.0, 200.0, 0.2),        # low_area_low_cavity
                _rec("c_outside", 1000.0, 800.0, 0.8),        # outside three clusters
                _rec("c_missing", 1000.0, None, 0.2),         # missing measurement
            ]
        }

    def test_cluster_counts_and_denominators(self):
        rep = run_evaluation(self.data)
        self.assertEqual(rep["n_records"], 5)
        for c in CLUSTERS:
            self.assertEqual(rep["cluster_counts"][c]["denominator"], 5)
        self.assertEqual(rep["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(rep["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(rep["cluster_counts"]["low_area_low_cavity"]["count"], 1)

    def test_missing_measurement_not_imputed(self):
        rep = run_evaluation(self.data)
        self.assertIn("c_missing", rep["missing_measurement_cases"])
        self.assertIn("c_outside", rep["unclassified_cases"])
        self.assertIsNone(rep["cases"]["c_missing"]["cluster"])

    def test_simulated_transitions(self):
        rep = run_evaluation(self.data)
        self.assertEqual(rep["simulated_transitions"]["low_area_low_cavity"]["trash"], 1)
        self.assertEqual(rep["simulated_transitions"]["adequate_area_low_cavity"]["checked"], 1)
        self.assertEqual(rep["simulated_transitions"]["low_area_adequate_cavity"]["checked"], 1)

    def test_shadow_triage_only_and_escalation_blocked(self):
        rep = run_evaluation(self.data)
        self.assertEqual(rep["routing_mode"], ROUTING_MODE)
        self.assertFalse(rep["escalation_gate"]["escalation_allowed"])
        self.assertTrue(rep["escalation_gate"]["requires_heldout_per_cluster_counts"])

    def test_additive_edit_invariants_declared_unmeasured(self):
        rep = run_evaluation(self.data)
        for inv in ADDITIVE_EDIT_INVARIANTS:
            self.assertEqual(
                rep["additive_edit_invariants"][inv]["status"],
                "declared_unmeasured_no_masks_or_geometry",
            )

    def test_suppression_coverage_unavailable(self):
        rep = run_evaluation(self.data)
        for cond in SUPPRESSION_CONDITIONS:
            self.assertIsNone(rep["suppression_coverage"][cond]["count"])
            self.assertEqual(
                rep["suppression_coverage"][cond]["status"],
                "unavailable_no_masks_or_geometry",
            )

    def test_zero_writes_and_audit(self):
        rep = run_evaluation(self.data)
        self.assertEqual(rep["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(rep["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", rep["rollback_command"])

    def test_module_hash_stable_and_json_serializable(self):
        self.assertEqual(module_hash(), module_hash())
        rep = run_evaluation(self.data)
        self.assertIsInstance(report_to_json(rep), str)
        json.loads(report_to_json(rep))

    def test_audit_diff_with_previous(self):
        rep = run_evaluation(self.data)
        diff = audit_diff(rep, rep)
        self.assertEqual(diff["changed_cases"], [])
        self.assertEqual(diff["production_writes"], 0)

    def test_empty_input(self):
        rep = run_evaluation({"records": []})
        self.assertEqual(rep["n_records"], 0)
        for c in CLUSTERS:
            self.assertEqual(rep["cluster_counts"][c]["count"], 0)


if __name__ == "__main__":
    unittest.main()
