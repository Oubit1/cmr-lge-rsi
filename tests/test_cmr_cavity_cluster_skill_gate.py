"""Offline tests for the read-only CMR cavity-context cluster skill gate (S757)."""

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_transition_check import CLUSTERS
from experiments.cmr_cavity_cluster_skill_gate import (
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
        {"case_id": "c2", "reference_area": 1000.0, "before_area": 300.0, "before_cavity": 0.9},
        # low area, low cavity -> low_area_low_cavity
        {"case_id": "c3", "reference_area": 1000.0, "before_area": 300.0, "before_cavity": 0.2},
        # missing measurement -> unclassified, not imputed as zero
        {"case_id": "c4", "reference_area": None, "before_area": None, "before_cavity": None},
    ]
}


class SkillGateTest(unittest.TestCase):
    def test_rule_text_has_three_clauses(self):
        text = rule_text()
        self.assertEqual(set(text), {
            "R1_cluster_routing",
            "R2_additive_edit_invariants",
            "R3_escalation_gate",
        })
        self.assertEqual(text["R1_cluster_routing"]["routing_mode"], ROUTING_MODE)
        self.assertEqual(
            set(text["R2_additive_edit_invariants"]["invariants"]),
            set(ADDITIVE_EDIT_INVARIANTS),
        )

    def test_cluster_counts_and_denominators(self):
        report = run_evaluation(SYNTHETIC)
        self.assertEqual(report["n_records"], 4)
        for c in CLUSTERS:
            self.assertEqual(report["cluster_counts"][c]["denominator"], 4)
        self.assertEqual(report["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(report["cluster_counts"]["low_area_low_cavity"]["count"], 1)
        self.assertEqual(report["unclassified_cases"], ["c4"])
        self.assertEqual(report["missing_measurement_cases"], ["c4"])

    def test_routing_is_shadow_only(self):
        report = run_evaluation(SYNTHETIC)
        for case in report["cases"].values():
            if case["cluster"] is not None:
                self.assertEqual(case["routing"], ROUTING_MODE)
            else:
                self.assertEqual(case["routing"], "unrouted")

    def test_transition_denominators_match_cluster_counts(self):
        report = run_evaluation(SYNTHETIC)
        for c in CLUSTERS:
            t = report["simulated_transitions"][c]
            self.assertEqual(t["denominator"], report["cluster_counts"][c]["count"])
            self.assertEqual(
                t["checked"] + t["trash"] + t["no_transition"],
                t["denominator"],
            )

    def test_escalation_blocked_without_evidence(self):
        report = run_evaluation(SYNTHETIC)
        gate = report["escalation_gate"]
        self.assertTrue(gate["requires_per_cluster_transition_counts"])
        self.assertFalse(gate["escalation_allowed"])

    def test_additive_edit_invariants_declared_unmeasured(self):
        report = run_evaluation(SYNTHETIC)
        for inv in ADDITIVE_EDIT_INVARIANTS:
            self.assertEqual(
                report["additive_edit_invariants"][inv]["status"],
                "declared_unmeasured_no_masks_or_geometry",
            )

    def test_suppression_coverage_unavailable_with_denominator(self):
        report = run_evaluation(SYNTHETIC)
        for cond, entry in report["suppression_coverage"].items():
            self.assertIsNone(entry["count"])
            self.assertEqual(entry["denominator"], 4)
            self.assertEqual(entry["status"], "unavailable_no_masks_or_geometry")

    def test_unavailable_metrics_not_fabricated(self):
        report = run_evaluation(SYNTHETIC)
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, report["unavailable"])

    def test_zero_writes_and_rollback(self):
        report = run_evaluation(SYNTHETIC)
        self.assertEqual(report["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(report["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout", report["rollback_command"])

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())
        self.assertEqual(len(module_hash()), 64)

    def test_audit_diff_detects_changes(self):
        prev = {"module_hash": "abc", "cases": {"c1": {"cluster": "x"}}}
        cur = {"module_hash": "def", "cases": {"c1": {"cluster": "y"}, "c2": {}}}
        diff = audit_diff(prev, cur)
        self.assertEqual(diff["changed_cases"], ["c1"])
        self.assertEqual(diff["added_cases"], ["c2"])
        self.assertEqual(diff["production_writes"], 0)

    def test_report_is_json_serializable(self):
        report = run_evaluation(SYNTHETIC)
        parsed = json.loads(report_to_json(report))
        self.assertEqual(parsed["n_records"], 4)

    def test_empty_input(self):
        report = run_evaluation({"records": []})
        self.assertEqual(report["n_records"], 0)
        self.assertFalse(report["escalation_gate"]["escalation_allowed"])


if __name__ == "__main__":
    unittest.main()
