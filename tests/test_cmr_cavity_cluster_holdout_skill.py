"""Offline tests for the read-only CMR cavity-context cluster skill manifest."""

import json
import unittest

from experiments.cmr_cavity_cluster_holdout_skill import (
    ADDITIVE_EDIT_INVARIANTS,
    CLUSTERS,
    ROUTING_MODE,
    SUPPRESSION_CONDITIONS,
    audit_diff,
    module_hash,
    report_to_json,
    run_evaluation,
    skill_manifest,
)


SAMPLE = {
    "records": [
        # adequate area, low cavity
        {"case_id": "c1", "reference_area": 1000.0, "before_area": 800.0, "before_cavity": 0.2},
        # low area, adequate cavity
        {"case_id": "c2", "reference_area": 1000.0, "before_area": 300.0, "before_cavity": 0.9},
        # low area, low cavity
        {"case_id": "c3", "reference_area": 1000.0, "before_area": 200.0, "before_cavity": 0.1},
        # adequate area, adequate cavity -> outside the three clusters
        {"case_id": "c4", "reference_area": 1000.0, "before_area": 900.0, "before_cavity": 0.9},
        # missing measurement -> must not be imputed as zero
        {"case_id": "c5", "reference_area": 1000.0, "before_area": None, "before_cavity": 0.3},
    ]
}


class SkillManifestTests(unittest.TestCase):
    def test_manifest_declares_routing_and_invariants(self):
        m = skill_manifest()
        self.assertEqual(m["routing_mode"], ROUTING_MODE)
        self.assertEqual(tuple(m["clusters"]), CLUSTERS)
        self.assertEqual(tuple(m["additive_edit_invariants"]), ADDITIVE_EDIT_INVARIANTS)
        self.assertEqual(tuple(m["suppression_conditions"]), SUPPRESSION_CONDITIONS)
        self.assertIn("held-out per-cluster counts", m["escalation_precondition"])

    def test_cluster_counts_with_denominators(self):
        rep = run_evaluation(SAMPLE)
        self.assertEqual(rep["n_records"], 5)
        for c in CLUSTERS:
            self.assertEqual(rep["cluster_counts"][c]["denominator"], 5)
        self.assertEqual(rep["cluster_counts"]["adequate_area_low_cavity"]["count"], 1)
        self.assertEqual(rep["cluster_counts"]["low_area_adequate_cavity"]["count"], 1)
        self.assertEqual(rep["cluster_counts"]["low_area_low_cavity"]["count"], 1)

    def test_simulated_transitions_per_cluster(self):
        rep = run_evaluation(SAMPLE)
        t = rep["simulated_transitions"]
        self.assertEqual(t["adequate_area_low_cavity"]["checked"], 1)
        self.assertEqual(t["low_area_adequate_cavity"]["checked"], 1)
        self.assertEqual(t["low_area_low_cavity"]["trash"], 1)
        for c in CLUSTERS:
            self.assertEqual(t[c]["denominator"], rep["cluster_counts"][c]["count"])

    def test_missing_measurement_not_imputed(self):
        rep = run_evaluation(SAMPLE)
        self.assertIn("c5", rep["missing_measurement_cases"])
        self.assertIsNone(rep["cases"]["c5"]["cluster"])
        self.assertEqual(rep["cases"]["c5"]["routing"], "unrouted")

    def test_escalation_blocked_without_heldout_counts(self):
        rep = run_evaluation(SAMPLE)
        self.assertFalse(rep["escalation_gate"]["escalation_allowed"])
        self.assertTrue(rep["escalation_gate"]["requires_heldout_per_cluster_counts"])

    def test_invariants_and_suppression_unavailable(self):
        rep = run_evaluation(SAMPLE)
        for inv in ADDITIVE_EDIT_INVARIANTS:
            self.assertEqual(
                rep["additive_edit_invariants"][inv]["status"],
                "declared_unmeasured_no_masks_or_geometry",
            )
        for cond in SUPPRESSION_CONDITIONS:
            self.assertEqual(rep["suppression_coverage"][cond]["status"], "unavailable_no_masks_or_geometry")
        for key in ("concordance", "precision", "ci"):
            self.assertIn(key, rep["unavailable"])

    def test_zero_write_audit_and_rollback(self):
        rep = run_evaluation(SAMPLE)
        self.assertEqual(rep["writes"], {"labels": 0, "code": 0, "datasets": 0})
        self.assertEqual(rep["audit_diff"]["production_writes"], 0)
        self.assertIn("git checkout --", rep["rollback_command"])
        self.assertEqual(rep["module_hash"], module_hash())

    def test_audit_diff_detects_changes(self):
        prev = run_evaluation(SAMPLE)
        cur = run_evaluation(SAMPLE)
        cur["cases"]["c1"]["cluster"] = "low_area_low_cavity"
        diff = audit_diff(prev, cur)
        self.assertIn("c1", diff["changed_cases"])
        self.assertEqual(diff["production_writes"], 0)

    def test_report_is_json_serializable(self):
        rep = run_evaluation(SAMPLE)
        parsed = json.loads(report_to_json(rep))
        self.assertEqual(parsed["n_records"], 5)


if __name__ == "__main__":
    unittest.main()
