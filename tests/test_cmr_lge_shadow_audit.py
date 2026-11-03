"""
Unit tests for CMR/LGE Shadow Audit Orchestrator.
Validates offline execution, metric correctness, and shadow-mode isolation.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.cmr_lge_shadow_audit import (
    CachedDigestCache,
    CMRLGEShadowAuditOrchestrator,
    arm_noop,
    arm_lge_largest_retention,
    arm_cmr_paired_merge,
    arm_cross_slice_propagation,
    compute_metrics
)

class TestCachedDigestCache(unittest.TestCase):
    def setUp(self):
        self.cache = CachedDigestCache()

    def test_load_cohort_counts(self):
        cases = self.cache.load_cohort(n_flagged=20, n_controls=5)
        self.assertEqual(len(cases), 25)

        flagged = [c for c in cases if "mi_multiple_components" in c.flags]
        controls = [c for c in cases if len(c.flags) == 0]

        self.assertEqual(len(flagged), 20)
        self.assertEqual(len(controls), 5)

    def test_flagged_structure(self):
        cases = self.cache.load_cohort(n_flagged=1, n_controls=0)
        case = cases[0]
        self.assertIn("myo_topology", case.flags)
        self.assertEqual(len(case.slices), 3)
        for s in case.slices:
            self.assertIsNotNone(s.cmr_ring)
            self.assertTrue(s.cmr_ring.is_contiguous)
            self.assertGreater(len(s.lge_components), 1)

    def test_control_structure(self):
        cases = self.cache.load_cohort(n_flagged=0, n_controls=1)
        case = cases[0]
        self.assertEqual(len(case.flags), 0)
        for s in case.slices:
            self.assertIsNone(s.cmr_ring)


class TestProcessingArms(unittest.TestCase):
    def setUp(self):
        self.cache = CachedDigestCache()
        self.cases = self.cache.load_cohort(n_flagged=2, n_controls=1)

    def test_arm_noop_identity(self):
        result = arm_noop(self.cases[0])
        self.assertEqual(result.case_id, self.cases[0].case_id)
        self.assertEqual(len(result.slices), len(self.cases[0].slices))
        for r_s, o_s in zip(result.slices, self.cases[0].slices):
            self.assertEqual(len(r_s.lge_components), len(o_s.lge_components))

    def test_arm_lge_largest_retention(self):
        result = arm_lge_largest_retention(self.cases[0])
        for s in result.slices:
            if s.lge_components:
                self.assertEqual(len(s.lge_components), 1)

    def test_arm_cmr_paired_merge(self):
        result = arm_cmr_paired_merge(self.cases[0], voxel_window=15.0)
        # Flagged cases have components near origin, ring at radius 10
        # Components at 0,0 should be within window
        # Should reduce component count in some slices
        for s in result.slices:
            # At least one component should remain
            self.assertGreaterEqual(len(s.lge_components), 1)

    def test_arm_cross_slice_propagation(self):
        result = arm_cross_slice_propagation(self.cases[0], voxel_window=15.0)
        # Should not crash and should produce valid structure
        self.assertEqual(len(result.slices), 3)


class TestMetricsComputation(unittest.TestCase):
    def setUp(self):
        self.cache = CachedDigestCache()
        self.cases = self.cache.load_cohort(n_flagged=20, n_controls=5)

    def test_metrics_structure(self):
        metrics = compute_metrics(self.cases, "test_arm")
        required_keys = [
            "mi_multiple_components_prevalence",
            "mean_lge_mi_component_count",
            "mi_fraction_high_prevalence",
            "myo_topology_cooccurrence",
            "cmr_lge_ring_break_agreement_rate",
            "candidate_merge_drop_slice_count",
            "adjudication_discordance_rate",
            "false_repair_rate"
        ]
        for key in required_keys:
            self.assertIn(key, metrics)

    def test_flagged_prevalence(self):
        metrics = compute_metrics(self.cases, "test_arm")
        # 20 out of 25 cases are flagged
        self.assertAlmostEqual(metrics["mi_multiple_components_prevalence"], 0.8, places=1)

    def test_myco_cooccurrence(self):
        metrics = compute_metrics(self.cases, "test_arm")
        # All 20 flagged cases have myo_topology
        self.assertEqual(metrics["myo_topology_cooccurrence"], 20)

    def test_false_repair_rate_controls(self):
        metrics = compute_metrics(self.cases, "test_arm")
        # Controls should not increase false repair rate
        self.assertLessEqual(metrics["false_repair_rate"], 0.1) # Loose bound for mock


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.cache = CachedDigestCache()
        self.orchestrator = CMRLGEShadowAuditOrchestrator(self.cache)

    def test_run_audit_all_arms(self):
        results = self.orchestrator.run_audit(n_flagged=20, n_controls=5)
        self.assertIn("arm_1_noop", results)
        self.assertIn("arm_2_lge_largest", results)
        self.assertIn("arm_3_cmr_paired_merge", results)
        self.assertIn("arm_4_cross_slice", results)

    def test_shadow_mode_isolation(self):
        # Ensure no global state changes
        initial_cache_state = self.cache._cache.copy()
        self.orchestrator.run_audit(n_flagged=1, n_controls=1)
        self.assertEqual(self.cache._cache, initial_cache_state)

    def test_ready_for_validation_criteria(self):
        results = self.orchestrator.run_audit(n_flagged=20, n_controls=5)
        # Check that all arms report required metrics
        for arm_name, metrics in results.items():
            self.assertGreater(metrics["mi_multiple_components_prevalence"], 0)
            self.assertGreater(metrics["mean_lge_mi_component_count"], 0)

if __name__ == '__main__':
    unittest.main()
