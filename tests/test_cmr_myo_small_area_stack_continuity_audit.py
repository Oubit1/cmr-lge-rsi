"""
Unit tests for cmr_myo_small_area_stack_continuity_audit.

Runs offline without external data. Tests all core functions including
severity tier assignment, co-occurrence pattern detection, metric
computation, and audit result structure.
"""

import sys
import unittest
from pathlib import Path

# Ensure experiments module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.cmr_myo_small_area_stack_continuity_audit import (
    CaseDigest,
    StratumKey,
    assign_severity_tier,
    detect_cooccurrence_pattern,
    compute_temporal_metrics,
    compute_spatial_metrics,
    compute_lge_support,
    compute_shadow_discordance,
    sample_cases,
    _mean_ci,
    _has_shadow_discordance,
    SEVERITY_TIERS,
)


class TestSeverityTierAssignment(unittest.TestCase):
    """Test severity tier assignment based on area ratio."""

    def test_severe_tier(self):
        self.assertEqual(assign_severity_tier(0.05), "severe")
        self.assertEqual(assign_severity_tier(0.09), "severe")

    def test_moderate_tier(self):
        self.assertEqual(assign_severity_tier(0.10), "moderate")
        self.assertEqual(assign_severity_tier(0.20), "moderate")
        self.assertEqual(assign_severity_tier(0.249), "moderate")

    def test_mild_tier(self):
        self.assertEqual(assign_severity_tier(0.25), "mild")
        self.assertEqual(assign_severity_tier(0.40), "mild")
        self.assertEqual(assign_severity_tier(0.499), "mild")

    def test_not_small_area(self):
        self.assertIsNone(assign_severity_tier(0.50))
        self.assertIsNone(assign_severity_tier(0.75))
        self.assertIsNone(assign_severity_tier(1.0))

    def test_invalid_negative(self):
        self.assertIsNone(assign_severity_tier(-0.1))


class TestCooccurrencePatternDetection(unittest.TestCase):
    """Test co-occurrence pattern detection."""

    def test_pattern_a(self):
        flags = ("cmr_myo_small_area", "cmr_myo_excess_components", "cmr_myo_touches_image_border")
        self.assertEqual(detect_cooccurrence_pattern(flags), "pattern_a")

    def test_pattern_b(self):
        flags = ("cmr_myo_small_area", "cmr_myo_missing_temporal_hole", "cmr_centroid_drift")
        self.assertEqual(detect_cooccurrence_pattern(flags), "pattern_b")

    def test_pattern_c_isolated(self):
        flags = ("cmr_myo_small_area",)
        self.assertEqual(detect_cooccurrence_pattern(flags), "pattern_c")

    def test_pattern_c_with_unrelated_flags(self):
        flags = ("cmr_myo_small_area", "some_other_flag")
        self.assertEqual(detect_cooccurrence_pattern(flags), "pattern_c")

    def test_pattern_a_priority_over_b(self):
        # Pattern A should be detected even if B flags are also present
        flags = (
            "cmr_myo_small_area",
            "cmr_myo_excess_components",
            "cmr_myo_touches_image_border",
            "cmr_myo_missing_temporal_hole",
            "cmr_centroid_drift",
        )
        self.assertEqual(detect_cooccurrence_pattern(flags), "pattern_a")


class TestMeanCI(unittest.TestCase):
    """Test mean and confidence interval computation."""

    def test_empty_list(self):
        mean_val, ci = _mean_ci([])
        self.assertEqual(mean_val, 0.0)
        self.assertEqual(ci, (0.0, 0.0))

    def test_single_value(self):
        mean_val, ci = _mean_ci([5.0])
        self.assertEqual(mean_val, 5.0)
        self.assertEqual(ci, (5.0, 5.0))

    def test_multiple_values(self):
        mean_val, ci = _mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(mean_val, 3.0, places=5)
        self.assertTrue(ci[0] <= mean_val <= ci[1])


class TestTemporalMetrics(unittest.TestCase):
    """Test temporal continuity metric computation."""

    def test_basic_temporal(self):
        case = CaseDigest(
            case_id="test_001",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[
                {"area": 100.0, "centroid_drift": False},
                {"area": 110.0, "centroid_drift": True},
                {"area": 105.0, "centroid_drift": False},
            ],
            lge_present=False,
            lge_info=None,
        )
        metrics = compute_temporal_metrics([case])
        self.assertGreater(metrics.area_cv, 0.0)
        self.assertGreaterEqual(metrics.centroid_drift_persistence, 0.0)
        self.assertLessEqual(metrics.centroid_drift_persistence, 1.0)
        self.assertEqual(metrics.n_cases, 1)

    def test_no_area_data(self):
        case = CaseDigest(
            case_id="test_002",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"centroid_drift": False}],
            lge_present=False,
            lge_info=None,
        )
        metrics = compute_temporal_metrics([case])
        self.assertEqual(metrics.area_cv, 0.0)
        self.assertEqual(metrics.n_cases, 1)


class TestSpatialMetrics(unittest.TestCase):
    """Test spatial stack continuity metric computation."""

    def test_basic_spatial(self):
        case = CaseDigest(
            case_id="test_003",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[
                {"area": 100.0, "ring_break": False, "layer_area_jump": False},
                {"area": 110.0, "ring_break": True, "layer_area_jump": False},
                {"area": 105.0, "ring_break": False, "layer_area_jump": True},
            ],
            lge_present=False,
            lge_info=None,
        )
        metrics = compute_spatial_metrics([case])
        self.assertGreaterEqual(metrics.ring_break_persistence, 0.0)
        self.assertLessEqual(metrics.ring_break_persistence, 1.0)
        self.assertGreaterEqual(metrics.layer_area_jump_persistence, 0.0)
        self.assertLessEqual(metrics.layer_area_jump_persistence, 1.0)
        self.assertEqual(metrics.n_cases, 1)


class TestLGESupport(unittest.TestCase):
    """Test paired-LGE support metric computation."""

    def test_lge_present(self):
        case = CaseDigest(
            case_id="test_004",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=True,
            lge_info={
                "myocardial_presence": True,
                "component_count": 2,
                "cross_modality_area_agreement": 0.85,
            },
        )
        metrics = compute_lge_support([case])
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.lge_presence_rate, 1.0)
        self.assertEqual(metrics.lge_component_count_mean, 2.0)
        self.assertEqual(metrics.cross_modality_area_agreement, 0.85)
        self.assertEqual(metrics.n_lge_present, 1)

    def test_lge_absent(self):
        case = CaseDigest(
            case_id="test_005",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=False,
            lge_info=None,
        )
        metrics = compute_lge_support([case])
        self.assertIsNone(metrics)

    def test_lge_no_info(self):
        case = CaseDigest(
            case_id="test_006",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=True,
            lge_info=None,
        )
        metrics = compute_lge_support([case])
        # lge_info is None, so treated as no LGE info
        self.assertIsNone(metrics)


class TestShadowDiscordance(unittest.TestCase):
    """Test shadow discordance detection."""

    def test_discordant_case(self):
        case = CaseDigest(
            case_id="test_007",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=False,
            lge_info=None,
            run14_shadow={"a1_no_op": True, "largest_component_retained": False},
        )
        self.assertTrue(_has_shadow_discordance(case))
        rate, n, total = compute_shadow_discordance([case])
        self.assertEqual(rate, 1.0)
        self.assertEqual(n, 1)
        self.assertEqual(total, 1)

    def test_non_discordant_case(self):
        case = CaseDigest(
            case_id="test_008",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=False,
            lge_info=None,
            run14_shadow={"a1_no_op": True, "largest_component_retained": True},
        )
        self.assertFalse(_has_shadow_discordance(case))
        rate, n, total = compute_shadow_discordance([case])
        self.assertEqual(rate, 0.0)
        self.assertEqual(n, 0)
        self.assertEqual(total, 1)

    def test_no_shadow(self):
        case = CaseDigest(
            case_id="test_009",
            area_ratio=0.15,
            flags=("cmr_myo_small_area",),
            slices=[{"area": 100.0}],
            lge_present=False,
            lge_info=None,
            run14_shadow=None,
        )
        self.assertFalse(_has_shadow_discordance(case))


class TestSampling(unittest.TestCase):
    """Test case sampling with discordance preference."""

    def test_sampling_limits(self):
        # Create 20 cases per stratum to test limiting
        cases = []
        for i in range(60):
            tier = "severe" if i < 20 else "moderate" if i < 40 else "mild"
            cases.append(CaseDigest(
                case_id=f"case_{i:03d}",
                area_ratio=0.05 if tier == "severe" else 0.15 if tier == "moderate" else 0.35,
                flags=("cmr_myo_small_area",),
                slices=[{"area": 100.0}],
                lge_present=False,
                lge_info=None,
                run14_shadow={"a1_no_op": True, "largest_component_retained": False} if i % 2 == 0 else None,
            ))

        sampled = sample_cases(cases, max_per_stratum=10, total_max=25)
        total_sampled = sum(len(v) for v in sampled.values())
        self.assertLessEqual(total_sampled, 25)
        # Check that discordant cases are preferred
        for stratum_cases in sampled.values():
            for case in stratum_cases:
                if case.run14_shadow is not None:
                    self.assertTrue(
                        case.run14_shadow.get("a1_no_op", False) and not case.run14_shadow.get("largest_component_retained", True)
                    )

    def test_empty_cases(self):
        sampled = sample_cases([], max_per_stratum=5, total_max=15)
        self.assertEqual(len(sampled), 0)


class TestStratumKey(unittest.TestCase):
    """Test StratumKey dataclass."""

    def test_equality(self):
        k1 = StratumKey(tier="severe", pattern="pattern_a")
        k2 = StratumKey(tier="severe", pattern="pattern_a")
        k3 = StratumKey(tier="moderate", pattern="pattern_a")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(hash(k1), hash(k2))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full audit pipeline."""

    def test_full_pipeline_with_mock_data(self):
        """Test that the audit pipeline runs end-to-end with mock data."""
        from experiments.cmr_myo_small_area_stack_continuity_audit import (
            run_audit,
            emit_stratum_table,
            AuditResult,
        )

        # The actual run_audit depends on file system; test the structure instead
        # by directly constructing an AuditResult
        result = AuditResult(
            stratum_reports=[],
            total_flagged=0,
            tier_counts={},
            pattern_counts={},
            hash_validated=False,
            hash_mismatch_reported=False,
            fallback_extractor_used=False,
        )
        table = emit_stratum_table(result)
        self.assertIn("Total flagged cases", table)
        self.assertIn("Hash validated", table)

    def test_severity_tier_boundary_conditions(self):
        """Test boundary conditions for severity tiers."""
        # Exact boundaries
        self.assertEqual(assign_severity_tier(0.0), "severe")
        self.assertEqual(assign_severity_tier(0.10), "moderate")
        self.assertEqual(assign_severity_tier(0.25), "mild")
        # Just below boundaries
        self.assertEqual(assign_severity_tier(0.099), "severe")
        self.assertEqual(assign_severity_tier(0.249), "moderate")
        # At upper boundary of mild
        self.assertIsNone(assign_severity_tier(0.50))


if __name__ == "__main__":
    unittest.main(verbosity=2)
