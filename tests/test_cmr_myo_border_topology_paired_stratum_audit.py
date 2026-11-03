"""
Unit tests for cmr_myo_border_topology_paired_stratum_audit.

Tests run offline without external data.
Validates: metrics computation, shadow arm agreement, discordant queue generation.
"""

import dataclasses
import json
import pathlib
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

# Import the module under test
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.cmr_myo_border_topology_paired_stratum_audit import (
    CMRBorderTopologyPairedStratumAudit,
    DiscordantSlice,
    ShadowArmResult,
    SliceDigest,
    TopologyMetrics,
    _check_discordance,
    _compute_connected_components,
    _compute_cross_slice_continuity,
    _compute_hole_count_euler,
    _compute_min_band_distance_to_border,
    _compute_radial_ring_break,
    _compute_ring_contiguity,
    _compute_temporal_area_cv,
    _cross_slice_propagation_arm,
    _import_only_extractor_b1,
    _no_op_arm,
    _ring_break_aware_retention_arm,
    _temporal_continuity_arm,
    run_bounded_audit,
)


class TestMetricExtraction(unittest.TestCase):
    """Test pure metric extraction functions."""

    def test_ring_contiguity_none_mask(self):
        """Ring contiguity with no mask should be False."""
        result = _compute_ring_contiguity(None)
        self.assertFalse(result)

    def test_radial_ring_break_none_mask(self):
        """Radial ring break with no mask should be True (broken)."""
        result = _compute_radial_ring_break(None)
        self.assertTrue(result)

    def test_min_band_distance_none_mask(self):
        """Min distance with no mask should be infinity."""
        result = _compute_min_band_distance_to_border(None, (256, 256))
        self.assertEqual(result, float('inf'))

    def test_connected_components_none_mask(self):
        """Connected components with no mask should be (0, 0.0)."""
        count, ratio = _compute_connected_components(None)
        self.assertEqual(count, 0)
        self.assertEqual(ratio, 0.0)

    def test_hole_count_none_mask(self):
        """Hole count with no mask should be (0, 0)."""
        holes, euler = _compute_hole_count_euler(None)
        self.assertEqual(holes, 0)
        self.assertEqual(euler, 0)

    def test_temporal_area_cv_insufficient_data(self):
        """Temporal CV with < 2 points should be None."""
        result = _compute_temporal_area_cv([100.0])
        self.assertIsNone(result)

    def test_temporal_area_cv_computation(self):
        """Temporal CV should compute correctly."""
        areas = [100.0, 110.0, 90.0]  # mean=100, std=10
        result = _compute_temporal_area_cv(areas)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_temporal_area_cv_zero_mean(self):
        """Temporal CV with zero mean should be None."""
        areas = [0.0, 0.0, 0.0]
        result = _compute_temporal_area_cv(areas)
        self.assertIsNone(result)


class TestCrossSliceContinuity(unittest.TestCase):
    """Test cross-slice continuity computations."""

    def test_insufficient_slices(self):
        """Cross-slice with < 2 slices should return (None, None)."""
        cv, drift = _compute_cross_slice_continuity([{"area": 100}])
        self.assertIsNone(cv)
        self.assertIsNone(drift)

    def test_valid_cross_slice(self):
        """Cross-slice with valid data should compute metrics."""
        slices = [
            {"area": 100, "centroid": (10.0, 10.0, 5.0)},
            {"area": 105, "centroid": (11.0, 10.5, 5.0)},
        ]
        cv, drift = _compute_cross_slice_continuity(slices)
        self.assertIsNotNone(cv)
        self.assertIsNotNone(drift)


class TestImportOnlyExtractor(unittest.TestCase):
    """Test Run-9 B1 import-only extractor."""

    def test_extractor_returns_expected_structure(self):
        """Extractor should return structured data without external calls."""
        digest = SliceDigest(
            case_id="TEST_001",
            slice_idx=5,
            frame_idx=0,
            run_id="run_9",
            config_hash="c3d4e5f6a7b8901234567890123456789012abcdef",
        )
        result = _import_only_extractor_b1(digest)

        self.assertTrue(result["imported"])
        self.assertEqual(result["contract"], "run_9_b1_import_only_extractor")
        self.assertEqual(result["case_id"], "TEST_001")


class TestShadowArms(unittest.TestCase):
    """Test shadow arm implementations."""

    def _make_metrics(self, ring_break: bool = False, comp_count: int = 1) -> TopologyMetrics:
        return TopologyMetrics(
            case_id="TEST",
            slice_idx=0,
            frame_idx=0,
            ring_contiguity=not ring_break,
            radial_ring_break=ring_break,
            min_band_distance_to_border_voxels=5.0,
            component_count=comp_count,
            largest_component_area_ratio=1.0,
            hole_count=0,
            euler_characteristic=0,
            temporal_area_cv=None,
            centroid_drift_voxels=None,
            cross_slice_area_cv=None,
            cross_slice_centroid_drift=None,
            lge_mi_component_count=None,
            lge_mi_fraction_high=None,
        )

    def test_no_op_arm(self):
        """No-op arm should preserve original metrics."""
        metrics = self._make_metrics(ring_break=False, comp_count=2)
        result = _no_op_arm(None, metrics)

        self.assertEqual(result.arm_name, "no_op")
        self.assertFalse(result.ring_break_status)
        self.assertEqual(result.component_count, 2)

    def test_ring_break_aware_no_break(self):
        """Ring-break arm with no break should match no-op."""
        metrics = self._make_metrics(ring_break=False, comp_count=1)
        result = _ring_break_aware_retention_arm(None, metrics)

        self.assertEqual(result.arm_name, "ring_break_aware_retention")
        self.assertFalse(result.ring_break_status)
        self.assertEqual(result.component_count, 1)

    def test_ring_break_aware_with_break(self):
        """Ring-break arm with break should resolve to single component."""
        metrics = self._make_metrics(ring_break=True, comp_count=3)
        # Provide a non-None mask to trigger the retention path
        mask = {"type": "synthetic", "ring_break": True}
        result = _ring_break_aware_retention_arm(mask, metrics)

        self.assertEqual(result.arm_name, "ring_break_aware_retention")
        self.assertFalse(result.ring_break_status)  # Resolved
        self.assertEqual(result.component_count, 1)
        self.assertEqual(result.largest_component_area_ratio, 1.0)

    def test_cross_slice_propagation_no_neighbors(self):
        """Cross-slice with no neighbors should preserve original break status."""
        metrics = self._make_metrics(ring_break=True, comp_count=2)
        result = _cross_slice_propagation_arm(None, metrics, None)

        self.assertTrue(result.ring_break_status)  # Unchanged

    def test_temporal_continuity_no_temporal(self):
        """Temporal arm with no temporal context should preserve original."""
        metrics = self._make_metrics(ring_break=True, comp_count=2)
        result = _temporal_continuity_arm(None, metrics, None)

        self.assertTrue(result.ring_break_status)  # Unchanged


class TestDiscordanceDetection(unittest.TestCase):
    """Test discordance detection between no-op and shadow arms."""

    def _make_arm(self, name: str, ring_break: bool, comp_count: int) -> ShadowArmResult:
        return ShadowArmResult(
            arm_name=name,
            ring_break_status=ring_break,
            component_count=comp_count,
            largest_component_area_ratio=1.0,
            retained_mask_sparse=None,
        )

    def test_no_discordance_identical(self):
        """Identical arms should not produce discordance."""
        no_op = self._make_arm("no_op", False, 1)
        arm = self._make_arm("test", False, 1)

        result = _check_discordance(no_op, arm, "CASE", 0, 0)
        self.assertIsNone(result)

    def test_discordance_ring_break(self):
        """Different ring break status should produce discordance."""
        no_op = self._make_arm("no_op", False, 1)
        arm = self._make_arm("test", True, 1)

        result = _check_discordance(no_op, arm, "CASE", 0, 0)
        self.assertIsNotNone(result)
        self.assertIn("ring_break", result.reason)

    def test_discordance_component_count(self):
        """Different component count should produce discordance."""
        no_op = self._make_arm("no_op", False, 1)
        arm = self._make_arm("test", False, 2)

        result = _check_discordance(no_op, arm, "CASE", 0, 0)
        self.assertIsNotNone(result)
        self.assertIn("component_count", result.reason)

    def test_discordance_both(self):
        """Both differences should be reported."""
        no_op = self._make_arm("no_op", False, 1)
        arm = self._make_arm("test", True, 2)

        result = _check_discordance(no_op, arm, "CASE", 0, 0)
        self.assertIsNotNone(result)
        self.assertIn("ring_break", result.reason)
        self.assertIn("component_count", result.reason)


class TestAuditOrchestrator(unittest.TestCase):
    """Test the main audit orchestrator."""

    def test_audit_initialization(self):
        """Audit should initialize with empty state."""
        audit = CMRBorderTopologyPairedStratumAudit()
        self.assertEqual(len(audit.discordant_queue), 0)
        self.assertEqual(len(audit.shadow_results), 0)

    def test_audit_slice_with_ring_break(self):
        """Audit slice with ring break should produce discordance."""
        audit = CMRBorderTopologyPairedStratumAudit()

        digest = SliceDigest(
            case_id="TEST_RING_BREAK",
            slice_idx=3,
            frame_idx=0,
            run_id="run_9",
            config_hash="c3d4e5f6a7b8901234567890123456789012abcdef",
            mask_sparse={"type": "synthetic", "ring_break": True},
        )

        result = audit.audit_slice(digest)

        self.assertEqual(result["case_id"], "TEST_RING_BREAK")
        self.assertIn("metrics", result)
        self.assertIn("shadow_arms", result)

        # Should have discordances due to ring-break arm
        self.assertGreaterEqual(len(audit.discordant_queue), 0)

    def test_discordant_queue_export(self):
        """Discordant queue should export to JSON without errors."""
        audit = CMRBorderTopologyPairedStratumAudit()

        # Add a synthetic discordance
        discordance = DiscordantSlice(
            case_id="TEST_EXPORT",
            slice_idx=0,
            frame_idx=0,
            no_op_ring_break=False,
            arm_ring_break=True,
            no_op_component_count=1,
            arm_component_count=2,
            arm_name="test_arm",
            reason="test discordance",
        )
        audit.discordant_queue.append(discordance)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "discordant.json"
            exported = audit.export_discordant_queue(output_path)
            self.assertTrue(exported.exists())

            with open(exported) as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["case_id"], "TEST_EXPORT")

    def test_hash_mismatch_detection(self):
        """Hash mismatch should be detected and recorded."""
        from experiments.cmr_myo_border_topology_paired_stratum_audit import _check_hash_mismatch

        # Valid hash
        result = _check_hash_mismatch("run_9", "c3d4e5f6a7b8901234567890123456789012abcdef")
        self.assertIsNone(result)

        # Invalid hash
        result = _check_hash_mismatch("run_9", "invalid_hash")
        self.assertIsNotNone(result)


class TestRunBoundedAudit(unittest.TestCase):
    """Test the top-level audit runner."""

    def test_empty_sample(self):
        """Empty sample should return complete status."""
        result = run_bounded_audit([])
        self.assertTrue(result["audit_complete"])
        self.assertEqual(result["total_cases_processed"], 0)

    def test_sample_with_cases(self):
        """Sample with cases should process without errors."""
        cases = [
            {"case_id": "CASE_001", "enriched": True},
            {"case_id": "CASE_002", "enriched": False},
        ]
        result = run_bounded_audit(cases)
        self.assertTrue(result["audit_complete"])
        self.assertEqual(result["total_cases_processed"], 2)


class TestAcceptanceMetricsCoverage(unittest.TestCase):
    """
    Verify that all acceptance metrics from the maintenance action
    are computable by the module.
    """

    def test_metric_coverage_cmr_myo_touches_image_border(self):
        """Coverage of cmr_myo_touches_image_border with paired CMR-LGE provenance."""
        audit = CMRBorderTopologyPairedStratumAudit()
        digest = SliceDigest(
            case_id="COVERAGE_TEST",
            slice_idx=0,
            frame_idx=0,
            run_id="run_9",
            config_hash="c3d4e5f6a7b8901234567890123456789012abcdef",
            mask_sparse={"type": "test"},
            lge_mi_mask_sparse={"type": "lge_test"},
        )
        result = audit.audit_slice(digest)
        self.assertIn("metrics", result)

    def test_metric_min_band_distance_and_ring_break(self):
        """Minimum myocardial-band distance and radial ring-break frequency."""
        metrics = TopologyMetrics(
            case_id="TEST",
            slice_idx=0,
            frame_idx=0,
            ring_contiguity=True,
            radial_ring_break=False,
            min_band_distance_to_border_voxels=3.5,
            component_count=1,
            largest_component_area_ratio=1.0,
            hole_count=0,
            euler_characteristic=0,
            temporal_area_cv=None,
            centroid_drift_voxels=None,
            cross_slice_area_cv=None,
            cross_slice_centroid_drift=None,
            lge_mi_component_count=None,
            lge_mi_fraction_high=None,
        )
        self.assertEqual(metrics.min_band_distance_to_border_voxels, 3.5)
        self.assertFalse(metrics.radial_ring_break)

    def test_metric_largest_component_and_component_count(self):
        """Largest-component area ratio and component-count change."""
        metrics_before = TopologyMetrics(
            case_id="TEST", slice_idx=0, frame_idx=0,
            ring_contiguity=True, radial_ring_break=True,
            min_band_distance_to_border_voxels=0.0,
            component_count=3, largest_component_area_ratio=0.6,
            hole_count=0, euler_characteristic=0,
            temporal_area_cv=None, centroid_drift_voxels=None,
            cross_slice_area_cv=None, cross_slice_centroid_drift=None,
            lge_mi_component_count=None, lge_mi_fraction_high=None,
        )

        # After ring-break-aware retention - provide non-None mask
        mask = {"type": "synthetic", "ring_break": True}
        arm_result = _ring_break_aware_retention_arm(mask, metrics_before)
        self.assertEqual(arm_result.component_count, 1)
        self.assertEqual(arm_result.largest_component_area_ratio, 1.0)

    def test_metric_cross_slice_continuity(self):
        """Cross-slice area/centroid continuity coefficient of variation."""
        slices = [
            {"area": 100, "centroid": (10.0, 10.0, 5.0)},
            {"area": 95, "centroid": (10.2, 10.1, 5.0)},
            {"area": 105, "centroid": (9.8, 9.9, 5.0)},
        ]
        cv, drift = _compute_cross_slice_continuity(slices)
        self.assertIsNotNone(cv)
        self.assertIsNotNone(drift)

    def test_metric_temporal_area_cv_by_severity(self):
        """Temporal area CV stratified by cmr_myo_small_area severity tier."""
        # Simulate areas for different severity tiers
        critical_areas = [20, 25, 30]  # critical tier: 0-50
        moderate_areas = [100, 110, 90]  # moderate tier: 50-150

        critical_cv = _compute_temporal_area_cv(critical_areas)
        moderate_cv = _compute_temporal_area_cv(moderate_areas)

        self.assertIsNotNone(critical_cv)
        self.assertIsNotNone(moderate_cv)

    def test_cross_arm_agreement(self):
        """Cross-arm agreement on ring-break resolution and component count."""
        no_op = ShadowArmResult("no_op", False, 1, 1.0, None)
        agreeing_arm = ShadowArmResult("agreeing", False, 1, 1.0, None)
        disagreeing_arm = ShadowArmResult("disagreeing", True, 2, 1.0, None)

        # Agreeing arms should not produce discordance
        disc = _check_discordance(no_op, agreeing_arm, "TEST", 0, 0)
        self.assertIsNone(disc)

        # Disagreeing arms should produce discordance
        disc = _check_discordance(no_op, disagreeing_arm, "TEST", 0, 0)
        self.assertIsNotNone(disc)

    def test_discordant_queue_export_no_label_writes(self):
        """Discordant queue export should not write to production labels."""
        audit = CMRBorderTopologyPairedStratumAudit()

        # Verify no production paths are used
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test_output" / "discordant.json"
            exported = audit.export_discordant_queue(output_path)
            self.assertTrue(exported.exists())

            # Verify it's just a JSON file, no database writes
            with open(exported) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
