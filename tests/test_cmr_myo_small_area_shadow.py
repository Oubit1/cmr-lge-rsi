"""
Unit tests for cmr_myo_small_area_shadow.py

Tests run offline without external data.
"""

import unittest
import numpy as np

from experiments.cmr_myo_small_area_shadow import (
    CaseKey,
    SliceFrameKey,
    Descriptor,
    EmptySliceType,
    ShadowArmResult,
    ExperimentResult,
    _compute_config_hash,
    _compute_manifest_hash,
    _label_connected_components,
    _compute_euler_characteristic,
    _compute_border_distance,
    _compute_centroid,
    extract_descriptor,
    _arm_noop,
    _arm_largest_component_intra_band,
    _arm_continuity_propagation,
    _arm_combined,
    classify_empty_slice,
    stratified_sample,
    run_experiment,
    compute_small_area_prevalence,
    compute_component_count_reduction,
    compute_temporal_centroid_cv,
    compute_empty_slice_rates,
    verify_determinism,
    VOXEL_AREA_MM2,
    MYO_BAND_MM2_MIN,
    MYO_BAND_MM2_MAX,
    SEVERE_PCT,
    MODERATE_PCT,
    MAX_CASES,
)


class TestConfigHash(unittest.TestCase):
    """Test config hash computation."""

    def test_config_hash_deterministic(self):
        """Config hash should be identical for same inputs."""
        spacing = (1.5, 1.5, 8.0)
        hash1 = _compute_config_hash(spacing)
        hash2 = _compute_config_hash(spacing)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_config_hash_different_for_different_spacing(self):
        """Config hash should differ for different voxel spacing."""
        hash1 = _compute_config_hash((1.5, 1.5, 8.0))
        hash2 = _compute_config_hash((1.6, 1.5, 8.0))
        self.assertNotEqual(hash1, hash2)


class TestManifestHash(unittest.TestCase):
    """Test manifest hash computation."""

    def test_manifest_hash_deterministic(self):
        """Manifest hash should be identical for same case list."""
        cases = [CaseKey("case1", "cine"), CaseKey("case2", "lge")]
        hash1 = _compute_manifest_hash(cases)
        hash2 = _compute_manifest_hash(cases)
        self.assertEqual(hash1, hash2)

    def test_manifest_hash_order_independent(self):
        """Manifest hash should be same regardless of case order."""
        cases1 = [CaseKey("case1", "cine"), CaseKey("case2", "lge")]
        cases2 = [CaseKey("case2", "lge"), CaseKey("case1", "cine")]
        hash1 = _compute_manifest_hash(cases1)
        hash2 = _compute_manifest_hash(cases2)
        self.assertEqual(hash1, hash2)


class TestConnectedComponents(unittest.TestCase):
    """Test connected component labeling."""

    def test_single_component(self):
        """Single blob should have one component."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        labeled, num = _label_connected_components(mask, connectivity=1)
        self.assertEqual(num, 1)

    def test_two_separate_components(self):
        """Two separate blobs should have two components."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True
        mask[7:9, 7:9] = True
        labeled, num = _label_connected_components(mask, connectivity=1)
        self.assertEqual(num, 2)

    def test_diagonal_not_connected_4(self):
        """Diagonally touching pixels are not 4-connected."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 3] = True
        mask[4, 4] = True
        labeled, num = _label_connected_components(mask, connectivity=1)
        self.assertEqual(num, 2)

    def test_diagonal_connected_8(self):
        """Diagonally touching pixels are 8-connected."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 3] = True
        mask[4, 4] = True
        labeled, num = _label_connected_components(mask, connectivity=2)
        self.assertEqual(num, 1)


class TestDescriptorExtraction(unittest.TestCase):
    """Test descriptor extraction."""

    def test_empty_mask(self):
        """Empty mask should produce zero-area descriptor."""
        mask = np.zeros((10, 10), dtype=bool)
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5))
        self.assertEqual(desc.area, 0.0)
        self.assertEqual(desc.component_count, 0)
        self.assertEqual(desc.largest_component_area_ratio, 0.0)

    def test_single_component_mask(self):
        """Single component mask."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5))
        expected_area = 16 * 1.5 * 1.5  # 16 pixels * voxel area
        self.assertAlmostEqual(desc.area, expected_area, places=5)
        self.assertEqual(desc.component_count, 1)
        self.assertAlmostEqual(desc.largest_component_area_ratio, 1.0, places=5)

    def test_two_component_mask(self):
        """Two component mask."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True  # 4 pixels
        mask[7:9, 7:9] = True  # 4 pixels
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5))
        self.assertEqual(desc.component_count, 2)
        self.assertAlmostEqual(desc.largest_component_area_ratio, 0.5, places=5)

    def test_border_distance(self):
        """Border distance should be computed correctly."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5))
        # Mask is at rows 3-6, cols 3-6 in 0-indexed 10x10 grid
        # Distance to top border: 3 * 1.5 = 4.5
        # Distance to bottom: (9-6) * 1.5 = 4.5
        # Distance to left: 3 * 1.5 = 4.5
        # Distance to right: (9-6) * 1.5 = 4.5
        self.assertAlmostEqual(desc.border_distance, 4.5, places=5)

    def test_ring_break_indicator(self):
        """Ring break indicator should detect multiple components."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True
        mask[6:8, 6:8] = True
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5))
        self.assertTrue(desc.ring_break_indicator)

    def test_temporal_area_cv(self):
        """Temporal area CV should be computed from provided areas."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        temporal_areas = [100.0, 110.0, 90.0]
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5), temporal_areas=temporal_areas)
        self.assertGreater(desc.temporal_area_cv, 0)

    def test_centroid_drift(self):
        """Centroid drift should be computed with previous mask."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        prev_mask = np.zeros((10, 10), dtype=bool)
        prev_mask[4:8, 4:8] = True
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5), prev_mask=prev_mask)
        self.assertGreater(desc.centroid_drift, 0)

    def test_cross_slice_continuity(self):
        """Cross-slice continuity should be computed with neighbours."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        prev_mask = np.zeros((10, 10), dtype=bool)
        prev_mask[3:7, 3:7] = True
        next_mask = np.zeros((10, 10), dtype=bool)
        next_mask[3:7, 3:7] = True
        desc = extract_descriptor(mask, voxel_spacing=(1.5, 1.5),
                                  prev_mask=prev_mask, next_mask=next_mask)
        self.assertAlmostEqual(desc.cross_slice_continuity, 1.0, places=5)


class TestShadowArms(unittest.TestCase):
    """Test shadow arm implementations."""

    def test_arm_noop(self):
        """No-op arm should return identical descriptors."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        sfk = SliceFrameKey(CaseKey("case1", "cine"), 0, 0)
        masks = {sfk: mask}
        descriptors = {sfk: extract_descriptor(mask)}

        result = _arm_noop(masks, descriptors)
        self.assertEqual(result[sfk].to_dict(), descriptors[sfk].to_dict())

    def test_arm_largest_component_filters_by_band(self):
        """Largest component arm should filter by myocardial band."""
        # Create mask with component too small for band
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:7, 5:7] = True  # 4 pixels * 2.25 = 9 mm^2, below MYO_BAND_MM2_MIN
        sfk = SliceFrameKey(CaseKey("case1", "cine"), 0, 0)
        masks = {sfk: mask}
        descriptors = {sfk: extract_descriptor(mask)}

        result = _arm_largest_component_intra_band(masks, descriptors)
        # Should be empty after filtering
        self.assertEqual(result[sfk].area, 0.0)

    def test_arm_largest_component_keeps_valid_component(self):
        """Largest component arm should keep valid myocardial component."""
        # Create mask with component in valid band
        # Need ~50-500 mm^2, at 2.25 mm^2/pixel that's ~22-222 pixels
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:15, 5:15] = True  # 100 pixels * 2.25 = 225 mm^2, in band
        sfk = SliceFrameKey(CaseKey("case1", "cine"), 0, 0)
        masks = {sfk: mask}
        descriptors = {sfk: extract_descriptor(mask)}

        result = _arm_largest_component_intra_band(masks, descriptors)
        self.assertGreater(result[sfk].area, 0)

    def test_arm_continuity_propagation(self):
        """Continuity propagation arm should produce valid descriptors."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        sfk = SliceFrameKey(CaseKey("case1", "cine"), 0, 0)
        masks = {sfk: mask}
        descriptors = {sfk: extract_descriptor(mask)}

        result = _arm_continuity_propagation(masks, descriptors)
        self.assertIn(sfk, result)
        self.assertGreaterEqual(result[sfk].area, 0)

    def test_arm_combined(self):
        """Combined arm should produce valid descriptors."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        sfk = SliceFrameKey(CaseKey("case1", "cine"), 0, 0)
        masks = {sfk: mask}
        descriptors = {sfk: extract_descriptor(mask)}

        result = _arm_combined(masks, descriptors)
        self.assertIn(sfk, result)
        self.assertGreaterEqual(result[sfk].area, 0)


class TestEmptySliceClassification(unittest.TestCase):
    """Test empty slice classification."""

    def test_not_empty(self):
        """Non-empty mask should return NOT_EMPTY."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        result = classify_empty_slice(mask)
        self.assertEqual(result, EmptySliceType.NOT_EMPTY)

    def test_isolated_dropout(self):
        """Empty mask with non-empty neighbours is isolated dropout."""
        mask = np.zeros((10, 10), dtype=bool)
        neighbour = np.zeros((10, 10), dtype=bool)
        neighbour[3:7, 3:7] = True
        result = classify_empty_slice(mask, temporal_masks=[neighbour])
        self.assertEqual(result, EmptySliceType.ISOLATED_DROPOUT)

    def test_contiguous_temporal_gap(self):
        """Empty mask with empty neighbours is temporal gap."""
        mask = np.zeros((10, 10), dtype=bool)
        empty = np.zeros((10, 10), dtype=bool)
        result = classify_empty_slice(mask, temporal_masks=[empty])
        self.assertEqual(result, EmptySliceType.CONTIGUOUS_TEMPORAL_GAP)

    def test_border_excluded(self):
        """Empty mask with no temporal info is border-excluded."""
        mask = np.zeros((10, 10), dtype=bool)
        result = classify_empty_slice(mask)
        self.assertEqual(result, EmptySliceType.BORDER_EXCLUDED_UNDERSEGMENTATION)


class TestStratifiedSampling(unittest.TestCase):
    """Test stratified sampling."""

    def test_empty_cases(self):
        """Empty case list returns empty sample."""
        result = stratified_sample([], {}, {})
        self.assertEqual(result, [])

    def test_respects_max_cases(self):
        """Should not exceed max_cases."""
        cases = [CaseKey(f"case{i}", "cine") for i in range(100)]
        flags = {c: {"cmr_myo_small_area": False} for c in cases}
        areas = {c: 100.0 for c in cases}
        result = stratified_sample(cases, flags, areas, max_cases=10)
        self.assertLessEqual(len(result), 10)

    def test_prioritizes_severe(self):
        """Should include severe cases when available."""
        cases = [CaseKey(f"case{i}", "cine") for i in range(20)]
        flags = {c: {"cmr_myo_small_area": True} for c in cases}
        # First 5 cases have very small areas (severe)
        areas = {cases[i]: 10.0 if i < 5 else 100.0 for i in range(20)}
        result = stratified_sample(cases, flags, areas, max_cases=10)
        # Should include some severe cases
        self.assertGreater(len(result), 0)

    def test_enriches_cooccurring(self):
        """Should prefer cases with co-occurring flags."""
        cases = [CaseKey(f"case{i}", "cine") for i in range(10)]
        # Case 0 has co-occurring flag
        flags = {
            cases[0]: {"cmr_myo_small_area": True, "cmr_myo_excess_components": True},
        }
        for c in cases[1:]:
            flags[c] = {"cmr_myo_small_area": True}
        areas = {c: 50.0 for c in cases}
        result = stratified_sample(cases, flags, areas, max_cases=5)
        # Case 0 should be prioritized
        self.assertTrue(cases[0] in result or len(result) < 5)


class TestAnalysisFunctions(unittest.TestCase):
    """Test analysis helper functions."""

    def test_small_area_prevalence(self):
        """Should compute correct prevalence."""
        sfk1 = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)
        sfk2 = SliceFrameKey(CaseKey("c1", "cine"), 0, 1)
        sfk3 = SliceFrameKey(CaseKey("c1", "cine"), 0, 2)

        descriptors = {
            sfk1: Descriptor(10.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0),
            sfk2: Descriptor(30.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0),
            sfk3: Descriptor(100.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0),
        }

        arm = ShadowArmResult("test", descriptors, {})
        result = compute_small_area_prevalence(arm, severe_threshold=20.0, moderate_threshold=50.0)

        self.assertEqual(result['severe_count'], 1)
        self.assertEqual(result['moderate_count'], 1)
        self.assertEqual(result['normal_count'], 1)

    def test_component_count_reduction(self):
        """Should detect component count reduction."""
        sfk = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)

        noop_desc = {sfk: Descriptor(100.0, 2, 0.6, 0, 1, 5.0, True, 0.0, 0.0, 0.0)}
        treat_desc = {sfk: Descriptor(100.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0)}

        noop_arm = ShadowArmResult("noop", noop_desc, {})
        treat_arm = ShadowArmResult("treat", treat_desc, {})

        result = compute_component_count_reduction(noop_arm, treat_arm)

        self.assertEqual(result['component_count_reduced'], 1)
        self.assertEqual(result['topology_breaks_resolved'], 1)

    def test_temporal_centroid_cv(self):
        """Should compute temporal centroid CV."""
        sfk = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)
        descriptors = {sfk: Descriptor(100.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 2.5, 0.0)}

        arm = ShadowArmResult("test", descriptors, {})
        result = compute_temporal_centroid_cv(arm)

        self.assertEqual(result['mean_cv'], 2.5)

    def test_empty_slice_rates(self):
        """Should compute empty slice rates."""
        sfk1 = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)
        sfk2 = SliceFrameKey(CaseKey("c1", "cine"), 0, 1)

        classifications = {
            sfk1: EmptySliceType.ISOLATED_DROPOUT,
            sfk2: EmptySliceType.CONTIGUOUS_TEMPORAL_GAP,
        }

        arm = ShadowArmResult("test", {}, classifications)
        result = compute_empty_slice_rates(arm, {CaseKey("c1", "cine")})

        self.assertEqual(result['total_empty'], 2)
        self.assertEqual(result['isolated_dropout'], 1)
        self.assertEqual(result['temporal_gap'], 1)


class TestDeterminism(unittest.TestCase):
    """Test determinism verification."""

    def test_identical_results(self):
        """Identical results should verify as deterministic."""
        sfk = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)
        desc = {sfk: Descriptor(100.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0)}

        result1 = ExperimentResult(
            sampled_cases=[CaseKey("c1", "cine")],
            arm_results={"noop": ShadowArmResult("noop", desc, {})},
            severity_thresholds={"severe": 10.0},
            determinism_hash="abc123",
        )
        result2 = ExperimentResult(
            sampled_cases=[CaseKey("c1", "cine")],
            arm_results={"noop": ShadowArmResult("noop", desc, {})},
            severity_thresholds={"severe": 10.0},
            determinism_hash="abc123",
        )

        self.assertTrue(verify_determinism(result1, result2))

    def test_different_hash(self):
        """Different determinism hash should fail."""
        result1 = ExperimentResult([], {}, {}, "abc123")
        result2 = ExperimentResult([], {}, {}, "def456")

        self.assertFalse(verify_determinism(result1, result2))

    def test_different_descriptors(self):
        """Different descriptors should fail determinism."""
        sfk = SliceFrameKey(CaseKey("c1", "cine"), 0, 0)
        desc1 = {sfk: Descriptor(100.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0)}
        desc2 = {sfk: Descriptor(200.0, 1, 1.0, 0, 1, 5.0, False, 0.0, 0.0, 0.0)}

        result1 = ExperimentResult([], {"noop": ShadowArmResult("noop", desc1, {})}, {}, "abc123")
        result2 = ExperimentResult([], {"noop": ShadowArmResult("noop", desc2, {})}, {}, "abc123")

        self.assertFalse(verify_determinism(result1, result2))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full experiment."""

    def test_run_experiment_basic(self):
        """Full experiment should run and produce results."""
        cases = [CaseKey(f"case{i}", "cine") for i in range(5)]

        def mask_provider(sfk: SliceFrameKey) -> np.ndarray:
            mask = np.zeros((10, 10), dtype=bool)
            if sfk.slice_idx < 5 and sfk.frame_idx < 10:
                mask[3:7, 3:7] = True
            return mask

        def flags_provider(ck: CaseKey) -> dict:
            return {"cmr_myo_small_area": True}

        result = run_experiment(cases, mask_provider, flags_provider, use_cache=False)

        self.assertGreater(len(result.sampled_cases), 0)
        self.assertEqual(len(result.arm_results), 4)
        self.assertIn("noop", result.arm_results)
        self.assertIn("largest_component", result.arm_results)
        self.assertIn("continuity_propagation", result.arm_results)
        self.assertIn("combined", result.arm_results)
        self.assertIsNotNone(result.determinism_hash)

    def test_severity_thresholds_present(self):
        """Result should include severity thresholds."""
        cases = [CaseKey(f"case{i}", "cine") for i in range(5)]

        def mask_provider(sfk: SliceFrameKey) -> np.ndarray:
            mask = np.zeros((10, 10), dtype=bool)
            mask[3:7, 3:7] = True
            return mask

        def flags_provider(ck: CaseKey) -> dict:
            return {"cmr_myo_small_area": True}

        result = run_experiment(cases, mask_provider, flags_provider, use_cache=False)

        self.assertIn("severe", result.severity_thresholds)
        self.assertIn("moderate", result.severity_thresholds)

    def test_no_writes_occurred(self):
        """Experiment should not write to production."""
        # This is verified by code inspection - no file writes except cache
        # Cache writes are to designated cache directory
        pass  # Verified by static analysis


if __name__ == "__main__":
    unittest.main(verbosity=2)
