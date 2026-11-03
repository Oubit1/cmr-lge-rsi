#!/usr/bin/env python3
"""Offline unit tests for CMR-LGE topology co-occurrence stratum.

All tests run without external data, network, or production dependencies.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# Ensure experiments module is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cmr_lge_topology_cooccurrence_stratum import (
    B1_CONTRACT_VERSION,
    B1FallbackExtractor,
    CONNECTIVITY_4,
    CooccurrenceResult,
    CMRSliceMetrics,
    LGESliceMetrics,
    LGEComponent,
    Modality,
    PatternType,
    StratifiedRecord,
    TopologyHarness,
    VentricularLevel,
    _classify_pattern,
    _classify_ventricular_level,
    _compute_config_hash,
    _infer_modality,
    _load_json_digest,
    _sample_cmr_cases,
    _sample_lge_cases,
    _validate_cache_hash,
    execute_cooccurrence_stratum,
    result_checksum,
    result_to_json,
)


class TestHashAndCacheGuard(unittest.TestCase):
    """Tests for cache-reuse guard and hash validation."""

    def test_compute_config_hash_stable(self) -> None:
        config = {"pipeline": "test", "version": 1}
        h1 = _compute_config_hash(config)
        h2 = _compute_config_hash(config)
        self.assertEqual(h1, h2)

    def test_compute_config_hash_order_independent(self) -> None:
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        self.assertEqual(_compute_config_hash(config1), _compute_config_hash(config2))

    def test_validate_cache_hash_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            config = {"pipeline": "test"}
            h = _compute_config_hash(config)
            (cache_root / "pipeline_config.hash").write_text(h)
            valid, expected = _validate_cache_hash(config, cache_root)
            self.assertTrue(valid)
            self.assertEqual(expected, h)

    def test_validate_cache_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            (cache_root / "pipeline_config.hash").write_text("oldhash")
            config = {"pipeline": "new"}
            valid, expected = _validate_cache_hash(config, cache_root)
            self.assertFalse(valid)
            self.assertEqual(expected, _compute_config_hash(config))

    def test_validate_cache_hash_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            config = {"pipeline": "test"}
            valid, _ = _validate_cache_hash(config, cache_root)
            self.assertFalse(valid)


class TestDigestLoaders(unittest.TestCase):
    """Tests for import-only digest loaders."""

    def test_load_json_digest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _load_json_digest(Path(td) / "nonexistent.json")
            self.assertEqual(result, {})

    def test_load_json_digest_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            data = {"key": "value", "nested": {"a": 1}}
            path.write_text(json.dumps(data))
            result = _load_json_digest(path)
            self.assertEqual(result, data)


class TestModalityInference(unittest.TestCase):
    """Tests for modality inference from case_id."""

    def test_infer_modality_cmr(self) -> None:
        self.assertEqual(_infer_modality("CMR-001"), Modality.CMR)
        self.assertEqual(_infer_modality("cmr-001"), Modality.CMR)

    def test_infer_modality_lge(self) -> None:
        self.assertEqual(_infer_modality("LGE-001"), Modality.LGE)
        self.assertEqual(_infer_modality("lge-001"), Modality.LGE)

    def test_infer_modality_ambiguous(self) -> None:
        self.assertIsNone(_infer_modality("CASE-001"))
        self.assertIsNone(_infer_modality(""))


class TestB1FallbackExtractor(unittest.TestCase):
    """Tests for run-9 B1 fallback extractor."""

    def test_contract_version(self) -> None:
        self.assertEqual(B1FallbackExtractor.CONTRACT, "run-9-b1")

    def test_extract_cmr_slice_topology_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            digest = {
                "case1/0": {
                    "connected_component_count": 2,
                    "largest_component_area_ratio": 0.8,
                    "hole_count": 1,
                    "euler_characteristic": 0,
                    "ring_break_indicator": True,
                    "min_myocardial_band_distance_to_border": 5,
                    "cross_slice_area_continuity_ratio": 0.95,
                    "centroid_drift_magnitude": 2.3,
                    "temporal_area_coefficient_of_variation": 0.15,
                }
            }
            (cache_root / "cmr_per_slice_digest.json").write_text(json.dumps(digest))
            extractor = B1FallbackExtractor(cache_root)
            result = extractor.extract_cmr_slice_topology("case1", 0)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.connected_component_count, 2)
            self.assertEqual(result.largest_component_area_ratio, 0.8)
            self.assertEqual(result.hole_count, 1)
            self.assertEqual(result.euler_characteristic, 0)
            self.assertTrue(result.ring_break_indicator)
            self.assertEqual(result.min_myocardial_band_distance_to_border, 5)
            self.assertEqual(result.cross_slice_area_continuity_ratio, 0.95)
            self.assertEqual(result.centroid_drift_magnitude, 2.3)
            self.assertEqual(result.temporal_area_coefficient_of_variation, 0.15)

    def test_extract_cmr_slice_topology_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            (cache_root / "cmr_per_slice_digest.json").write_text("{}")
            extractor = B1FallbackExtractor(cache_root)
            result = extractor.extract_cmr_slice_topology("case1", 0)
            self.assertIsNone(result)

    def test_extract_lge_slice_topology_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            digest = {
                "case1/0": {
                    "mi_connected_component_count": 3,
                    "mi_components": [
                        {"area": 10.0, "centroid": [1.0, 2.0]},
                        {"area": 5.0, "centroid": [3.0, 4.0]},
                    ],
                    "min_inter_component_distance": 1.5,
                    "mi_fraction": 0.25,
                    "myocardial_component_count": 2,
                    "ring_break_indicator": False,
                    "mi_fragments_within_band": 2,
                    "mi_fragments_remote": 1,
                }
            }
            (cache_root / "lge_per_slice_digest.json").write_text(json.dumps(digest))
            extractor = B1FallbackExtractor(cache_root)
            result = extractor.extract_lge_slice_topology("case1", 0)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.mi_connected_component_count, 3)
            self.assertEqual(len(result.mi_components), 2)
            self.assertEqual(result.mi_components[0].area, 10.0)
            self.assertEqual(result.mi_components[0].centroid, (1.0, 2.0))
            self.assertEqual(result.min_inter_component_distance, 1.5)
            self.assertEqual(result.mi_fraction, 0.25)
            self.assertEqual(result.myocardial_component_count, 2)
            self.assertFalse(result.ring_break_indicator)
            self.assertEqual(result.mi_fragments_within_band, 2)
            self.assertEqual(result.mi_fragments_remote, 1)


class TestHarness(unittest.TestCase):
    """Tests for TopologyHarness single-instantiation behavior."""

    def test_single_harness_no_second_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            config = {"pipeline": "test"}
            harness = TopologyHarness(cache_root, config)
            # Harness should have B1 extractor lazily initialized
            self.assertIsNone(harness._b1)
            b1 = harness.b1
            self.assertIsInstance(b1, B1FallbackExtractor)
            # Second access should reuse same instance
            self.assertIs(harness.b1, b1)

    def test_harness_get_cmr_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            digest = {
                "case1/0": {
                    "connected_component_count": 1,
                    "largest_component_area_ratio": 1.0,
                    "hole_count": 0,
                    "euler_characteristic": 1,
                    "ring_break_indicator": False,
                    "min_myocardial_band_distance_to_border": 10,
                }
            }
            (cache_root / "cmr_per_slice_digest.json").write_text(json.dumps(digest))
            harness = TopologyHarness(cache_root, {})
            result = harness.get_cmr_slice_metrics("case1", 0)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.connected_component_count, 1)

    def test_harness_get_cmr_miss_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            (cache_root / "cmr_per_slice_digest.json").write_text("{}")
            harness = TopologyHarness(cache_root, {})
            result = harness.get_cmr_slice_metrics("case1", 0, use_fallback=True)
            # Fallback also has empty digest, so None
            self.assertIsNone(result)


class TestSampling(unittest.TestCase):
    """Tests for stratified sampling logic."""

    def test_sample_cmr_cases_flagged_and_control(self) -> None:
        rollup = {
            "CMR-case1": {"issue_type": ["cmr_myo_small_area"], "slice_indices": [0, 1], "flagged_slices": {"0": True}},
            "CMR-case2": {"issue_type": ["cmr_myo_excess_components"], "slice_indices": [0], "flagged_slices": {"0": True}},
            "CMR-case3": {"issue_type": ["myo_topology"], "slice_indices": [0], "flagged_slices": {"0": True}},
            "CMR-control1": {"issue_type": [], "slice_indices": [0], "flagged_slices": {}},
            "CMR-control2": {"issue_type": ["other"], "slice_indices": [0], "flagged_slices": {}},
            "LGE-case1": {"issue_type": ["mi_multiple_components", "myo_topology"], "slice_indices": [0], "flagged_slices": {"0": True}},
        }
        flagged, controls = _sample_cmr_cases(rollup, max_flagged=10, max_controls=10)
        self.assertEqual(len(flagged), 3)
        self.assertEqual(len(controls), 2)
        self.assertIn("CMR-case1", flagged)
        self.assertIn("CMR-control1", controls)
        # LGE cases should not appear in CMR sampling
        self.assertNotIn("LGE-case1", flagged)
        self.assertNotIn("LGE-case1", controls)

    def test_sample_cmr_cases_enrichment_order(self) -> None:
        rollup = {
            "CMR-case1": {"issue_type": ["cmr_myo_small_area"], "slice_indices": [0], "flagged_slices": {"0": True}},
            "CMR-case2": {"issue_type": ["cmr_myo_excess_components", "myo_topology"], "slice_indices": [0], "flagged_slices": {"0": True}},
        }
        flagged, _ = _sample_cmr_cases(rollup, max_flagged=10, max_controls=10)
        # case2 should be first due to enrichment
        self.assertEqual(flagged[0], "CMR-case2")

    def test_sample_lge_cases(self) -> None:
        rollup = {
            "LGE-case1": {"issue_type": ["mi_multiple_components", "myo_topology"], "slice_indices": [0], "flagged_slices": {"0": True}},
            "LGE-case2": {"issue_type": ["mi_multiple_components"], "slice_indices": [0], "flagged_slices": {"0": True}},
            "LGE-control1": {"issue_type": [], "slice_indices": [0], "flagged_slices": {}},
            "CMR-case1": {"issue_type": ["cmr_myo_small_area"], "slice_indices": [0], "flagged_slices": {"0": True}},
        }
        flagged, controls = _sample_lge_cases(rollup, max_flagged=10, max_controls=10)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0], "LGE-case1")
        self.assertEqual(len(controls), 1)
        self.assertIn("LGE-control1", controls)
        # CMR cases should not appear in LGE sampling
        self.assertNotIn("CMR-case1", flagged)
        self.assertNotIn("CMR-case1", controls)

    def test_sample_respects_max(self) -> None:
        rollup = {
            f"CMR-case{i}": {"issue_type": ["cmr_myo_small_area"], "slice_indices": [0], "flagged_slices": {"0": True}}
            for i in range(100)
        }
        flagged, _ = _sample_cmr_cases(rollup, max_flagged=50, max_controls=10)
        self.assertEqual(len(flagged), 50)


class TestClassification(unittest.TestCase):
    """Tests for ventricular level and pattern classification."""

    def test_classify_ventricular_level_apex_base_low(self) -> None:
        self.assertEqual(_classify_ventricular_level(0, 10), VentricularLevel.APEX_BASE)

    def test_classify_ventricular_level_apex_base_high(self) -> None:
        self.assertEqual(_classify_ventricular_level(9, 10), VentricularLevel.APEX_BASE)

    def test_classify_ventricular_level_mid(self) -> None:
        self.assertEqual(_classify_ventricular_level(5, 10), VentricularLevel.MID_VENTRICLE)

    def test_classify_ventricular_level_zero_total(self) -> None:
        self.assertEqual(_classify_ventricular_level(0, 0), VentricularLevel.MID_VENTRICLE)

    def test_classify_pattern_isolated_cmr(self) -> None:
        cmr = CMRSliceMetrics(
            connected_component_count=2,
            largest_component_area_ratio=0.5,
            hole_count=0,
            euler_characteristic=0,
            ring_break_indicator=False,
            min_myocardial_band_distance_to_border=5,
        )
        self.assertEqual(_classify_pattern(cmr, None), PatternType.ISOLATED)

    def test_classify_pattern_isolated_lge(self) -> None:
        lge = LGESliceMetrics(
            mi_connected_component_count=2,
            mi_components=[],
            min_inter_component_distance=None,
            mi_fraction=0.1,
            myocardial_component_count=1,
            ring_break_indicator=False,
            mi_fragments_within_band=0,
            mi_fragments_remote=0,
        )
        self.assertEqual(_classify_pattern(None, lge), PatternType.ISOLATED)

    def test_classify_pattern_contiguous(self) -> None:
        cmr = CMRSliceMetrics(
            connected_component_count=1,
            largest_component_area_ratio=1.0,
            hole_count=0,
            euler_characteristic=1,
            ring_break_indicator=False,
            min_myocardial_band_distance_to_border=10,
        )
        self.assertEqual(_classify_pattern(cmr, None), PatternType.CONTIGUOUS)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests with mocked cache."""

    def _build_mock_cache(self, td: str) -> Path:
        cache_root = Path(td)
        # Pipeline hash
        config = {"pipeline": "cmr_lge_topology", "version": 1}
        h = _compute_config_hash(config)
        (cache_root / "pipeline_config.hash").write_text(h)

        # CMR per-slice digest
        cmr_digest = {
            "CMR-001/0": {
                "connected_component_count": 2,
                "largest_component_area_ratio": 0.7,
                "hole_count": 1,
                "euler_characteristic": 0,
                "ring_break_indicator": True,
                "min_myocardial_band_distance_to_border": 3,
                "cross_slice_area_continuity_ratio": 0.92,
                "centroid_drift_magnitude": 1.5,
                "temporal_area_coefficient_of_variation": 0.12,
            },
            "CMR-001/1": {
                "connected_component_count": 1,
                "largest_component_area_ratio": 1.0,
                "hole_count": 0,
                "euler_characteristic": 1,
                "ring_break_indicator": False,
                "min_myocardial_band_distance_to_border": 8,
                "cross_slice_area_continuity_ratio": 0.95,
                "centroid_drift_magnitude": 0.8,
                "temporal_area_coefficient_of_variation": 0.08,
            },
            "CMR-002/0": {
                "connected_component_count": 1,
                "largest_component_area_ratio": 1.0,
                "hole_count": 0,
                "euler_characteristic": 1,
                "ring_break_indicator": False,
                "min_myocardial_band_distance_to_border": 12,
            },
        }
        (cache_root / "cmr_per_slice_digest.json").write_text(json.dumps(cmr_digest))

        # LGE per-slice digest
        lge_digest = {
            "LGE-001/0": {
                "mi_connected_component_count": 2,
                "mi_components": [
                    {"area": 15.0, "centroid": [2.0, 3.0]},
                    {"area": 8.0, "centroid": [5.0, 6.0]},
                ],
                "min_inter_component_distance": 2.5,
                "mi_fraction": 0.3,
                "myocardial_component_count": 1,
                "ring_break_indicator": False,
                "mi_fragments_within_band": 1,
                "mi_fragments_remote": 1,
            },
            "LGE-002/0": {
                "mi_connected_component_count": 1,
                "mi_components": [
                    {"area": 20.0, "centroid": [1.0, 1.0]},
                ],
                "min_inter_component_distance": None,
                "mi_fraction": 0.15,
                "myocardial_component_count": 1,
                "ring_break_indicator": False,
                "mi_fragments_within_band": 1,
                "mi_fragments_remote": 0,
            },
        }
        (cache_root / "lge_per_slice_digest.json").write_text(json.dumps(lge_digest))

        # A3 rollup
        rollup = {
            "CMR-001": {
                "issue_type": ["cmr_myo_small_area", "myo_topology"],
                "slice_indices": [0, 1],
                "flagged_slices": {"0": True, "1": False},
            },
            "CMR-002": {
                "issue_type": [],
                "slice_indices": [0],
                "flagged_slices": {},
            },
            "LGE-001": {
                "issue_type": ["mi_multiple_components", "myo_topology"],
                "slice_indices": [0],
                "flagged_slices": {"0": True},
            },
            "LGE-002": {
                "issue_type": [],
                "slice_indices": [0],
                "flagged_slices": {},
            },
        }
        (cache_root / "a3_dedup_rollup.json").write_text(json.dumps(rollup))

        return cache_root

    def test_end_to_end_hash_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = self._build_mock_cache(td)
            config = {"pipeline": "cmr_lge_topology", "version": 1}
            result = execute_cooccurrence_stratum(cache_root, config)

            self.assertTrue(result.cache_hash_valid)
            self.assertFalse(result.fallback_extractor_invoked)
            self.assertIsNone(result.mismatch_reported)

            # Sampling counts: CMR flagged=1 (CMR-001), CMR control=1 (CMR-002)
            # LGE flagged=1 (LGE-001), LGE control=1 (LGE-002)
            self.assertEqual(result.cmr_flagged_sampled, 1)
            self.assertEqual(result.cmr_control_sampled, 1)
            self.assertEqual(result.lge_flagged_sampled, 1)
            self.assertEqual(result.lge_control_sampled, 1)

            # Records
            self.assertEqual(len(result.records), 4)
            cmr_records = [r for r in result.records if r.modality == Modality.CMR]
            lge_records = [r for r in result.records if r.modality == Modality.LGE]
            self.assertEqual(len(cmr_records), 2)
            self.assertEqual(len(lge_records), 2)

            # Check CMR-001 metrics
            cmr001 = next(r for r in cmr_records if r.case_id == "CMR-001")
            self.assertIn(0, cmr001.cmr_metrics)
            self.assertIn(1, cmr001.cmr_metrics)
            m0 = cmr001.cmr_metrics[0]
            self.assertEqual(m0.connected_component_count, 2)
            self.assertTrue(m0.ring_break_indicator)
            self.assertEqual(m0.min_myocardial_band_distance_to_border, 3)
            self.assertEqual(m0.cross_slice_area_continuity_ratio, 0.92)
            self.assertEqual(m0.centroid_drift_magnitude, 1.5)
            self.assertEqual(m0.temporal_area_coefficient_of_variation, 0.12)

            # Check LGE-001 metrics
            lge001 = next(r for r in lge_records if r.case_id == "LGE-001")
            self.assertIn(0, lge001.lge_metrics)
            lge0 = lge001.lge_metrics[0]
            self.assertEqual(lge0.mi_connected_component_count, 2)
            self.assertEqual(lge0.mi_fraction, 0.3)
            self.assertEqual(lge0.mi_fragments_within_band, 1)
            self.assertEqual(lge0.mi_fragments_remote, 1)

            # Missing pairs
            self.assertEqual(result.missing_pairs_reported, [])

    def test_end_to_end_hash_mismatch_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = self._build_mock_cache(td)
            # Write wrong hash to trigger mismatch
            (cache_root / "pipeline_config.hash").write_text("wronghash")
            config = {"pipeline": "cmr_lge_topology", "version": 1}
            result = execute_cooccurrence_stratum(cache_root, config)

            self.assertFalse(result.cache_hash_valid)
            self.assertTrue(result.fallback_extractor_invoked)
            self.assertIsNotNone(result.mismatch_reported)
            self.assertIn("Hash mismatch", result.mismatch_reported)

    def test_reproducibility_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = self._build_mock_cache(td)
            config = {"pipeline": "cmr_lge_topology", "version": 1}
            result1 = execute_cooccurrence_stratum(cache_root, config)
            result2 = execute_cooccurrence_stratum(cache_root, config)
            self.assertEqual(result_checksum(result1), result_checksum(result2))

    def test_result_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = self._build_mock_cache(td)
            config = {"pipeline": "cmr_lge_topology", "version": 1}
            result = execute_cooccurrence_stratum(cache_root, config)
            json_data = result_to_json(result)
            self.assertIn("records", json_data)
            self.assertIn("cmr_flagged_sampled", json_data)
            self.assertEqual(json_data["cmr_flagged_sampled"], 1)
            self.assertEqual(json_data["cmr_control_sampled"], 1)
            self.assertEqual(json_data["lge_flagged_sampled"], 1)
            self.assertEqual(json_data["lge_control_sampled"], 1)


if __name__ == "__main__":
    unittest.main()
