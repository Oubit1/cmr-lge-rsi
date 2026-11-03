"""Unit tests for experiments/cmr_s1_stratum_analysis.py

Runs offline without external data.
"""

import unittest

from experiments.cmr_s1_stratum_analysis import (
    run_s1_cmr_stratum_analysis,
    _filter_cmr_stratum,
    _compute_temporal_centroid_cv,
    _compute_attribution_fractions,
    _parse_records,
    S1DigestRecord,
)


class TestFilterCMRStratum(unittest.TestCase):
    def test_excludes_lge(self):
        records = [
            {"modality": "LGE", "mi_fragmentation_flag": False},
            {"modality": "CMR", "mi_fragmentation_flag": False},
        ]
        result = _filter_cmr_stratum(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["modality"], "CMR")

    def test_excludes_mi_fragmentation(self):
        records = [
            {"modality": "CMR", "mi_fragmentation_flag": True},
            {"modality": "CMR", "mi_fragmentation_flag": False},
        ]
        result = _filter_cmr_stratum(records)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["mi_fragmentation_flag"])


class TestTemporalCentroidCV(unittest.TestCase):
    def test_cv_computation(self):
        records = [
            S1DigestRecord("s1", "noop", False, centroid_x=1.0, centroid_y=2.0, centroid_z=3.0),
            S1DigestRecord("s2", "noop", False, centroid_x=2.0, centroid_y=3.0, centroid_z=4.0),
            S1DigestRecord("s3", "temporal_continuity", False, centroid_x=1.5, centroid_y=2.5, centroid_z=3.5),
        ]
        pre, post = _compute_temporal_centroid_cv(records)
        self.assertGreaterEqual(pre, 0.0)
        self.assertGreaterEqual(post, 0.0)

    def test_empty_returns_zero(self):
        pre, post = _compute_temporal_centroid_cv([])
        self.assertEqual(pre, 0.0)
        self.assertEqual(post, 0.0)


class TestAttributionFractions(unittest.TestCase):
    def test_all_phenotypes_explained(self):
        records = [
            S1DigestRecord(
                "s1", "noop", True,
                cmr_myo_small_area=True,
                cmr_myo_excess_components=True,
                cmr_myo_touches_image_border=True,
                myo_topology=True,
                cmr_centroid_drift=True,
                cmr_layer_area_jump=True,
                temporal_instability=True,
            ),
        ]
        result = _compute_attribution_fractions(records)
        self.assertEqual(result["unexplained_residual"], 0.0)
        self.assertGreater(result["cmr_myo_small_area"], 0.0)

    def test_unexplained_residual(self):
        records = [
            S1DigestRecord("s1", "noop", True),
        ]
        result = _compute_attribution_fractions(records)
        self.assertEqual(result["unexplained_residual"], 1.0)
        self.assertEqual(result["cmr_myo_small_area"], 0.0)

    def test_no_outliers(self):
        records = [
            S1DigestRecord("s1", "noop", False),
        ]
        result = _compute_attribution_fractions(records)
        self.assertEqual(result["unexplained_residual"], 1.0)


class TestRunAnalysis(unittest.TestCase):
    def test_full_pipeline(self):
        raw = [
            {
                "study_id": "cmr_001",
                "modality": "CMR",
                "mi_fragmentation_flag": False,
                "arm": "noop",
                "review_centroid_outlier": True,
                "cmr_myo_small_area": True,
                "cmr_myo_excess_components": False,
                "cmr_myo_touches_image_border": False,
                "myo_topology": False,
                "cmr_centroid_drift": False,
                "cmr_layer_area_jump": False,
                "temporal_instability": False,
                "centroid_x": 1.0,
                "centroid_y": 2.0,
                "centroid_z": 3.0,
            },
            {
                "study_id": "cmr_001",
                "modality": "CMR",
                "mi_fragmentation_flag": False,
                "arm": "largest_component",
                "review_centroid_outlier": True,
                "cmr_myo_small_area": True,
                "cmr_myo_excess_components": False,
                "cmr_myo_touches_image_border": False,
                "myo_topology": False,
                "cmr_centroid_drift": False,
                "cmr_layer_area_jump": False,
                "temporal_instability": False,
                "centroid_x": 1.1,
                "centroid_y": 2.1,
                "centroid_z": 3.1,
            },
            {
                "study_id": "cmr_001",
                "modality": "CMR",
                "mi_fragmentation_flag": False,
                "arm": "temporal_continuity",
                "review_centroid_outlier": False,
                "cmr_myo_small_area": False,
                "cmr_myo_excess_components": False,
                "cmr_myo_touches_image_border": False,
                "myo_topology": False,
                "cmr_centroid_drift": False,
                "cmr_layer_area_jump": False,
                "temporal_instability": False,
                "centroid_x": 1.05,
                "centroid_y": 2.05,
                "centroid_z": 3.05,
            },
            {
                "study_id": "lge_001",
                "modality": "LGE",
                "mi_fragmentation_flag": False,
                "arm": "noop",
                "review_centroid_outlier": True,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "centroid_z": 0.0,
            },
        ]
        report = run_s1_cmr_stratum_analysis(raw)
        self.assertEqual(report.total_cmr_cases, 3)
        self.assertGreaterEqual(report.outlier_persistence_noop, 0.0)
        self.assertGreaterEqual(report.outlier_persistence_largest, 0.0)
        self.assertGreaterEqual(report.outlier_persistence_temporal, 0.0)
        self.assertIn("cmr_myo_small_area", report.cooccurrence_rates)
        self.assertIn("cmr_myo_excess_components", report.cooccurrence_rates)
        self.assertIn("cmr_myo_touches_image_border", report.cooccurrence_rates)
        self.assertIn("myo_topology", report.cooccurrence_rates)
        self.assertIn("cmr_centroid_drift", report.cooccurrence_rates)
        self.assertGreaterEqual(report.temporal_centroid_cv_pre, 0.0)
        self.assertGreaterEqual(report.temporal_centroid_cv_post, 0.0)
        self.assertIn("noop_vs_largest_component", report.cross_arm_agreement)
        self.assertIn("noop_vs_temporal_continuity", report.cross_arm_agreement)
        self.assertIn("largest_component_vs_temporal_continuity", report.cross_arm_agreement)
        self.assertIn("cmr_myo_small_area", report.attribution_fractions)
        self.assertGreaterEqual(report.unexplained_residual, 0.0)
        self.assertLessEqual(report.unexplained_residual, 1.0)

    def test_empty_input(self):
        report = run_s1_cmr_stratum_analysis([])
        self.assertEqual(report.total_cmr_cases, 0)
        self.assertEqual(report.outlier_persistence_noop, 0.0)
        self.assertEqual(report.temporal_centroid_cv_pre, 0.0)
        self.assertEqual(report.temporal_centroid_cv_post, 0.0)


if __name__ == "__main__":
    unittest.main()
