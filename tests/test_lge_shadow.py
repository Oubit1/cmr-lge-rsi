from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.lge_shadow import (
    ShadowParameters,
    export_review_queue,
    extract_case_features,
    extract_slice_features,
    run_shadow_simulation,
    simulate_candidates,
    validate_split,
)


class LgeShadowTests(unittest.TestCase):
    def test_component_distance_and_ring_hole_features(self):
        myo = np.zeros((15, 15), dtype=bool)
        myo[2:13, 2:13] = True
        myo[5:10, 5:10] = False
        mi = np.zeros_like(myo)
        mi[3:5, 3:5] = True
        mi[8:10, 10:12] = True
        features = extract_slice_features("case", 0, mi, myo)
        self.assertEqual(features["mi_component_count"], 2)
        self.assertEqual(features["myo_hole_count"], 1)
        self.assertEqual(features["myo_euler_characteristic"], 0)
        self.assertEqual(features["mi_min_distance_to_myo"], 0.0)
        self.assertEqual(features["mi_overlap_with_myo"], 8)

    def test_case_features_have_adjacent_similarity(self):
        myo = np.zeros((2, 8, 8), dtype=bool)
        myo[:, 2:6, 2:6] = True
        mi = np.zeros_like(myo)
        mi[0, 3:4, 3:4] = True
        rows = extract_case_features("case", mi, myo)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["adjacent_slice_dice"], 1.0)
        self.assertEqual(rows[0]["adjacent_slice_iou"], 1.0)

    def test_candidates_are_pseudo_only_and_preserve_inputs(self):
        myo = np.ones((12, 12), dtype=bool)
        mi = np.zeros_like(myo)
        mi[2:4, 2:4] = True
        mi[8:9, 8:9] = True
        original = mi.copy()
        result = simulate_candidates(mi, myo, parameters=ShadowParameters(a_min=2))
        self.assertTrue(result["largest_component_only"]["pseudo_flag"])
        self.assertTrue(result["absolute_small_area_removal"]["pseudo_flag"])
        self.assertTrue(np.array_equal(mi, original))
        self.assertFalse(result["no_op"]["pseudo_flag"])

    def test_frozen_split_and_review_queue(self):
        params = ShadowParameters(a_min=20, d_max=1.0, connectivity=8)
        self.assertEqual(validate_split(["a"], ["b"], params)["parameters_frozen"]["a_min"], 20)
        with self.assertRaises(ValueError):
            validate_split(["a"], ["a"], params)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            export_review_queue([{"case_id": "case", "stratum": "mi_multiple_components"}], path)
            self.assertIn("case", path.read_text(encoding="utf-8"))

    def test_bounded_simulation_is_read_only_and_stratified(self):
        myo = np.ones((10, 10), dtype=bool)
        mi = np.zeros_like(myo)
        mi[2:4, 2:4] = True
        result = run_shadow_simulation(
            [{"case_id": "case_1", "mi_mask": mi, "myo_mask": myo,
              "issue_types": ["mi_multiple_components", "mi_fraction_high"]}],
            pilot_ids=["pilot"], validation_ids=["case_1"], blind_salt="test",
        )
        self.assertFalse(result["production_writes"])
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(len(result["review_queue"]), 2)
        self.assertIsNone(result["review_queue"][0]["case_id"])
        self.assertIn("no_op", result["distributions"])


if __name__ == "__main__":
    unittest.main()
