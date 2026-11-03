from __future__ import annotations

import unittest

from experiments.advisory_rules import (
    AdvisoryParameters,
    area_floor_pixels,
    decompose_myo_topology,
    evaluate_mi_fragment_rule,
    validate_advisory_rows,
)


class AdvisoryRuleTests(unittest.TestCase):
    def _row(self, **updates):
        row = {
            "slice_id": "s1", "mi_component_count": 2,
            "mi_components": [
                {"area": 50, "centroid_boundary_distance": 4},
                {"area": 10, "centroid_boundary_distance": 1},
            ],
            "myo_radial_ring_break_count": 1,
            "myo_component_count": 1, "myo_hole_count": 1,
            "myo_hole_area": 9, "myo_euler_characteristic": 0,
            "myo_gap_width": 2, "myo_border_touch": False, "myo_area": 100,
        }
        row.update(updates)
        return row

    def test_fragment_rule_requires_all_conditions(self):
        params = AdvisoryParameters(a_min_mm2=10)
        self.assertEqual(area_floor_pixels(1.0, params), 10)
        result = evaluate_mi_fragment_rule(self._row(), voxel_area_mm2=1.0, parameters=params)
        self.assertTrue(result["pseudo_flag"])
        for key in ("component_count_at_least_two", "non_dominant_small_component",
                    "small_component_near_myo_boundary", "myo_ring_break_present"):
            changed = self._row()
            changed["myo_radial_ring_break_count"] = 0 if key == "myo_ring_break_present" else 1
            if key == "component_count_at_least_two":
                changed["mi_component_count"] = 1
                changed["mi_components"] = changed["mi_components"][:1]
            if key == "non_dominant_small_component":
                changed["mi_components"][1]["area"] = 11
            if key == "small_component_near_myo_boundary":
                changed["mi_components"][1]["centroid_boundary_distance"] = 2
            self.assertFalse(evaluate_mi_fragment_rule(changed, voxel_area_mm2=1.0, parameters=params)["pseudo_flag"])

    def test_topology_signals_separate_local_outlier(self):
        row = self._row(myo_radial_ring_break_count=1, slice_to_slice_centroid_jump=2)
        neighbour = self._row(myo_radial_ring_break_count=0)
        result = decompose_myo_topology(row, previous=neighbour, following=neighbour)
        self.assertEqual(result["classification"], "single_slice_outlier")
        self.assertEqual(result["sub_signals"]["ring_gap_width"], 2)

    def test_validation_is_disjoint_and_non_promoting(self):
        row = self._row(simulated_candidates={"a1_noop": self._row(myo_radial_ring_break_count=0)})
        result = validate_advisory_rows([row], pilot_ids=["pilot"], validation_ids=["s1"], voxel_area_mm2=1.0)
        self.assertTrue(result["promotion_blocked"])
        self.assertFalse(result["label_mutation"])
        self.assertEqual(result["rows"][0]["candidate_before_after"]["a1_noop"],
                         {"before": True, "after": False})
        self.assertIn("mi_fraction_high_alone_is_insufficient", result["non_rules"])
        with self.assertRaises(ValueError):
            validate_advisory_rows([], pilot_ids=["same"], validation_ids=["same"], voxel_area_mm2=1.0)


if __name__ == "__main__":
    unittest.main()
