"""
Unit tests for lge_mi_fragmentation_topology_review.py

Runs offline without external data.
"""

import unittest

from experiments.lge_mi_fragmentation_topology_review import (
    CLOSING_RADIUS_MM,
    MERGE_DISTANCE_MM,
    MINIMUM_COMPONENT_SIZE_VOXELS,
    Component,
    SliceTopology,
    apply_bounded_variant,
    check_escalation_required,
    check_paired_cmr_lge_topology,
    invariant_single_component_requires_per_slice_review,
    run_fragmentation_review,
    should_route_to_fragmentation_check,
)


class TestRoutingPredicates(unittest.TestCase):
    def test_route_when_multiple_and_topology(self):
        flags = ["mi_multiple_components", "myo_topology"]
        self.assertTrue(should_route_to_fragmentation_check(flags))

    def test_route_when_multiple_and_fraction(self):
        flags = ["mi_multiple_components", "mi_fraction_high"]
        self.assertTrue(should_route_to_fragmentation_check(flags))

    def test_route_when_multiple_and_both(self):
        flags = ["mi_multiple_components", "myo_topology", "mi_fraction_high"]
        self.assertTrue(should_route_to_fragmentation_check(flags))

    def test_no_route_when_only_multiple(self):
        flags = ["mi_multiple_components"]
        self.assertFalse(should_route_to_fragmentation_check(flags))

    def test_no_route_when_no_multiple(self):
        flags = ["myo_topology", "mi_fraction_high"]
        self.assertFalse(should_route_to_fragmentation_check(flags))

    def test_no_route_empty_flags(self):
        self.assertFalse(should_route_to_fragmentation_check([]))


class TestInvariant(unittest.TestCase):
    def test_single_component_without_review_fails(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        self.assertFalse(
            invariant_single_component_requires_per_slice_review([comp], per_slice_reviewed=False)
        )

    def test_single_component_with_review_passes(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        self.assertTrue(
            invariant_single_component_requires_per_slice_review([comp], per_slice_reviewed=True)
        )

    def test_multiple_components_no_review_passes(self):
        comp1 = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        comp2 = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        self.assertTrue(
            invariant_single_component_requires_per_slice_review([comp1, comp2], per_slice_reviewed=False)
        )

    def test_empty_components_passes(self):
        self.assertTrue(
            invariant_single_component_requires_per_slice_review([], per_slice_reviewed=False)
        )


class TestBoundedVariant(unittest.TestCase):
    def test_small_components_filtered(self):
        comp_small = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=5, slice_indices=(0,))
        comp_large = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        result = apply_bounded_variant([comp_small, comp_large])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, 2)

    def test_all_large_components_preserved(self):
        comp1 = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=15, slice_indices=(0,))
        comp2 = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        result = apply_bounded_variant([comp1, comp2])
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        self.assertEqual(apply_bounded_variant([]), [])

    def test_pre_declared_params_present(self):
        self.assertEqual(CLOSING_RADIUS_MM, 2.0)
        self.assertEqual(MERGE_DISTANCE_MM, 5.0)
        self.assertEqual(MINIMUM_COMPONENT_SIZE_VOXELS, 10)


class TestEscalation(unittest.TestCase):
    def test_no_escalation_single_component(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        required, reason = check_escalation_required([comp])
        self.assertFalse(required)
        self.assertIsNone(reason)

    def test_escalation_multiple_components(self):
        comp1 = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        comp2 = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        required, reason = check_escalation_required([comp1, comp2])
        self.assertTrue(required)
        self.assertEqual(reason, "multiple_components_persist_after_bounded_variant")

    def test_escalation_multifocal_uncertain(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        required, reason = check_escalation_required([comp], true_multifocal_cannot_be_excluded=True)
        self.assertTrue(required)
        self.assertEqual(reason, "true_multifocal_scar_cannot_be_excluded")

    def test_escalation_both_conditions(self):
        comp1 = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        comp2 = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        required, reason = check_escalation_required([comp1, comp2], true_multifocal_cannot_be_excluded=True)
        self.assertTrue(required)
        # Multiple components takes precedence in implementation
        self.assertEqual(reason, "multiple_components_persist_after_bounded_variant")


class TestPairedCMRLGE(unittest.TestCase):
    def test_no_paired_data(self):
        lge = [SliceTopology(slice_index=0, num_components=1, component_labels=(1,))]
        result = check_paired_cmr_lge_topology(lge)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].overlap_with_cmr)

    def test_paired_data_overlap(self):
        lge = [SliceTopology(slice_index=0, num_components=2, component_labels=(1, 2))]
        cmr = [SliceTopology(slice_index=0, num_components=2, component_labels=(1, 2))]
        result = check_paired_cmr_lge_topology(lge, cmr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].overlap_with_cmr, 1.0)

    def test_paired_data_partial_overlap(self):
        lge = [SliceTopology(slice_index=0, num_components=2, component_labels=(1, 2))]
        cmr = [SliceTopology(slice_index=0, num_components=2, component_labels=(2, 3))]
        result = check_paired_cmr_lge_topology(lge, cmr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].overlap_with_cmr, 1.0 / 3.0)

    def test_paired_data_no_matching_slice(self):
        lge = [SliceTopology(slice_index=0, num_components=1, component_labels=(1,))]
        cmr = [SliceTopology(slice_index=1, num_components=1, component_labels=(1,))]
        result = check_paired_cmr_lge_topology(lge, cmr)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].overlap_with_cmr)

    def test_multiple_slices(self):
        lge = [
            SliceTopology(slice_index=0, num_components=1, component_labels=(1,)),
            SliceTopology(slice_index=1, num_components=2, component_labels=(2, 3)),
        ]
        cmr = [
            SliceTopology(slice_index=0, num_components=1, component_labels=(1,)),
            SliceTopology(slice_index=1, num_components=1, component_labels=(2,)),
        ]
        result = check_paired_cmr_lge_topology(lge, cmr)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].overlap_with_cmr, 1.0)
        self.assertEqual(result[1].overlap_with_cmr, 0.5)


class TestRunFragmentationReview(unittest.TestCase):
    def test_not_routed(self):
        result = run_fragmentation_review(flags=["myo_topology"], components=[])
        self.assertFalse(result.routed)
        self.assertFalse(result.escalation_required)

    def test_routed_no_escalation(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        result = run_fragmentation_review(
            flags=["mi_multiple_components", "myo_topology"],
            components=[comp],
            per_slice_reviewed=True,
        )
        self.assertTrue(result.routed)
        self.assertTrue(result.invariant_held)
        self.assertTrue(result.tool_guidance_applied)
        self.assertFalse(result.escalation_required)
        self.assertFalse(result.paired_cmr_checked)

    def test_routed_with_escalation(self):
        comp1 = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        comp2 = Component(label=2, centroid_xyz=(10.0, 0.0, 0.0), volume_voxels=20, slice_indices=(1,))
        result = run_fragmentation_review(
            flags=["mi_multiple_components", "mi_fraction_high"],
            components=[comp1, comp2],
            per_slice_reviewed=True,
        )
        self.assertTrue(result.routed)
        self.assertTrue(result.escalation_required)
        self.assertEqual(result.escalation_reason, "multiple_components_persist_after_bounded_variant")

    def test_routed_with_paired_cmr(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        cmr = [SliceTopology(slice_index=0, num_components=1, component_labels=(1,))]
        result = run_fragmentation_review(
            flags=["mi_multiple_components", "myo_topology"],
            components=[comp],
            per_slice_reviewed=True,
            cmr_slice_topologies=cmr,
        )
        self.assertTrue(result.paired_cmr_checked)
        self.assertEqual(len(result.slice_topologies), 1)

    def test_invariant_violation(self):
        comp = Component(label=1, centroid_xyz=(0.0, 0.0, 0.0), volume_voxels=20, slice_indices=(0,))
        result = run_fragmentation_review(
            flags=["mi_multiple_components", "myo_topology"],
            components=[comp],
            per_slice_reviewed=False,
        )
        self.assertFalse(result.invariant_held)


if __name__ == "__main__":
    unittest.main()
