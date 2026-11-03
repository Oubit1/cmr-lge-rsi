import unittest

import numpy as np

from experiments.topology_shadow import evaluate_topology_shadow


class TopologyShadowTests(unittest.TestCase):
    def test_candidates_are_report_only_and_band_limited(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2] = mask[2:7, 6] = True
        mask[2, 2:7] = mask[6, 2:7] = True
        mask[4, 2] = False
        band = np.zeros_like(mask)
        band[1:8, 1:8] = True
        mi = np.zeros_like(mask)
        mi[3, 3] = mi[5, 5] = True
        original = mask.copy()

        report = evaluate_topology_shadow([mask], [mi], band_slices=[band])

        self.assertTrue(report["shadow_only"])
        self.assertFalse(report["contraindications"]["mask_writes"])
        self.assertTrue(np.array_equal(mask, original))
        self.assertEqual(set(report["candidates"]), {"ring_close_holes", "largest_merge", "neighbor_propagation", "noop"})
        self.assertEqual(report["candidates"]["noop"]["mi_fragmentation_outcome"], "persist-independently")

    def test_motion_is_accounted_for_and_dice_is_reported(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        mi = np.zeros_like(mask)
        reference = mask.copy()
        report = evaluate_topology_shadow([mask, mask], [mi, mi], myo_references=[reference, reference],
                                          mi_references=[mi, mi], through_plane_motion=[0.0, 2.0])
        result = report["candidates"]["noop"]
        self.assertEqual(result["myo_dice_after"], [1.0, 1.0])
        self.assertEqual(result["mi_dice_after"], [1.0, 1.0])
        self.assertEqual(result["through_plane_motion_accounted"], [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
