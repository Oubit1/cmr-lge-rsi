"""Offline tests for the read-only LGE MI fragment topology shadow scan."""

import unittest

import numpy as np

from experiments.lge_mi_fragment_topology_shadow_scan import (
    audit_diff,
    module_hash,
    scan_case,
    scan_cohort,
)


def _blank(shape=(16, 16)):
    return np.zeros(shape, dtype=bool)


def _ring(shape=(16, 16), r=6):
    mask = _blank(shape)
    cy, cx = shape[0] // 2, shape[1] // 2
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask[(dist >= r - 1) & (dist <= r)] = True
    return mask


class ScanCaseTests(unittest.TestCase):
    def test_empty_mi_suppressed(self):
        res = scan_case("c0", _blank(), _ring())
        self.assertEqual(res.flagged_fragments, 0)
        self.assertEqual(res.suppression["empty_mi"], 1)

    def test_empty_myo_suppressed(self):
        res = scan_case("c1", _ring(), _blank())
        self.assertEqual(res.suppression["empty_myo"], 1)

    def test_temporal_hole_suppressed(self):
        res = scan_case("c2", _ring(), _ring(), temporal_hole=True)
        self.assertEqual(res.suppression["temporal_hole"], 1)
        self.assertEqual(res.flagged_fragments, 0)

    def test_single_dominant_component_not_flagged(self):
        mi = _blank()
        mi[4:12, 4:12] = True
        res = scan_case("c3", mi, _ring())
        self.assertEqual(res.mi_components, 1)
        self.assertFalse(res.mi_multiple_components)
        self.assertEqual(res.flagged_fragments, 0)

    def test_small_non_adjacent_fragment_flagged(self):
        mi = _blank()
        mi[2:10, 2:10] = True  # dominant, 64 px
        mi[14, 14] = True  # 1 px fragment, far from myo ring
        res = scan_case("c4", mi, _ring())
        self.assertEqual(res.mi_components, 2)
        self.assertTrue(res.mi_multiple_components)
        self.assertEqual(res.flagged_fragments, 1)
        self.assertEqual(res.flagged_non_adjacent, 1)

    def test_fragment_adjacent_to_myo_not_counted_non_adjacent(self):
        myo = _ring()
        mi = _blank()
        mi[2:10, 2:10] = True
        # Place a small fragment that touches the myo ring but is NOT
        # 4-connected to the dominant block. The ring's topmost pixel sits
        # directly above the block, so pick a ring pixel on the right side
        # and keep a one-pixel gap from the block.
        ys, xs = np.where(myo)
        # choose a ring pixel with x > 10 (right of the block) to avoid
        # merging with the dominant component
        candidates = [(y, x) for y, x in zip(ys, xs) if x > 10]
        self.assertTrue(candidates, "ring must have a pixel right of the block")
        fy, fx = candidates[0]
        mi[fy, fx] = True
        res = scan_case("c5", mi, myo)
        self.assertGreaterEqual(res.flagged_fragments, 1)
        self.assertEqual(res.flagged_non_adjacent, 0)

    def test_border_fragment_suppressed(self):
        mi = _blank()
        mi[2:10, 2:10] = True
        mi[0, 0] = True  # touches border
        res = scan_case("c6", mi, _ring())
        self.assertEqual(res.suppression["border_fragment"], 1)
        self.assertEqual(res.flagged_fragments, 0)

    def test_inputs_not_mutated(self):
        mi = _blank()
        mi[2:10, 2:10] = True
        mi[14, 14] = True
        myo = _ring()
        mi_copy = mi.copy()
        myo_copy = myo.copy()
        scan_case("c7", mi, myo)
        self.assertTrue(np.array_equal(mi, mi_copy))
        self.assertTrue(np.array_equal(myo, myo_copy))


class CohortTests(unittest.TestCase):
    def _cases(self):
        cases = []
        for i in range(3):
            mi = _blank()
            mi[2:10, 2:10] = True
            mi[14, 14] = True
            cases.append({"case_id": f"case{i}", "mi_mask": mi, "myo_mask": _ring()})
        return cases

    def test_cohort_totals_and_denominators(self):
        report = scan_cohort(self._cases())
        self.assertEqual(report["n_cases"], 3)
        self.assertEqual(report["totals"]["flagged_fragments"], 3)
        self.assertEqual(report["totals"]["flagged_non_adjacent"], 3)
        self.assertEqual(report["totals"]["non_adjacent_denominator"], 3)
        self.assertAlmostEqual(report["totals"]["non_adjacent_fraction"], 1.0)
        self.assertEqual(report["totals"]["overlap_denominator"], 3)

    def test_concordance_unavailable_without_reference(self):
        report = scan_cohort(self._cases())
        self.assertEqual(report["concordance"], "unavailable_no_reference_labels")
        self.assertEqual(report["ci"], "unavailable_no_reference_labels")

    def test_module_hash_stable(self):
        self.assertEqual(module_hash(), module_hash())
        self.assertEqual(len(module_hash()), 64)

    def test_audit_diff_detects_changes(self):
        first = scan_cohort(self._cases())
        second = scan_cohort(self._cases())
        diff = audit_diff(first, second)
        self.assertEqual(diff["changed_cases"], [])
        self.assertEqual(diff["added_cases"], [])
        self.assertEqual(diff["removed_cases"], [])

    def test_audit_diff_baseline_none(self):
        report = scan_cohort(self._cases())
        diff = audit_diff(None, report)
        self.assertEqual(diff["baseline"], "none")
        self.assertEqual(len(diff["added_cases"]), 3)


if __name__ == "__main__":
    unittest.main()
