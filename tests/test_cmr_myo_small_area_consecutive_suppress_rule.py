import unittest

from experiments.cmr_myo_small_area_consecutive_suppress_rule import (
    SliceObservation,
    evaluate,
    suppression_coverage,
)


def _obs(i, myo, cav, **kw):
    return SliceObservation(slice_index=i, myo_area=myo, cavity_area=cav, **kw)


class TestConsecutiveSmallAreaRule(unittest.TestCase):
    def test_two_consecutive_small_slices_are_candidate(self):
        obs = [_obs(0, 10, 100), _obs(1, 12, 100), _obs(2, 50, 100)]
        d = evaluate(obs)
        self.assertTrue(d.candidate)
        self.assertFalse(d.suppressed)
        self.assertEqual(d.route, "small_area_review_candidate")
        self.assertEqual(d.consecutive_run, 2)

    def test_single_small_slice_not_candidate(self):
        obs = [_obs(0, 10, 100), _obs(1, 50, 100)]
        d = evaluate(obs)
        self.assertFalse(d.candidate)
        self.assertEqual(d.route, "no_action")

    def test_empty_myo_suppresses(self):
        obs = [_obs(0, 10, 100), _obs(1, 12, 100, myo_empty=True)]
        d = evaluate(obs)
        self.assertTrue(d.suppressed)
        self.assertEqual(d.route, "stack_level_review")
        self.assertIn("myo_empty", d.reasons)

    def test_border_touch_suppresses(self):
        obs = [_obs(0, 10, 100), _obs(1, 12, 100, touches_border=True)]
        d = evaluate(obs)
        self.assertTrue(d.suppressed)
        self.assertIn("myo_touches_border", d.reasons)

    def test_temporal_hole_suppresses(self):
        obs = [_obs(0, 10, 100), _obs(1, 12, 100, temporal_hole=True)]
        d = evaluate(obs)
        self.assertTrue(d.suppressed)
        self.assertIn("temporal_hole", d.reasons)

    def test_centroid_drift_suppresses(self):
        obs = [
            _obs(0, 10, 100, centroid=(0.0, 0.0)),
            _obs(1, 12, 100, centroid=(20.0, 0.0)),
        ]
        d = evaluate(obs, centroid_drift_threshold=5.0)
        self.assertTrue(d.suppressed)
        self.assertTrue(any(r.startswith("centroid_drift=") for r in d.reasons))

    def test_suppression_coverage_metric(self):
        decisions = [
            evaluate([_obs(0, 10, 100), _obs(1, 12, 100)]),
            evaluate([_obs(0, 10, 100), _obs(1, 12, 100, myo_empty=True)]),
        ]
        cov = suppression_coverage(decisions, [False, True])
        self.assertEqual(cov, 1.0)

    def test_no_writes_side_effect(self):
        # Rule is pure: repeated evaluation yields identical decisions.
        obs = [_obs(0, 10, 100), _obs(1, 12, 100)]
        self.assertEqual(evaluate(obs), evaluate(obs))


if __name__ == "__main__":
    unittest.main()
