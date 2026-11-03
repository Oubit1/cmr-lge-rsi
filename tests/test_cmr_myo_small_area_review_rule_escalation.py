"""Offline unit tests for R-CMR-MYO-SMALL-AREA-REVIEW-01 (review-only draft)."""
import copy
import unittest

from experiments.cmr_myo_small_area_review_rule_escalation import (
    CASE_SUPPORT_MIN,
    DEFAULT_ENABLED,
    PAIRED_LGE_SUPPORT_KEY,
    PRODUCTION_WRITE_CHANNELS,
    RULE_ID,
    SLICE_POSITIVE_MIN,
    SMALL_AREA_FLAG,
    SUPPORTING_FLAGS,
    assert_review_only,
    evaluate_case,
    filter_escalation_candidates,
    rule_metadata,
    slice_positive,
    supporting_flags_on_slice,
)


def _slice(slice_id, modality="cmr", flag=True, support=(), extra=None):
    record = {"slice_id": slice_id, "modality": modality, SMALL_AREA_FLAG: flag}
    for name in support:
        record[name] = True
    if extra:
        record.update(extra)
    return record


def _case(case_id, slices, paired=True):
    return {
        "case_id": case_id,
        "slices": list(slices),
        PAIRED_LGE_SUPPORT_KEY: paired,
    }


class RuleDraftDefaultOffTests(unittest.TestCase):
    def test_rule_is_default_off_and_review_only(self):
        meta = rule_metadata()
        self.assertEqual(meta["rule_id"], RULE_ID)
        self.assertEqual(RULE_ID, "R-CMR-MYO-SMALL-AREA-REVIEW-01")
        self.assertFalse(DEFAULT_ENABLED)
        self.assertFalse(meta["default_enabled"])
        self.assertEqual(meta["mode"], "review_only")
        self.assertTrue(meta["advisory_only"])
        self.assertFalse(meta["mutates_production"])
        self.assertEqual(meta["new_numeric_measure_thresholds"], 0)

    def test_no_production_write_channels(self):
        self.assertEqual(PRODUCTION_WRITE_CHANNELS, ())
        self.assertEqual(rule_metadata()["production_write_channels"], [])
        assert_review_only()  # must not raise

    def test_disabled_rule_never_escalates(self):
        case = _case(
            "c-off",
            [
                _slice("s0", support=("myo_touches_border",)),
                _slice("s1", support=("myo_centroid_drift",)),
            ],
        )
        review = evaluate_case(case)
        self.assertFalse(review.enabled)
        self.assertFalse(review.escalated)
        self.assertEqual(review.reason, "rule_disabled")
        self.assertEqual(review.positive_slices, ())
        self.assertEqual(filter_escalation_candidates([case]), ())

    def test_structural_cardinalities_match_rule_text(self):
        self.assertEqual(SLICE_POSITIVE_MIN, 1)
        self.assertEqual(CASE_SUPPORT_MIN, 2)
        self.assertEqual(rule_metadata()["slice_positive_min_support"], 1)
        self.assertEqual(rule_metadata()["case_support_min"], 2)


class SupportingFlagScopeTests(unittest.TestCase):
    def test_supporting_flags_are_limited_to_three(self):
        self.assertEqual(
            set(SUPPORTING_FLAGS),
            {"myo_touches_border", "myo_excess_components", "myo_centroid_drift"},
        )
        self.assertEqual(
            supporting_flags_on_slice(
                _slice("s0", support=("myo_touches_border",), extra={"some_flag": True})
            ),
            frozenset({"myo_touches_border"}),
        )

    def test_unknown_flags_do_not_make_a_slice_positive(self):
        record = _slice("s0", support=(), extra={"some_other_flag": True})
        self.assertEqual(supporting_flags_on_slice(record), frozenset())
        self.assertFalse(slice_positive(record))


class SlicePositivityTests(unittest.TestCase):
    def test_cmr_flagged_slice_with_support_is_positive(self):
        self.assertTrue(
            slice_positive(_slice("s0", support=("myo_excess_components",)))
        )

    def test_non_cmr_slice_is_never_positive(self):
        record = _slice("s0", modality="lge", support=("myo_centroid_drift",))
        self.assertFalse(slice_positive(record))

    def test_small_area_flag_must_be_true(self):
        self.assertFalse(slice_positive(_slice("s0", flag=False, support=("myo_touches_border",))))

    def test_supporting_flag_required_on_the_same_slice(self):
        self.assertFalse(slice_positive(_slice("s0", support=())))


class CaseEscalationTests(unittest.TestCase):
    def test_escalates_on_one_positive_slice_and_two_case_flags(self):
        review = evaluate_case(
            _case(
                "c-up",
                [
                    _slice("s0", support=("myo_touches_border",)),
                    _slice("s1", support=("myo_centroid_drift",)),
                ],
            ),
            enabled=True,
        )
        self.assertTrue(review.escalated)
        self.assertEqual(review.reason, "escalation_candidate_review_only")
        self.assertEqual(review.positive_slices, ("s0", "s1"))
        self.assertEqual(review.case_support, ("myo_centroid_drift", "myo_touches_border"))

    def test_without_paired_lge_myo_support_no_escalation(self):
        review = evaluate_case(
            _case(
                "c-unpaired",
                [
                    _slice("s0", support=("myo_touches_border",)),
                    _slice("s1", support=("myo_centroid_drift",)),
                ],
                paired=False,
            ),
            enabled=True,
        )
        self.assertFalse(review.escalated)
        self.assertEqual(review.reason, "no_paired_lge_myo_support")
        self.assertEqual(
            filter_escalation_candidates(
                [
                    _case(
                        "c-unpaired",
                        [_slice("s0", support=("myo_touches_border",))],
                        paired=False,
                    )
                ],
                enabled=True,
            ),
            (),
        )

    def test_single_distinct_case_level_flag_is_not_enough(self):
        review = evaluate_case(
            _case(
                "c-one-flag",
                [
                    _slice("s0", support=("myo_touches_border",)),
                    _slice("s1", support=("myo_touches_border",)),
                ],
            ),
            enabled=True,
        )
        self.assertFalse(review.escalated)
        self.assertEqual(review.reason, "insufficient_case_level_support")
        self.assertEqual(review.case_support, ("myo_touches_border",))

    def test_at_least_one_positive_slice_is_required(self):
        review = evaluate_case(
            _case(
                "c-no-positive",
                [
                    _slice("s0", support=()),
                    _slice("s1", support=()),
                ],
            ),
            enabled=True,
        )
        self.assertFalse(review.escalated)
        self.assertEqual(review.reason, "no_positive_slice")
        self.assertEqual(review.positive_slices, ())

    def test_non_cmr_support_does_not_feed_case_level_support(self):
        review = evaluate_case(
            _case(
                "c-lge-support",
                [
                    _slice("s0", support=("myo_touches_border",)),
                    _slice("s1", modality="lge", support=("myo_centroid_drift",)),
                ],
            ),
            enabled=True,
        )
        self.assertFalse(review.escalated)
        self.assertEqual(review.reason, "insufficient_case_level_support")
        self.assertEqual(review.case_support, ("myo_touches_border",))

    def test_batch_filter_returns_only_escalations(self):
        escalated = _case(
            "c-yes",
            [
                _slice("s0", support=("myo_touches_border",)),
                _slice("s1", support=("myo_excess_components",)),
            ],
        )
        quiet = _case("c-no", [_slice("s0", support=("myo_touches_border",))])
        snapshot = copy.deepcopy([escalated, quiet])

        reviews = filter_escalation_candidates([escalated, quiet], enabled=True)

        self.assertEqual([r.case_id for r in reviews], ["c-yes"])
        self.assertEqual(reviews[0].as_dict()["escalated"], True)
        self.assertEqual([escalated, quiet], snapshot)

    def test_inputs_are_not_mutated(self):
        case = _case(
            "c-immutable",
            [
                _slice("s0", support=("myo_touches_border",)),
                _slice("s1", support=("myo_centroid_drift",)),
            ],
        )
        snapshot = copy.deepcopy(case)
        evaluate_case(case, enabled=True)
        self.assertEqual(case, snapshot)


if __name__ == "__main__":
    unittest.main()
