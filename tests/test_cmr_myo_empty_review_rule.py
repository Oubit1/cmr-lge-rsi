'''Offline tests for R-CMR-MYO-EMPTY-REVIEW-01 draft rule.'''

import json
import unittest

from experiments.cmr_myo_empty_review_rule import (
    MIN_CASE_SUPPORTING_FLAGS,
    MIN_SLICE_SUPPORTING_FLAGS,
    PRIMARY_FLAG,
    RULE_CACHE_HASH,
    RULE_ID,
    RULE_SPEC,
    RULE_VERSION,
    evaluate_case,
    evaluate_slice,
    rule_cache_hash,
    rule_trace,
)


def _slice(
    *,
    slice_id: str = 's1',
    modality: str = 'cmr',
    primary: bool = True,
    supporting: tuple[str, ...] = ('support_a',),
    paired: bool = True,
) -> dict:
    return {
        'slice_id': slice_id,
        'modality': modality,
        'flags': {PRIMARY_FLAG: primary},
        'supporting_flags': list(supporting),
        'paired_lge_myocardium_present': paired,
    }


def _case(
    *,
    case_id: str = 'case-1',
    positive_slices: int = 0,
    negative_slices: int = 0,
    case_support: tuple[str, ...] = ('case_a', 'case_b'),
) -> dict:
    slices = []
    for index in range(positive_slices):
        slices.append(
            _slice(
                slice_id=f'p{index}',
                primary=True,
                supporting=('support_a',),
                paired=True,
            )
        )
    for index in range(negative_slices):
        slices.append(
            _slice(
                slice_id=f'n{index}',
                primary=False,
                supporting=(),
                paired=False,
            )
        )
    return {
        'case_id': case_id,
        'slices': slices,
        'case_supporting_flags': list(case_support),
    }


class CmrMyoEmptyReviewRuleTests(unittest.TestCase):
    def test_default_disabled_and_review_only(self):
        case = _case(positive_slices=1, case_support=('case_a', 'case_b'))
        decision = evaluate_case(case)
        self.assertFalse(decision.enabled)
        self.assertTrue(decision.review_only)
        self.assertFalse(decision.escalated)
        self.assertEqual(('default_disabled',), decision.reasons)
        self.assertEqual(RULE_ID, decision.rule_id)
        self.assertEqual(RULE_VERSION, decision.rule_version)
        self.assertEqual(RULE_CACHE_HASH, decision.rule_cache_hash)

    def test_slice_positive_requires_all_conditions(self):
        self.assertTrue(evaluate_slice(_slice()).positive)

        negative_cases = (
            _slice(modality='lge'),
            _slice(primary=False),
            _slice(supporting=()),
            _slice(paired=False),
        )
        for item in negative_cases:
            with self.subTest(item=item):
                self.assertFalse(evaluate_slice(item).positive)

    def test_case_escalation_requires_positive_slice_and_case_support(self):
        escalated = evaluate_case(
            _case(positive_slices=1, case_support=('case_a', 'case_b')),
            enabled=True,
        )
        self.assertTrue(escalated.escalated)

        no_positive_slice = evaluate_case(
            _case(positive_slices=0, case_support=('case_a', 'case_b')),
            enabled=True,
        )
        self.assertFalse(no_positive_slice.escalated)
        self.assertIn('no_positive_slice', no_positive_slice.reasons)

        insufficient_support = evaluate_case(
            _case(positive_slices=1, case_support=('case_a',)),
            enabled=True,
        )
        self.assertFalse(insufficient_support.escalated)
        self.assertIn('insufficient_case_supporting_flags', insufficient_support.reasons)

    def test_recomputable_on_30_plus_10_samples(self):
        positives = [
            _case(case_id=f'pos-{index}', positive_slices=1, case_support=('case_a', 'case_b'))
            for index in range(30)
        ]
        negatives = [
            _case(case_id=f'neg-{index}', positive_slices=0, case_support=('case_a', 'case_b'))
            for index in range(10)
        ]

        positive_decisions = [evaluate_case(case, enabled=True) for case in positives]
        negative_decisions = [evaluate_case(case, enabled=True) for case in negatives]

        self.assertEqual(30, sum(1 for decision in positive_decisions if decision.escalated))
        self.assertEqual(0, sum(1 for decision in negative_decisions if decision.escalated))
        self.assertEqual(30, sum(decision.positive_slice_count for decision in positive_decisions))
        self.assertEqual(0, sum(decision.positive_slice_count for decision in negative_decisions))

    def test_no_new_numeric_thresholds_and_no_production_writes(self):
        self.assertEqual({}, RULE_SPEC['numeric_thresholds'])
        self.assertEqual(1, MIN_SLICE_SUPPORTING_FLAGS)
        self.assertEqual(2, MIN_CASE_SUPPORTING_FLAGS)

        trace = rule_trace()
        self.assertFalse(trace['production_writes'])
        self.assertEqual('review_only', trace['mode'])
        self.assertFalse(trace['default_enabled'])

    def test_rule_version_and_cache_hash_are_traceable(self):
        self.assertEqual(RULE_CACHE_HASH, rule_cache_hash())
        trace = rule_trace()
        self.assertEqual(RULE_ID, trace['rule_id'])
        self.assertEqual(RULE_VERSION, trace['rule_version'])
        self.assertEqual(RULE_CACHE_HASH, trace['rule_cache_hash'])

    def test_evaluation_does_not_write_input_payload(self):
        case = _case(positive_slices=1, case_support=('case_a', 'case_b'))
        before = json.dumps(case, sort_keys=True)
        evaluate_case(case, enabled=True)
        after = json.dumps(case, sort_keys=True)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
