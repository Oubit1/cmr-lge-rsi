'''Offline tests for R-CMR-MYO-SMALL-AREA-REVIEW-01.'''

import unittest

from experiments.cmr_myo_small_area_review_rule import (
    NUMERIC_QC_THRESHOLDS,
    RULE_ID,
    CaseReplaySample,
    RuleConfig,
    SliceReplaySample,
    decide_case,
    decide_slice,
    record_replay_metrics,
)


class DefaultOffTests(unittest.TestCase):
    def test_rule_id(self):
        self.assertEqual(RULE_ID, 'R-CMR-MYO-SMALL-AREA-REVIEW-01')

    def test_default_disabled_and_review_only(self):
        config = RuleConfig()
        self.assertFalse(config.enabled)
        self.assertTrue(config.review_only)

    def test_disabled_rule_returns_non_positive_decision(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': True,
                'cmr_myo_small_area': True,
                'support_flags': {'myo_small_area_shadow': True},
            }
        )
        self.assertFalse(decision.positive)
        self.assertIn('rule_disabled', decision.reasons)
        self.assertFalse(decision.write_allowed)
        self.assertFalse(decision.label_modification_allowed)

    def test_review_only_guard(self):
        with self.assertRaises(ValueError):
            RuleConfig(review_only=False)


class SliceRuleTests(unittest.TestCase):
    def setUp(self):
        self.config = RuleConfig(enabled=True)

    def test_positive_when_cmr_small_area_and_one_support_flag(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': True,
                'cmr_myo_small_area': True,
                'support_flags': {'myo_small_area_shadow': True},
            },
            self.config,
        )
        self.assertTrue(decision.positive)
        self.assertEqual(decision.support_flags, ('myo_small_area_shadow',))
        self.assertIn('slice_positive', decision.reasons)
        self.assertFalse(decision.write_allowed)
        self.assertFalse(decision.label_modification_allowed)

    def test_negative_without_cmr(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': False,
                'cmr_myo_small_area': True,
                'support_flags': {'myo_small_area_shadow': True},
            },
            self.config,
        )
        self.assertFalse(decision.positive)
        self.assertIn('not_cmr', decision.reasons)

    def test_negative_without_small_area(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': True,
                'cmr_myo_small_area': False,
                'support_flags': {'myo_small_area_shadow': True},
            },
            self.config,
        )
        self.assertFalse(decision.positive)
        self.assertIn('cmr_myo_small_area_not_true', decision.reasons)

    def test_negative_without_support_flag(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': True,
                'cmr_myo_small_area': True,
                'support_flags': {'myo_small_area_shadow': False},
            },
            self.config,
        )
        self.assertFalse(decision.positive)
        self.assertIn('no_slice_support_flag', decision.reasons)

    def test_top_level_support_flag_is_accepted(self):
        decision = decide_slice(
            {
                'slice_id': 's1',
                'cmr': True,
                'cmr_myo_small_area': True,
                'myo_small_area_support': True,
            },
            self.config,
        )
        self.assertTrue(decision.positive)
        self.assertIn('myo_small_area_support', decision.support_flags)


class CaseRuleTests(unittest.TestCase):
    def setUp(self):
        self.config = RuleConfig(enabled=True)

    def test_escalate_with_one_positive_slice_and_two_case_flags(self):
        decision = decide_case(
            {'case_id': 'c1', 'support_flags': {'a': True, 'b': True}},
            [
                {
                    'slice_id': 's1',
                    'cmr': True,
                    'cmr_myo_small_area': True,
                    'support_flags': {'x': True},
                }
            ],
            self.config,
        )
        self.assertTrue(decision.escalate)
        self.assertEqual(decision.positive_slice_ids, ('s1',))
        self.assertEqual(len(decision.case_support_flags), 2)
        self.assertIn('case_escalate_review', decision.reasons)
        self.assertFalse(decision.write_allowed)
        self.assertFalse(decision.label_modification_allowed)

    def test_no_escalate_without_positive_slice(self):
        decision = decide_case(
            {'case_id': 'c1', 'support_flags': {'a': True, 'b': True}},
            [
                {
                    'slice_id': 's1',
                    'cmr': True,
                    'cmr_myo_small_area': False,
                    'support_flags': {'x': True},
                }
            ],
            self.config,
        )
        self.assertFalse(decision.escalate)
        self.assertIn('no_positive_slice', decision.reasons)

    def test_no_escalate_with_one_case_flag(self):
        decision = decide_case(
            {'case_id': 'c1', 'support_flags': {'a': True, 'b': False}},
            [
                {
                    'slice_id': 's1',
                    'cmr': True,
                    'cmr_myo_small_area': True,
                    'support_flags': {'x': True},
                }
            ],
            self.config,
        )
        self.assertFalse(decision.escalate)
        self.assertIn('case_support_flags_lt_2', decision.reasons)

    def test_no_escalate_with_zero_case_flags(self):
        decision = decide_case(
            {'case_id': 'c1', 'support_flags': {}},
            [
                {
                    'slice_id': 's1',
                    'cmr': True,
                    'cmr_myo_small_area': True,
                    'support_flags': {'x': True},
                }
            ],
            self.config,
        )
        self.assertFalse(decision.escalate)
        self.assertIn('case_support_flags_lt_2', decision.reasons)


class ReplayMetricsTests(unittest.TestCase):
    def test_records_slice_and_case_precision(self):
        config = RuleConfig(enabled=True)
        slice_samples = [
            SliceReplaySample(
                {
                    'slice_id': 's1',
                    'cmr': True,
                    'cmr_myo_small_area': True,
                    'support_flags': {'x': True},
                },
                True,
            ),
            SliceReplaySample(
                {
                    'slice_id': 's2',
                    'cmr': True,
                    'cmr_myo_small_area': True,
                    'support_flags': {'x': True},
                },
                False,
            ),
            SliceReplaySample(
                {
                    'slice_id': 's3',
                    'cmr': True,
                    'cmr_myo_small_area': False,
                    'support_flags': {'x': True},
                },
                True,
            ),
        ]
        case_samples = [
            CaseReplaySample(
                {'case_id': 'c1', 'support_flags': {'a': True, 'b': True}},
                [
                    {
                        'slice_id': 's1',
                        'cmr': True,
                        'cmr_myo_small_area': True,
                        'support_flags': {'x': True},
                    }
                ],
                True,
            ),
            CaseReplaySample(
                {'case_id': 'c2', 'support_flags': {'a': True, 'b': True}},
                [
                    {
                        'slice_id': 's2',
                        'cmr': True,
                        'cmr_myo_small_area': True,
                        'support_flags': {'x': True},
                    }
                ],
                False,
            ),
        ]

        report = record_replay_metrics(slice_samples, case_samples, config)

        self.assertEqual(report.slice.true_positive, 1)
        self.assertEqual(report.slice.false_positive, 1)
        self.assertEqual(report.slice.precision, 0.5)
        self.assertEqual(report.case.true_positive, 1)
        self.assertEqual(report.case.false_positive, 1)
        self.assertEqual(report.case.precision, 0.5)

    def test_zero_numeric_qc_thresholds(self):
        self.assertEqual(NUMERIC_QC_THRESHOLDS, ())


if __name__ == '__main__':
    unittest.main()
