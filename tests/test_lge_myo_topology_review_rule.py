import unittest

from experiments.lge_myo_topology_review_rule import (
    DEFAULT_ENABLED,
    LABELS_CHANGED,
    PRODUCTION_THRESHOLDS_CHANGED,
    REQUIRED_CASE_SUPPORT_FLAG_COUNT,
    REQUIRED_POSITIVE_SLICE_COUNT,
    REQUIRED_SLICE_SUPPORT_FLAG_COUNT,
    REVIEW_ONLY,
    RULE_ID,
    audit_positive_consistency,
    evaluate_case,
    evaluate_slice,
    static_validate,
)


def _slice(slice_id, lge, myo_topology, support_flags=()):
    return {
        'slice_id': slice_id,
        'lge': lge,
        'myo_topology': myo_topology,
        'support_flags': list(support_flags),
    }


class StaticDraftValidationTest(unittest.TestCase):
    def test_draft_is_default_off_and_has_no_production_changes(self):
        self.assertEqual(RULE_ID, 'R-LGE-MYO-TOPOLOGY-REVIEW-01')
        self.assertFalse(DEFAULT_ENABLED)
        self.assertTrue(REVIEW_ONLY)
        self.assertEqual(PRODUCTION_THRESHOLDS_CHANGED, 0)
        self.assertEqual(LABELS_CHANGED, 0)
        report = static_validate()
        self.assertTrue(report['valid'], report['violations'])
        self.assertFalse(report['default_enabled'])
        self.assertTrue(report['review_only'])
        self.assertEqual(report['production_thresholds_changed'], 0)
        self.assertEqual(report['labels_changed'], 0)

    def test_rule_is_inert_while_disabled(self):
        case = {
            'slices': [_slice('s1', True, True, ['myo_topology_support'])],
            'support_flags': ['case_a', 'case_b'],
        }
        result = evaluate_case(case)
        self.assertFalse(result['enabled'])
        self.assertFalse(result['review_required'])


class SlicePositiveTest(unittest.TestCase):
    def test_slice_positive_requires_lge_topology_and_same_slice_support(self):
        positive = evaluate_slice(_slice('s1', True, True, ['myo_topology_support']))
        self.assertTrue(positive['slice_positive'])
        self.assertEqual(
            positive['support_flag_count'], REQUIRED_SLICE_SUPPORT_FLAG_COUNT
        )

        self.assertFalse(
            evaluate_slice(_slice('s2', False, True, ['myo_topology_support']))[
                'slice_positive'
            ]
        )
        self.assertFalse(
            evaluate_slice(_slice('s3', True, False, ['myo_topology_support']))[
                'slice_positive'
            ]
        )
        self.assertFalse(evaluate_slice(_slice('s4', True, True, []))['slice_positive'])

    def test_support_flag_can_be_present_as_true_scalar_key(self):
        record = {
            'slice_id': 's1',
            'lge': True,
            'myo_topology': True,
            'support_border_contact': True,
        }
        result = evaluate_slice(record)
        self.assertTrue(result['slice_positive'])
        self.assertIn('support_border_contact', result['support_flags'])


class CaseEscalationTest(unittest.TestCase):
    def test_case_requires_one_positive_slice_and_two_case_level_support_flags(self):
        eligible = {
            'slices': [_slice('s1', True, True, ['myo_topology_support'])],
            'support_flags': ['case_a', 'case_b'],
        }
        result = evaluate_case(eligible, enabled=True)
        self.assertTrue(result['review_required'])
        self.assertEqual(result['positive_slice_count'], REQUIRED_POSITIVE_SLICE_COUNT)
        self.assertEqual(
            result['case_support_flag_count'], REQUIRED_CASE_SUPPORT_FLAG_COUNT
        )

    def test_case_without_enough_case_level_support_flags_is_not_escalated(self):
        case = {
            'slices': [_slice('s1', True, True, ['myo_topology_support'])],
            'support_flags': ['case_a'],
        }
        result = evaluate_case(case, enabled=True)
        self.assertFalse(result['review_required'])
        self.assertEqual(result['case_support_flag_count'], 1)

    def test_case_without_positive_slice_is_not_escalated(self):
        case = {
            'slices': [_slice('s1', True, False, ['myo_topology_support'])],
            'support_flags': ['case_a', 'case_b'],
        }
        result = evaluate_case(case, enabled=True)
        self.assertFalse(result['review_required'])
        self.assertEqual(result['positive_slice_count'], 0)

    def test_case_level_support_flags_can_be_boolean_keys(self):
        case = {
            'slices': [_slice('s1', True, True, ['myo_topology_support'])],
            'myo_topology_support': True,
            'border_contact_support': True,
        }
        result = evaluate_case(case, enabled=True)
        self.assertTrue(result['review_required'])
        self.assertEqual(result['case_support_flag_count'], 2)


class AuditConsistencyTest(unittest.TestCase):
    def test_audit_positive_set_consistency_is_at_least_eighty_percent(self):
        cases = [
            {
                'slices': [_slice('s1', True, True, ['myo_topology_support'])],
                'support_flags': ['case_a', 'case_b'],
            },
            {
                'slices': [
                    _slice('s1', True, True, ['myo_topology_support']),
                    _slice('s2', True, True, ['border_contact_support']),
                ],
                'support_flags': ['case_a', 'case_b'],
            },
            {
                'slices': [_slice('s1', True, False, ['myo_topology_support'])],
                'support_flags': ['case_a', 'case_b'],
            },
            {
                'slices': [_slice('s1', True, True, ['myo_topology_support'])],
                'support_flags': ['case_a'],
            },
            {
                'slices': [_slice('s1', True, True, [])],
                'support_flags': ['case_a', 'case_b'],
            },
            {
                'slices': [_slice('s1', False, True, ['myo_topology_support'])],
                'support_flags': ['case_a', 'case_b'],
            },
            {
                'slices': [_slice('s1', True, True, ['myo_topology_support'])],
                'myo_topology_support': True,
                'border_contact_support': True,
            },
            {
                'slices': [_slice('s1', True, True, ['myo_topology_support'])],
                'support_flags': ['case_a', 'case_b', 'case_c'],
            },
            {
                'slices': [_slice('s1', True, True, ['myo_topology_support'])],
                'support_flags': [],
            },
            {
                'slices': [
                    _slice('s1', True, True, ['myo_topology_support']),
                    _slice('s2', True, False, ['myo_topology_support']),
                ],
                'support_flags': ['case_a', 'case_b'],
            },
        ]
        expected = [True, True, False, False, False, False, True, True, False, True]
        report = audit_positive_consistency(cases, expected)
        self.assertGreaterEqual(report['consistency'], 0.80)
        self.assertEqual(report['matches'], 10)


if __name__ == '__main__':
    unittest.main()
