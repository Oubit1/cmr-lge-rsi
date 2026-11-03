'''Offline unit tests for the draft review-only rule R-CMR-MYO-TEMPORAL-HOLE-REVIEW-01.

No network, no datasets, no production writes: every case is built inline.
'''

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import cmr_myo_temporal_hole_review_rule as rule

TRIGGER = rule.TRIGGER_FLAG
ALLOWED_SUPPORT_FLAGS = frozenset((
    'cmr_myo_small_area',
    'myo_topology',
    'cmr_myo_empty',
    'cmr_myo_touches_image_border',
))


def make_slice(index, modality='cmr', trigger=True, support=(), **extra):
    payload = {'slice_index': index, 'modality': modality, TRIGGER: trigger}
    for flag in support:
        payload[flag] = True
    payload.update(extra)
    return payload


def make_case(slices, **extra):
    case = {'case_id': 'case-1', 'modality': 'cmr', 'slices': list(slices)}
    case.update(extra)
    return case


class RuleDraftDefaultsTest(unittest.TestCase):
    def test_rule_is_default_off_and_review_only(self):
        self.assertEqual(rule.RULE_ID, 'R-CMR-MYO-TEMPORAL-HOLE-REVIEW-01')
        self.assertIs(rule.DEFAULT_ENABLED, False)
        self.assertIs(rule.REVIEW_ONLY, True)
        self.assertIs(rule.WRITES_PRODUCTION_RULE_LIBRARY, False)
        status = rule.activation_status()
        self.assertFalse(status['enabled'])
        self.assertTrue(status['review_only'])
        self.assertFalse(status['writes_production_rule_library'])
        self.assertFalse(status['activation_allowed_in_this_draft'])
        self.assertTrue(status['a1_audit_required'])
        self.assertFalse(status['a1_audit_verified_in_this_draft'])

    def test_support_flags_are_limited_to_existing_flags(self):
        self.assertEqual(set(rule.SUPPORT_FLAGS), set(ALLOWED_SUPPORT_FLAGS))
        self.assertEqual(len(rule.SUPPORT_FLAGS), 4)
        self.assertNotIn(TRIGGER, rule.SUPPORT_FLAGS)

    def test_manifest_declares_draft_and_default_off(self):
        manifest = rule.rule_draft_manifest()
        self.assertEqual(manifest['rule_id'], rule.RULE_ID)
        self.assertEqual(manifest['status'], 'draft')
        self.assertFalse(manifest['enabled'])
        self.assertTrue(manifest['review_only'])
        self.assertFalse(manifest['writes_production_rule_library'])
        self.assertEqual(manifest['numeric_thresholds_added'], 0)
        self.assertEqual(tuple(manifest['support_flags']), rule.SUPPORT_FLAGS)
        self.assertEqual(manifest['trigger_flag'], TRIGGER)

    def test_no_new_numeric_thresholds_are_declared(self):
        self.assertEqual(rule.NUMERIC_THRESHOLDS_ADDED, 0)
        self.assertEqual(rule.activation_status()['numeric_thresholds_added'], 0)
        self.assertEqual(rule.rule_draft_manifest()['numeric_thresholds_added'], 0)

    def test_activation_and_production_write_guards_raise(self):
        with self.assertRaises(rule.RuleNotActivatableError):
            rule.guard_activation()
        with self.assertRaises(rule.RuleNotActivatableError):
            rule.guard_activation(a1_audit_passed=True)
        with self.assertRaises(rule.RuleNotActivatableError):
            rule.guard_production_write(rule.RULE_ID)


class SliceSemanticsTest(unittest.TestCase):
    def test_positive_slice_requires_cmr_modality(self):
        non_cmr = make_slice(0, modality='lge', support=['cmr_myo_empty'])
        self.assertFalse(rule.slice_is_positive(non_cmr))
        cmr = make_slice(0, modality='cmr', support=['cmr_myo_empty'])
        self.assertTrue(rule.slice_is_positive(cmr))

    def test_positive_slice_requires_trigger_flag(self):
        no_flag = make_slice(0, trigger=False, support=['cmr_myo_empty'])
        self.assertFalse(rule.slice_is_positive(no_flag))
        missing_flag = make_slice(0, support=['cmr_myo_empty'])
        missing_flag.pop(TRIGGER)
        self.assertFalse(rule.slice_is_positive(missing_flag))
        unparsable = make_slice(0, trigger='maybe', support=['cmr_myo_empty'])
        self.assertFalse(rule.slice_is_positive(unparsable))
        self.assertFalse(rule.slice_is_positive(make_slice(0, trigger='false')))
        truthy_string = make_slice(0, trigger='TRUE', support=['cmr_myo_empty'])
        self.assertTrue(rule.slice_is_positive(truthy_string))

    def test_positive_slice_requires_one_support_flag(self):
        self.assertFalse(rule.slice_is_positive(make_slice(0, support=())))
        self.assertTrue(
            rule.slice_is_positive(make_slice(0, support=['myo_topology']))
        )

    def test_non_whitelisted_flags_are_ignored(self):
        slice_payload = make_slice(
            0,
            support=(),
            cmr_myo_area_mm2=42.0,
            temporal_hole_gap_ms=5,
            unknown_support_flag=True,
        )
        self.assertEqual(rule.slice_support_flags(slice_payload), ())
        self.assertFalse(rule.slice_is_positive(slice_payload))

    def test_allowed_support_flags_are_reported_in_whitelist_order(self):
        slice_payload = make_slice(
            0,
            support=['cmr_myo_touches_image_border', 'myo_topology'],
        )
        self.assertEqual(
            rule.slice_support_flags(slice_payload),
            ('myo_topology', 'cmr_myo_touches_image_border'),
        )

    def test_flags_nested_in_flag_container_are_honoured(self):
        slice_payload = make_slice(0, support=())
        slice_payload['flags'] = {'cmr_myo_empty': True}
        self.assertEqual(rule.slice_support_flags(slice_payload), ('cmr_myo_empty',))
        self.assertTrue(rule.slice_is_positive(slice_payload))

    def test_case_modality_is_used_as_fallback(self):
        slice_payload = {TRIGGER: True, 'cmr_myo_empty': True}
        self.assertTrue(rule.slice_is_positive(slice_payload, {'modality': 'cmr'}))
        self.assertFalse(rule.slice_is_positive(slice_payload, {'modality': 'lge'}))
        self.assertFalse(rule.slice_is_positive(slice_payload))


class CaseReviewTest(unittest.TestCase):
    def test_case_without_positive_slice_does_not_escalate(self):
        case = make_case([
            make_slice(0, modality='lge', support=['cmr_myo_empty']),
            make_slice(1, modality='cmr', trigger=False, support=['myo_topology']),
        ])
        record = rule.evaluate_case(case)
        self.assertEqual(record['positive_slice_count'], 0)
        self.assertFalse(record['escalate_for_review'])
        self.assertEqual(record['action'], 'none')

    def test_single_case_level_support_flag_is_not_enough(self):
        case = make_case([make_slice(0, support=['cmr_myo_small_area'])])
        record = rule.evaluate_case(case)
        self.assertEqual(record['positive_slice_count'], 1)
        self.assertEqual(record['case_support_flags'], ['cmr_myo_small_area'])
        self.assertFalse(record['escalate_for_review'])
        self.assertEqual(record['action'], 'none')

    def test_case_escalates_with_one_positive_slice_and_two_case_flags(self):
        case = make_case([
            make_slice(0, support=['cmr_myo_small_area']),
            make_slice(1, trigger=False, support=['myo_topology']),
        ])
        record = rule.evaluate_case(case)
        self.assertEqual(record['positive_slice_count'], 1)
        self.assertEqual(
            record['case_support_flags'], ['cmr_myo_small_area', 'myo_topology']
        )
        self.assertTrue(record['escalate_for_review'])
        self.assertEqual(record['action'], 'review')
        self.assertEqual(record['positive_slices'][0]['slice_index'], 0)
        self.assertEqual(
            record['positive_slices'][0]['support_flags'], ['cmr_myo_small_area']
        )
        self.assertFalse(record['enabled'])
        self.assertTrue(record['review_only'])
        self.assertFalse(record['writes_production_rule_library'])

    def test_repeated_flag_across_slices_counts_once(self):
        case = make_case([
            make_slice(0, support=['cmr_myo_empty']),
            make_slice(1, support=['cmr_myo_empty']),
        ])
        record = rule.evaluate_case(case)
        self.assertEqual(record['case_support_flags'], ['cmr_myo_empty'])
        self.assertFalse(record['escalate_for_review'])

    def test_case_level_flag_is_counted_towards_escalation(self):
        case = make_case(
            [make_slice(0, support=['cmr_myo_empty'])],
            **{'cmr_myo_touches_image_border': True}
        )
        record = rule.evaluate_case(case)
        self.assertEqual(
            record['case_support_flags'],
            ['cmr_myo_empty', 'cmr_myo_touches_image_border'],
        )
        self.assertTrue(record['escalate_for_review'])

    def test_positive_slice_support_flags_are_reported_separately(self):
        case = make_case([
            make_slice(0, support=['cmr_myo_empty']),
            make_slice(1, trigger=False, support=['myo_topology']),
        ])
        record = rule.evaluate_case(case)
        self.assertEqual(record['positive_slice_support_flags'], ['cmr_myo_empty'])
        self.assertEqual(
            record['case_support_flags'], ['cmr_myo_empty', 'myo_topology']
        )

    def test_case_support_flags_follow_slice_observation_order(self):
        case = make_case([
            make_slice(0, support=['cmr_myo_empty']),
            make_slice(1, trigger=False, support=['myo_topology']),
            make_slice(2, trigger=False, support=['cmr_myo_small_area']),
        ])
        record = rule.evaluate_case(case)
        self.assertEqual(
            record['case_support_flags'],
            ['cmr_myo_empty', 'myo_topology', 'cmr_myo_small_area'],
        )
        self.assertEqual(record['positive_slice_support_flags'], ['cmr_myo_empty'])
        self.assertEqual(record['action'], 'review')
        self.assertTrue(record['escalate_for_review'])

    def test_case_level_explicit_flags_are_appended_last(self):
        case = make_case(
            [make_slice(0, support=['cmr_myo_empty'])],
            **{'myo_topology': True}
        )
        record = rule.evaluate_case(case)
        self.assertEqual(
            record['case_support_flags'], ['cmr_myo_empty', 'myo_topology']
        )

    def test_numeric_measurements_do_not_change_the_decision(self):
        small = make_case([
            make_slice(
                0,
                support=['cmr_myo_empty'],
                cmr_myo_area_mm2=1.0,
                temporal_hole_gap_ms=2,
            )
        ])
        large = make_case([
            make_slice(
                0,
                support=['cmr_myo_empty'],
                cmr_myo_area_mm2=99999.0,
                temporal_hole_gap_ms=99999,
            )
        ])
        record_small = rule.evaluate_case(small)
        record_large = rule.evaluate_case(large)
        self.assertEqual(record_small, record_large)
        self.assertFalse(record_small['escalate_for_review'])
        self.assertEqual(record_small['numeric_thresholds_added'], 0)

    def test_empty_or_malformed_payloads_fail_closed(self):
        for payload in (None, {}, [], {'slices': None}, {'slices': 7}, {'slices': []}):
            record = rule.evaluate_case(payload)
            self.assertEqual(record['action'], 'none')
            self.assertFalse(record['escalate_for_review'])
            self.assertEqual(record['positive_slice_count'], 0)

    def test_batch_evaluation_is_deterministic_and_default_off(self):
        batch = {
            'case-b': make_case([make_slice(0, support=['cmr_myo_empty'])]),
            'case-a': make_case([
                make_slice(0, support=['cmr_myo_small_area']),
                make_slice(1, trigger=False, support=['myo_topology']),
            ]),
        }
        records = rule.evaluate_case_batch(batch)
        self.assertEqual([r['case_id'] for r in records], ['case-a', 'case-b'])
        self.assertEqual([r['action'] for r in records], ['review', 'none'])
        for record in records:
            self.assertFalse(record['enabled'])
            self.assertTrue(record['review_only'])
            self.assertFalse(record['writes_production_rule_library'])

    def test_only_review_or_none_actions_are_possible(self):
        cases = (
            {},
            make_case([]),
            make_case([make_slice(0, support=['cmr_myo_empty'])]),
            make_case([
                make_slice(0, support=['cmr_myo_small_area']),
                make_slice(1, trigger=False, support=['myo_topology']),
            ]),
        )
        for case in cases:
            self.assertIn(rule.evaluate_case(case)['action'], rule.ALLOWED_ACTIONS)

    def test_record_invariant_rejects_a_write_action(self):
        bad_record = {
            'action': 'write',
            'enabled': False,
            'review_only': True,
            'writes_production_rule_library': False,
            'numeric_thresholds_added': 0,
        }
        with self.assertRaises(rule.ReviewOnlyViolation):
            rule._assert_review_only(bad_record)

    def test_record_invariant_rejects_an_enabled_record(self):
        bad_record = {
            'action': 'review',
            'enabled': True,
            'review_only': True,
            'writes_production_rule_library': False,
            'numeric_thresholds_added': 0,
        }
        with self.assertRaises(rule.ReviewOnlyViolation):
            rule._assert_review_only(bad_record)

    def test_record_invariant_rejects_a_threshold_bearing_record(self):
        bad_record = {
            'action': 'review',
            'enabled': False,
            'review_only': True,
            'writes_production_rule_library': False,
            'numeric_thresholds_added': 1,
        }
        with self.assertRaises(rule.ReviewOnlyViolation):
            rule._assert_review_only(bad_record)


if __name__ == '__main__':
    unittest.main()
