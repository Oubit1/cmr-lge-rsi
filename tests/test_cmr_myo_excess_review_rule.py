import unittest

from experiments import cmr_myo_excess_review_rule as rule


def _slice(**overrides):
    base = {
        'modality': 'cmr',
        'cmr_myo_excess_components': True,
        'cmr_myo_small_area': False,
        'cmr_myo_touches_image_border': False,
        'cmr_myo_missing_temporal_hole': False,
        'cmr_centroid_drift': False,
        'myo_topology': False,
    }
    base.update(overrides)
    return base


class MetadataTests(unittest.TestCase):
    def test_disabled_by_default_read_only_and_not_promoted(self):
        self.assertFalse(rule.DEFAULT_ENABLED)
        self.assertFalse(rule.METADATA['enabled_by_default'])
        self.assertEqual(rule.METADATA['activation'], 'not_activated')
        self.assertEqual(rule.METADATA['promotion'], 'not_promoted')
        self.assertFalse(rule.METADATA['writes_production_labels'])
        self.assertEqual(rule.METADATA['purpose'], 'QC review trigger only')
        self.assertIn('segmentation correction', rule.METADATA['not_for'])
        self.assertIn('clinical diagnosis', rule.METADATA['not_for'])

    def test_zero_new_numeric_thresholds_and_non_overlap_statement(self):
        self.assertEqual(rule.METADATA['new_numeric_thresholds'], 0)
        self.assertFalse(rule.METADATA['percentile_on_run_values'])
        self.assertEqual(rule.METADATA['non_overlap']['rule'], 'R-CMR-MYO-SMALL-01')
        statement = rule.METADATA['non_overlap']['statement']
        self.assertIn('does not duplicate or broaden', statement)
        self.assertIn('cmr_myo_excess_components', statement)
        self.assertIn('cmr_myo_small_area', statement)


class SliceRuleTests(unittest.TestCase):
    def test_positive_requires_modality_excess_and_same_slice_support(self):
        self.assertTrue(rule.evaluate_slice(_slice(cmr_myo_small_area=True))['positive'])
        self.assertTrue(rule.evaluate_slice(_slice(cmr_myo_touches_image_border=True))['positive'])
        self.assertTrue(rule.evaluate_slice(_slice(cmr_myo_missing_temporal_hole=True))['positive'])
        self.assertTrue(rule.evaluate_slice(_slice(cmr_centroid_drift=True))['positive'])
        self.assertTrue(rule.evaluate_slice(_slice(myo_topology=True))['positive'])

        self.assertFalse(
            rule.evaluate_slice(
                _slice(cmr_myo_excess_components=False, cmr_myo_small_area=True)
            )['positive']
        )
        self.assertFalse(
            rule.evaluate_slice(_slice(modality='lge', cmr_myo_small_area=True))['positive']
        )
        self.assertFalse(rule.evaluate_slice(_slice())['positive'])

    def test_reports_support_flag_co_occurrence_for_slice(self):
        result = rule.evaluate_slice(
            _slice(cmr_myo_small_area=True, cmr_centroid_drift=True)
        )
        self.assertTrue(result['positive'])
        self.assertEqual(
            result['support_flags'],
            ('cmr_myo_small_area', 'cmr_centroid_drift'),
        )

    def test_small_area_only_case_is_not_positive(self):
        self.assertFalse(
            rule.evaluate_slice(
                _slice(cmr_myo_excess_components=False, cmr_myo_small_area=True)
            )['positive']
        )


class CaseRuleTests(unittest.TestCase):
    def test_case_escalation_requires_positive_slice_and_escalation_flag(self):
        no_positive = {
            'case_id': 'no-positive',
            'slices': (_slice(cmr_myo_excess_components=False, cmr_myo_small_area=True),),
            'cmr_layer_area_jump': True,
        }
        self.assertFalse(rule.evaluate_case(no_positive)['case_escalation'])

        positive_no_escalation = {
            'case_id': 'positive-no-escalation',
            'slices': (_slice(cmr_myo_small_area=True),),
        }
        self.assertFalse(rule.evaluate_case(positive_no_escalation)['case_escalation'])

        positive_with_case_escalation = {
            'case_id': 'positive-case-escalation',
            'slices': (_slice(cmr_myo_small_area=True),),
            'cmr_layer_area_jump': True,
        }
        self.assertTrue(
            rule.evaluate_case(positive_with_case_escalation)['case_escalation']
        )

    def test_slice_level_escalation_flags_are_recognized(self):
        case = {
            'case_id': 'slice-escalation',
            'slices': (_slice(myo_topology=True, cmr_myo_empty=True),),
        }
        result = rule.evaluate_case(case)
        self.assertTrue(result['case_escalation'])
        self.assertIn('cmr_myo_empty', result['escalation_flags'])


class ShadowReplayTests(unittest.TestCase):
    def test_shadow_replay_counts_positives_escalations_cooccurrence_and_control_fpr(self):
        digest = {
            'cases': (
                {
                    'case_id': 'case-pos',
                    'slices': (_slice(cmr_myo_small_area=True), _slice()),
                    'cmr_layer_area_jump': True,
                },
                {
                    'case_id': 'case-no-escalation',
                    'slices': (_slice(myo_topology=True),),
                },
                {
                    'case_id': 'control-clean',
                    'is_control': True,
                    'slices': (
                        _slice(
                            cmr_myo_excess_components=False,
                            cmr_myo_small_area=True,
                        ),
                    ),
                },
                {
                    'case_id': 'control-flagged',
                    'is_control': True,
                    'slices': (_slice(cmr_centroid_drift=True),),
                    'cmr_myo_empty': True,
                },
            )
        }
        report = rule.replay_cached_digest(digest)
        self.assertEqual(report['per_slice_total'], 5)
        self.assertEqual(report['per_slice_positives'], 3)
        self.assertEqual(report['per_case_total'], 4)
        self.assertEqual(report['per_case_escalations'], 2)
        self.assertEqual(report['support_flag_cooccurrence']['cmr_myo_small_area'], 1)
        self.assertEqual(report['support_flag_cooccurrence']['myo_topology'], 1)
        self.assertEqual(report['support_flag_cooccurrence']['cmr_centroid_drift'], 1)
        self.assertEqual(report['support_flag_cooccurrence']['cmr_myo_touches_image_border'], 0)
        self.assertEqual(report['control_total'], 2)
        self.assertEqual(report['control_escalations'], 1)
        self.assertEqual(report['control_false_positive_rate'], 0.5)
        self.assertEqual(report['mode'], 'shadow_replay_read_only')
