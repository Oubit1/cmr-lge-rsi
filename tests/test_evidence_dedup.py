import unittest

from experiments.evidence_dedup import (
    deduplicate_representative_cases,
    guard_iteration_history,
    is_run_3_6_class,
    preservation_report,
    rollup_cmr_flags,
    round_trip_check,
)


class EvidenceDedupTests(unittest.TestCase):
    @staticmethod
    def _sample_records():
        records = []
        for _ in range(6):
            records.append(
                {
                    'case_id': 'C1',
                    'modality': 'CMR',
                    'issue_type': 'myo_small_area',
                    'slice_index': 1,
                }
            )
        records.append(
            {
                'case_id': 'C1',
                'modality': 'CMR',
                'issue_type': 'boundary_touch',
                'slice_index': 2,
            }
        )
        records.append(
            {
                'case_id': 'C1',
                'modality': 'CMR',
                'issue_type': 'myo_small_area',
                'slice_index': 2,
            }
        )
        records.append(
            {
                'case_id': 'C1',
                'modality': 'LGE',
                'issue_type': 'lge_scar',
                'slice_index': 3,
            }
        )
        records.append(
            {
                'case_id': 'C1',
                'modality': 'LGE',
                'issue_type': 'lge_scar',
                'slice_index': 3,
            }
        )
        return records

    def test_deduplicates_rows_and_preserves_issues_and_slices(self):
        records = self._sample_records()
        payload = deduplicate_representative_cases(records)
        self.assertEqual(len(payload['representative_cases']), 2)
        self.assertGreaterEqual(payload['stats']['row_reduction'], 0.70)
        issue_types = {
            issue
            for representative in payload['representative_cases']
            for issue in representative['issue_types']
        }
        self.assertEqual(
            issue_types,
            {'myo_small_area', 'boundary_touch', 'lge_scar'},
        )
        self.assertEqual(set(payload['stats']['all_slice_indices']), {1, 2, 3})
        report = preservation_report(records, payload)
        self.assertTrue(report['ok'], report)
        self.assertEqual(report['missing_issue_types'], [])
        self.assertTrue(report['round_trip']['ok'])

    def test_round_trip_keeps_every_flagged_slice_addressable(self):
        records = self._sample_records()
        payload = deduplicate_representative_cases(records)
        report = round_trip_check(records, payload)
        self.assertTrue(report['ok'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['original_key_count'], 4)

    def test_rollup_keeps_flagged_slices_addressable(self):
        payload = deduplicate_representative_cases(self._sample_records())
        rollups = rollup_cmr_flags(
            payload['representative_cases'],
            total_slices_by_case={'C1': 10},
            high_volume_threshold=3,
        )
        self.assertEqual(len(rollups), 1)
        rollup = rollups[0]
        self.assertEqual(rollup['case_id'], 'C1')
        self.assertEqual(rollup['cmr_flagged_count'], 3)
        self.assertAlmostEqual(rollup['cmr_flag_rate'], 0.3)
        self.assertEqual(rollup['worst_slice_index'], 3)
        self.assertEqual(rollup['severity_tier'], 'moderate')
        self.assertTrue(rollup['high_volume'])
        self.assertEqual(rollup['flagged_slice_indices'], [1, 2, 3])
        self.assertEqual(
            rollup['slice_indices_by_issue']['myo_small_area'],
            [1, 2],
        )

    def test_duplication_guard_flags_run_3_through_6_but_not_b1(self):
        history = [
            {'experiment_class': 'run-3-cmr-shadow'},
            'run_4_lge_shadow',
            {'name': 'run-5-topology-shadow'},
            'run-6-advisory-rules',
            'B1_evidence_dedup',
            'cmr_myo_small_area_shadow',
        ]
        result = guard_iteration_history(history, new_class='B1_evidence_dedup')
        self.assertEqual(result['flagged_count'], 4)
        self.assertFalse(result['new_class_flagged'])
        flagged_classes = {entry['experiment_class'] for entry in result['flagged']}
        self.assertEqual(
            flagged_classes,
            {
                'run-3-cmr-shadow',
                'run_4_lge_shadow',
                'run-5-topology-shadow',
                'run-6-advisory-rules',
            },
        )
        allowed_classes = {entry['experiment_class'] for entry in result['allowed']}
        self.assertIn('B1_evidence_dedup', allowed_classes)

    def test_guard_does_not_flag_run_30_or_b1(self):
        self.assertFalse(is_run_3_6_class('run-30-cmr-shadow'))
        self.assertFalse(is_run_3_6_class('B1_evidence_dedup'))
        self.assertTrue(is_run_3_6_class('run-3/4/5/6 experiment class'))
        result = guard_iteration_history([], new_class='B1_evidence_dedup')
        self.assertEqual(result['flagged_count'], 0)
        self.assertFalse(result['new_class_flagged'])


if __name__ == '__main__':
    unittest.main()
