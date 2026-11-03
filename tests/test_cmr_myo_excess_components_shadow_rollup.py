'''Offline unit tests for the excess-components shadow rollup audit.

All fixtures are synthetic in-memory mappings. Nothing is downloaded, written
or read from disk, and no labels, production rules or QC gates are touched.
'''

import copy
import json
import unittest

from experiments.cmr_myo_excess_components_shadow_rollup import (
    AuditInputError,
    B1_FALLBACK_EXTRACTOR,
    STRATUM,
    audit,
    dedup_compliance_rate,
    dedup_records,
    evaluate_group,
    hash_mismatch,
    resolve_extractor,
    select_sample,
    severity_tier,
)


def _slice(
    index,
    components,
    component_indices=None,
    slice_issue_types=('excess_components',),
    small_area=False,
    border=False,
    topology=False,
):
    if component_indices is None:
        component_indices = list(range(components))
    return {
        'slice_index': index,
        'components': components,
        'component_indices': list(component_indices),
        'issue_types': list(slice_issue_types),
        'small_area': bool(small_area),
        'border': bool(border),
        'topology': bool(topology),
    }


def _record(
    case_id,
    modality='LGE',
    label='positive',
    run_id='run-4',
    extractor='run4_cache',
    slices=None,
    issue_types=('excess_components',),
    **extra
):
    record = {
        'case_id': case_id,
        'modality': modality,
        'run_id': run_id,
        'extractor': extractor,
        'label': label,
        'issue_types': list(issue_types),
        'slices': list(slices or ()),
    }
    record.update(extra)
    return record


def _rolled(case_id, slices, label='positive', modality='LGE'):
    '''A record shaped like the dedup output, for evaluate_group tests.'''
    return {
        'case_id': case_id,
        'modality': modality,
        'label': label,
        'issue_types': ['excess_components'],
        'has_excess_issue': True,
        'slices': list(slices),
        'run_ids': ['run-4'],
        'extractors': ['run4_cache'],
        'hash_mismatch': False,
        'b1_fallback': False,
        'input_rows': 1,
    }


class SeverityTierTests(unittest.TestCase):
    def test_severity_tier_boundaries(self):
        self.assertEqual(severity_tier(0), 'none')
        self.assertEqual(severity_tier(1), 'none')
        self.assertEqual(severity_tier(2), 'mild')
        self.assertEqual(severity_tier(3), 'moderate')
        self.assertEqual(severity_tier(4), 'moderate')
        self.assertEqual(severity_tier(5), 'severe')
        self.assertEqual(severity_tier(9), 'severe')

    def test_severity_tier_rejects_bad_input(self):
        with self.assertRaises(AuditInputError):
            severity_tier('not-a-number')


class HashProvenanceTests(unittest.TestCase):
    def test_hash_mismatch_detection(self):
        stale = {'case_id': 'c1', 'modality': 'LGE', 'source_hash': 'aaa', 'expected_hash': 'bbb'}
        fresh = {'case_id': 'c2', 'modality': 'LGE', 'source_hash': 'aaa', 'expected_hash': 'aaa'}
        missing = {'case_id': 'c3', 'modality': 'LGE'}
        pinned = {
            'case_id': 'c4',
            'modality': 'LGE',
            'source_hash': 'aaa',
            'expected_hash': 'aaa',
            'hash_mismatch': True,
        }
        self.assertTrue(hash_mismatch(stale))
        self.assertFalse(hash_mismatch(fresh))
        self.assertFalse(hash_mismatch(missing))
        self.assertTrue(hash_mismatch(pinned))

    def test_b1_is_the_only_fallback_extractor(self):
        stale = {
            'case_id': 'c1',
            'modality': 'LGE',
            'extractor': 'run4_cache',
            'source_hash': 'aaa',
            'expected_hash': 'bbb',
        }
        fresh = {
            'case_id': 'c2',
            'modality': 'LGE',
            'extractor': 'run4_cache',
            'source_hash': 'aaa',
            'expected_hash': 'aaa',
        }
        missing = {'case_id': 'c3', 'modality': 'LGE', 'extractor': 'run5_cache'}
        self.assertEqual(resolve_extractor(stale), (B1_FALLBACK_EXTRACTOR, True))
        self.assertEqual(resolve_extractor(fresh), ('run4_cache', False))
        self.assertEqual(resolve_extractor(missing), ('run5_cache', False))


class DedupTests(unittest.TestCase):
    def test_one_record_per_case_modality_with_merged_components(self):
        raw = [
            _record(
                'case-a',
                run_id='run-4',
                slices=[_slice(3, 2, [0, 1], small_area=True)],
            ),
            _record(
                'case-a',
                run_id='run-5',
                extractor='run5_cache',
                slices=[_slice(3, 3, [1, 2, 3]), _slice(4, 1, [0])],
            ),
        ]
        deduped = dedup_records(raw)
        self.assertEqual(len(deduped), 1)
        record = deduped[0]
        self.assertEqual(record['case_id'], 'case-a')
        self.assertEqual(record['label'], 'positive')
        self.assertEqual(record['input_rows'], 2)
        self.assertEqual(record['run_ids'], ['run-4', 'run-5'])
        self.assertEqual([s['slice_index'] for s in record['slices']], [3, 4])
        merged = record['slices'][0]
        self.assertEqual(merged['components'], 3)
        self.assertEqual(merged['component_indices'], [0, 1, 2, 3])
        self.assertTrue(merged['small_area'])
        self.assertEqual(record['slices'][1]['components'], 1)
        self.assertEqual(dedup_compliance_rate(raw), 1.0)

    def test_dedup_does_not_mutate_input(self):
        raw = [
            _record('case-a', slices=[_slice(3, 2, [0, 1])]),
            _record('case-a', run_id='run-5', slices=[_slice(3, 3, [1, 2])]),
            _record('case-b', label='control', slices=[_slice(1, 1, [0])]),
        ]
        snapshot = copy.deepcopy(raw)
        dedup_records(raw)
        self.assertEqual(raw, snapshot)

    def test_dedup_merges_issue_types_and_slice_flags(self):
        raw = [
            _record(
                'case-x',
                run_id='run-4',
                issue_types=['excess_components'],
                slices=[
                    _slice(
                        2,
                        2,
                        [0, 1],
                        slice_issue_types=['excess_components'],
                        topology=False,
                    )
                ],
            ),
            _record(
                'case-x',
                run_id='run-15',
                extractor='run15_cache',
                issue_types=['border'],
                slices=[
                    _slice(
                        2,
                        3,
                        [1, 2],
                        slice_issue_types=['border'],
                        topology=True,
                    )
                ],
            ),
        ]
        deduped = dedup_records(raw)
        self.assertEqual(len(deduped), 1)
        record = deduped[0]
        self.assertEqual(record['issue_types'], ['border', 'excess_components'])
        self.assertTrue(record['has_excess_issue'])
        self.assertEqual(record['run_ids'], ['run-15', 'run-4'])
        self.assertEqual(record['extractors'], ['run15_cache', 'run4_cache'])
        merged = record['slices'][0]
        self.assertEqual(merged['components'], 3)
        self.assertEqual(merged['component_indices'], [0, 1, 2])
        self.assertEqual(merged['issue_types'], ['border', 'excess_components'])
        self.assertTrue(merged['topology'])
        self.assertFalse(merged['small_area'])

    def test_malformed_records_raise(self):
        with self.assertRaises(AuditInputError):
            dedup_records(['not-a-mapping'])
        with self.assertRaises(AuditInputError):
            dedup_records([{'modality': 'LGE'}])
        with self.assertRaises(AuditInputError):
            dedup_records([{'case_id': 'c', 'modality': 'LGE', 'slices': [{'components': 1}]}])
        with self.assertRaises(AuditInputError):
            dedup_records(
                [
                    {
                        'case_id': 'c',
                        'modality': 'LGE',
                        'slices': [{'slice_index': 1, 'components': -1}],
                    }
                ]
            )

    def test_empty_input_is_compliant_and_empty(self):
        self.assertEqual(dedup_records([]), [])
        self.assertEqual(dedup_compliance_rate([]), 1.0)


class SelectSampleTests(unittest.TestCase):
    def test_select_sample_is_bounded_and_order_independent(self):
        records = [
            _rolled('case-{0:03d}'.format(i), [_slice(1, 3, [0, 1, 2])])
            for i in range(60)
        ]
        records += [
            _rolled('ctrl-{0:03d}'.format(i), [_slice(1, 1, [0])], label='control')
            for i in range(12)
        ]
        forward = select_sample(records, max_positive=50, max_control=10)
        backward = select_sample(list(reversed(records)), max_positive=50, max_control=10)
        self.assertEqual(len(forward['positives']), 50)
        self.assertEqual(len(forward['controls']), 10)
        self.assertTrue(forward['positive_truncated'])
        self.assertTrue(forward['control_truncated'])
        self.assertEqual(forward['positive_pool'], 60)
        self.assertEqual(forward['control_pool'], 12)
        self.assertEqual(
            [row['case_id'] for row in forward['positives']],
            [row['case_id'] for row in backward['positives']],
        )
        self.assertEqual(
            [row['case_id'] for row in forward['controls']],
            [row['case_id'] for row in backward['controls']],
        )
        self.assertEqual(forward['positives'][0]['case_id'], 'case-000')
        self.assertEqual(forward['positives'][-1]['case_id'], 'case-049')

    def test_select_sample_zero_limits(self):
        records = [_rolled('case-000', [_slice(1, 2, [0, 1])])]
        sample = select_sample(records, max_positive=0, max_control=0)
        self.assertEqual(sample['positives'], [])
        self.assertEqual(sample['controls'], [])


class EvaluateGroupTests(unittest.TestCase):
    def test_group_metrics_component_distribution_and_cooccurrence(self):
        group = [
            _rolled('case-a', [_slice(1, 3, [0, 1, 2], small_area=True), _slice(2, 1, [0])]),
            _rolled('case-b', [_slice(1, 1, [0])]),
            _rolled('case-c', [_slice(3, 5, [0, 1, 2, 3, 4], border=True)]),
        ]
        metrics = evaluate_group(group, threshold=2)
        self.assertEqual(metrics['sample_size'], 3)
        self.assertEqual(metrics['confirmed_count'], 2)
        self.assertEqual(metrics['unconfirmed_count'], 1)
        self.assertEqual(metrics['confirmed_case_ids'], ['case-a', 'case-c'])
        self.assertEqual(metrics['unconfirmed_case_ids'], ['case-b'])
        self.assertAlmostEqual(metrics['true_positive_rate'], 2.0 / 3.0)
        self.assertTrue(metrics['insufficient_sample'])
        self.assertEqual(metrics['component_count_distribution'], {1: 2, 3: 1, 5: 1})
        self.assertEqual(metrics['max_component_count_distribution'], {1: 1, 3: 1, 5: 1})
        self.assertEqual(
            metrics['severity_tier_distribution'],
            {'none': 1, 'mild': 0, 'moderate': 1, 'severe': 1},
        )
        self.assertEqual(metrics['max_severity_tier'], 'severe')
        self.assertEqual(metrics['worst_slice_histogram'], {1: 2, 3: 1})
        self.assertEqual(
            metrics['worst_slices'][0],
            {'case_id': 'case-a', 'slice_index': 1, 'components': 3},
        )
        self.assertEqual(metrics['excess_slice_count'], 2)
        self.assertAlmostEqual(metrics['cooccurrence']['small_area'], 0.5)
        self.assertAlmostEqual(metrics['cooccurrence']['border'], 0.5)
        self.assertEqual(metrics['cooccurrence']['topology'], 0.0)
        self.assertEqual(metrics['cooccurrence_counts']['small_area'], 1)

    def test_group_metrics_handle_empty_group(self):
        metrics = evaluate_group([], threshold=2, expected_min_sample=1)
        self.assertEqual(metrics['sample_size'], 0)
        self.assertIsNone(metrics['true_positive_rate'])
        self.assertEqual(metrics['max_severity_tier'], 'none')
        self.assertEqual(metrics['excess_slice_count'], 0)
        self.assertEqual(metrics['cooccurrence']['small_area'], 0.0)
        self.assertEqual(metrics['worst_slices'], [])

    def test_group_metrics_reject_bad_threshold(self):
        with self.assertRaises(AuditInputError):
            evaluate_group([], threshold=0)


class AuditEndToEndTests(unittest.TestCase):
    def test_audit_rollup_report_and_hash_provenance(self):
        raw = [
            _record('case-a', run_id='run-4', slices=[_slice(3, 2, [0, 1], small_area=True)]),
            _record(
                'case-a',
                run_id='run-5',
                extractor='run5_cache',
                slices=[_slice(3, 3, [1, 2, 3]), _slice(4, 1, [0])],
            ),
            _record(
                'case-b',
                label='control',
                run_id='run-9',
                extractor='run9_cache',
                slices=[_slice(2, 1, [0])],
            ),
            _record(
                'case-c',
                run_id='run-13',
                extractor='run13_cache',
                source_hash='abc',
                expected_hash='def',
                slices=[_slice(5, 5, [0, 1, 2, 3, 4], border=True)],
            ),
        ]
        snapshot = copy.deepcopy(raw)
        report = audit(raw, max_positive=50, max_control=10)
        self.assertEqual(raw, snapshot)
        self.assertEqual(report['stratum'], STRATUM)
        self.assertTrue(report['read_only'])
        self.assertEqual(report['dedup']['input_rows'], 4)
        self.assertEqual(report['dedup']['unique_keys'], 3)
        self.assertEqual(report['dedup']['collapsed_rows'], 1)
        self.assertEqual(report['dedup']['dedup_compliance_rate'], 1.0)
        records = {row['case_id']: row for row in report['dedup']['records']}
        merged = records['case-a']['slices'][0]
        self.assertEqual(merged['slice_index'], 3)
        self.assertEqual(merged['components'], 3)
        self.assertEqual(merged['component_indices'], [0, 1, 2, 3])
        self.assertTrue(merged['small_area'])
        self.assertEqual(records['case-a']['run_ids'], ['run-4', 'run-5'])
        self.assertEqual(report['sample']['positive_sample_size'], 2)
        self.assertEqual(report['sample']['control_sample_size'], 1)
        self.assertEqual(report['sample']['positive_case_ids'], ['case-a', 'case-c'])
        self.assertEqual(report['sample']['control_case_ids'], ['case-b'])
        self.assertEqual(report['positives']['confirmed_count'], 2)
        self.assertEqual(report['positives']['true_positive_rate'], 1.0)
        self.assertTrue(report['positives']['insufficient_sample'])
        self.assertEqual(report['controls']['confirmed_count'], 0)
        self.assertEqual(report['hash_provenance']['hash_mismatch_count'], 1)
        self.assertEqual(report['hash_provenance']['hash_mismatch_cases'], ['case-c'])
        self.assertEqual(report['hash_provenance']['b1_fallback_count'], 1)
        self.assertEqual(report['hash_provenance']['b1_fallback_cases'], ['case-c'])
        self.assertEqual(
            report['hash_provenance']['b1_fallback_extractor'], B1_FALLBACK_EXTRACTOR
        )
        self.assertEqual(records['case-c']['extractors'], [B1_FALLBACK_EXTRACTOR])
        self.assertTrue(records['case-c']['b1_fallback'])
        self.assertNotIn('run13_cache', records['case-c']['extractors'])

    def test_audit_is_deterministic_and_serialisable(self):
        raw = [
            _record('case-b', slices=[_slice(1, 4, [0, 1, 2, 3], topology=True)]),
            _record('case-a', slices=[_slice(2, 2, [0, 1])]),
            _record('case-c', label='control', slices=[_slice(1, 1, [0])]),
        ]
        first = audit(copy.deepcopy(raw))
        second = audit(copy.deepcopy(raw))
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertEqual(first['dedup']['dedup_compliance_rate'], 1.0)
        self.assertEqual(
            [row['case_id'] for row in first['dedup']['records']],
            ['case-a', 'case-b', 'case-c'],
        )

    def test_audit_positive_sample_floor_uses_fifty_positives(self):
        raw = [
            _record('case-{0:03d}'.format(i), slices=[_slice(1, 2, [0, 1])])
            for i in range(51)
        ]
        raw.append(_record('ctrl-000', label='control', slices=[_slice(1, 1, [0])]))
        report = audit(raw)
        self.assertEqual(report['sample']['positive_sample_size'], 50)
        self.assertEqual(report['sample']['control_sample_size'], 1)
        self.assertTrue(report['sample']['positive_truncated'])
        self.assertFalse(report['sample']['control_truncated'])
        self.assertFalse(report['positives']['insufficient_sample'])
        self.assertEqual(report['positives']['sample_size'], 50)
        self.assertEqual(report['positives']['true_positive_rate'], 1.0)
        self.assertNotIn('case-050', report['sample']['positive_case_ids'])
        self.assertEqual(report['hash_provenance']['hash_mismatch_count'], 0)
        self.assertEqual(report['hash_provenance']['b1_fallback_count'], 0)

    def test_audit_has_no_write_paths(self):
        source_names = dir(audit)
        for forbidden in ('open', 'write', 'remove', 'unlink'):
            self.assertNotIn(forbidden, source_names)


if __name__ == '__main__':
    unittest.main()
