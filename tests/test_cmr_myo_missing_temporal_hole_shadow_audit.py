'''Offline unit tests for the cmr_myo_missing_temporal_hole shadow audit.

No external data, no downloads, no disk writes: every fixture is synthetic and
built in memory.
'''

import unittest

from experiments.cmr_myo_missing_temporal_hole_shadow_audit import (
    HOLE_MARKER,
    RUN9_B1_CONTRACT_VERSION,
    audit_case_temporal_holes,
    canonical_hash,
    classify_temporal_holes,
    cooccurrence_rates,
    enforce_single_record_per_case_modality,
    extract_via_run9_b1_contract,
    filter_reused_runs,
    index_slices_for_record,
    run_shadow_audit,
    verify_cache_hashes,
)


def make_hole(hole_id, frames=(), slices=(), persistent=None):
    hole = {'hole_id': hole_id, 'frames': list(frames), 'slices': list(slices)}
    if persistent is not None:
        hole['persistent'] = persistent
    return hole


def make_slice(index, markers=(), holes=()):
    return {'slice_index': index, 'markers': list(markers), 'holes': list(holes)}


def make_case(case_id, modality, run_id, slices=()):
    slices = list(slices)
    payload = {
        'run_id': run_id,
        'case_id': case_id,
        'modality': modality,
        'summary': {'slice_count': len(slices)},
    }
    return {
        'case_id': case_id,
        'modality': modality,
        'run_id': run_id,
        'cache_payload': payload,
        'cache_hash': canonical_hash(payload),
        'slices': slices,
    }


def make_run9_b1_payload(case_id, modality, slices):
    return {
        'run_id': 'run-9',
        'contract_version': RUN9_B1_CONTRACT_VERSION,
        'case_id': case_id,
        'modality': modality,
        'b1_slices': slices,
    }


class CanonicalHashTests(unittest.TestCase):
    def test_canonical_hash_is_stable_and_order_independent(self):
        first = {'b': [2, 1], 'a': {'y': 1, 'x': 2}}
        second = {'a': {'x': 2, 'y': 1}, 'b': [2, 1]}
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertNotEqual(canonical_hash(first), canonical_hash({'b': [2, 1], 'a': {'x': 9, 'y': 1}}))
        self.assertEqual(len(canonical_hash(first)), 64)


class CacheIntegrityTests(unittest.TestCase):
    def test_verify_cache_hashes_reports_pass_rate_and_mismatches(self):
        record = make_case('C-001', 'LGE', 'run-4', [])
        tampered = dict(record)
        tampered['cache_payload'] = dict(record['cache_payload'], summary={'slice_count': 99})
        verification = verify_cache_hashes([record, tampered])
        self.assertEqual(verification['checked'], 2)
        self.assertEqual(verification['passed'], 1)
        self.assertEqual(verification['mismatch_count'], 1)
        self.assertAlmostEqual(verification['pass_rate'], 0.5)
        self.assertEqual(verification['mismatched'][0]['case_id'], 'C-001')
        self.assertEqual(verification['mismatched'][0]['modality'], 'LGE')

    def test_verify_cache_hashes_handles_empty_input(self):
        verification = verify_cache_hashes([])
        self.assertEqual(verification['checked'], 0)
        self.assertEqual(verification['pass_rate'], 0.0)
        self.assertEqual(verification['mismatch_count'], 0)


def _keys(records):
    return sorted((r['case_id'], r['modality']) for r in records)


class KeyUniquenessTests(unittest.TestCase):
    def test_single_record_per_case_modality_detects_duplicates(self):
        first = make_case('C-001', 'LGE', 'run-4', [])
        duplicate = make_case('C-001', 'LGE', 'run-5', [])
        other_modality = make_case('C-001', 'CINE', 'run-4', [])
        result = enforce_single_record_per_case_modality([first, duplicate, other_modality])
        self.assertEqual(result['unique_keys'], 2)
        self.assertEqual(result['violations'], 1)
        self.assertFalse(result['one_record_per_case_modality'])
        self.assertEqual(result['duplicates'][0], {'case_id': 'C-001', 'modality': 'LGE'})
        self.assertEqual(_keys(result['records']), [('C-001', 'CINE'), ('C-001', 'LGE')])
        self.assertEqual(result['records'][1]['run_id'], 'run-4')

    def test_filter_reused_runs_keeps_only_allowed_runs(self):
        kept = make_case('C-001', 'LGE', 'run-13', [])
        skipped = make_case('C-002', 'LGE', 'run-2', [])
        result = filter_reused_runs([kept, skipped])
        self.assertEqual(len(result['kept']), 1)
        self.assertEqual(result['skipped_count'], 1)
        self.assertEqual(result['skipped_runs'], ['run-2'])
        self.assertEqual(result['skipped_case_ids'], ['C-002'])


class SliceIndexTests(unittest.TestCase):
    def test_slice_index_addressable_and_duplicate_detection(self):
        record = make_case(
            'C-001',
            'LGE',
            'run-4',
            [make_slice(0), make_slice(1), make_slice(3)],
        )
        indexed = index_slices_for_record(record)
        self.assertEqual(indexed['n_slices'], 3)
        self.assertTrue(indexed['addressable'])
        self.assertEqual(sorted(indexed['index'].keys()), [0, 1, 3])
        self.assertEqual(indexed['index'][3]['slice_index'], 3)

        duplicated = make_case('C-002', 'LGE', 'run-4', [make_slice(2), make_slice(2)])
        dup_indexed = index_slices_for_record(duplicated)
        self.assertFalse(dup_indexed['addressable'])
        self.assertEqual(dup_indexed['duplicate_slice_indices'], [2])


class TemporalHoleTests(unittest.TestCase):
    def test_temporal_hole_persistence_counts_and_ratios(self):
        holes = [
            make_hole('h1', frames=[3, 4, 5]),
            make_hole('h2', frames=[7], slices=[7]),
            make_hole('h3', slices=[2, 3]),
            make_hole('h4', frames=[9], slices=[1]),
        ]
        stats = classify_temporal_holes(holes)
        self.assertEqual(stats['total'], 4)
        self.assertEqual(stats['persistent'], 2)
        self.assertEqual(stats['isolated'], 2)
        self.assertAlmostEqual(stats['persistent_ratio'], 0.5)
        self.assertAlmostEqual(stats['isolated_ratio'], 0.5)

    def test_explicit_persistence_flag_overrides_span(self):
        forced_isolated = classify_temporal_holes([make_hole('h1', frames=[1, 2], persistent=False)])
        forced_persistent = classify_temporal_holes([make_hole('h2', frames=[1], persistent=True)])
        self.assertEqual(forced_isolated['isolated'], 1)
        self.assertEqual(forced_isolated['persistent'], 0)
        self.assertEqual(forced_persistent['persistent'], 1)
        self.assertEqual(forced_persistent['isolated'], 0)

    def test_per_case_aggregation_sums_slices(self):
        record = make_case(
            'C-001',
            'LGE',
            'run-4',
            [
                make_slice(0, [HOLE_MARKER], [make_hole('h1', frames=[0, 1])]),
                make_slice(1, [HOLE_MARKER], [make_hole('h2', frames=[1])]),
                make_slice(2),
            ],
        )
        per_case = audit_case_temporal_holes(record)
        aggregate = per_case['aggregate']
        self.assertEqual(aggregate['total'], 2)
        self.assertEqual(aggregate['persistent'], 1)
        self.assertEqual(aggregate['isolated'], 1)
        self.assertAlmostEqual(aggregate['persistent_ratio'], 0.5)
        self.assertEqual(per_case['per_slice']['2']['total'], 0)


class CooccurrenceTests(unittest.TestCase):
    def test_cooccurrence_rates_with_sibling_markers(self):
        slices = [
            make_slice(0, [HOLE_MARKER, 'cmr_myo_empty', 'cmr_myo_small_area']),
            make_slice(1, [HOLE_MARKER, 'cmr_myo_empty']),
            make_slice(2, [HOLE_MARKER]),
            make_slice(3, ['cmr_myo_empty']),
        ]
        report = cooccurrence_rates(slices)
        self.assertEqual(report['hole_marker'], HOLE_MARKER)
        self.assertEqual(report['hole_marker_slices'], 3)
        self.assertEqual(report['per_marker']['cmr_myo_empty']['cooccurring_slices'], 2)
        self.assertAlmostEqual(report['per_marker']['cmr_myo_empty']['rate'], 0.666667)
        self.assertEqual(report['per_marker']['cmr_myo_small_area']['rate'], 0.333333)
        self.assertEqual(report['per_marker']['cmr_centroid_drift']['cooccurring_slices'], 0)
        self.assertEqual(report['per_marker']['cmr_layer_area_jump']['rate'], 0.0)


def _run9_slice(index, markers, holes):
    return {
        'b1_slice_index': index,
        'b1_markers': list(markers),
        'b1_holes': holes,
    }


class Run9B1ContractTests(unittest.TestCase):
    def test_run9_b1_contract_extractor_maps_payload(self):
        payload = make_run9_b1_payload(
            'C-009',
            'LGE',
            [
                _run9_slice(
                    4,
                    [HOLE_MARKER],
                    [
                        {
                            'b1_hole_id': 'z1',
                            'b1_frames': [4, 5],
                            'b1_slice_span': [4, 5],
                            'b1_area_mm2': 12.5,
                        }
                    ],
                )
            ],
        )
        extracted = extract_via_run9_b1_contract(payload)
        self.assertEqual(extracted['case_id'], 'C-009')
        self.assertEqual(extracted['extracted_by'], 'run9_b1_contract')
        self.assertEqual(extracted['slices'][0]['slice_index'], 4)
        self.assertEqual(extracted['slices'][0]['holes'][0]['frames'], [4, 5])
        self.assertTrue(classify_temporal_holes(extracted['slices'][0]['holes'])['persistent'] == 1)

    def test_run9_b1_contract_extractor_rejects_foreign_payloads(self):
        with self.assertRaises(ValueError):
            extract_via_run9_b1_contract({'run_id': 'run-9'})
        with self.assertRaises(ValueError):
            extract_via_run9_b1_contract({'run_id': 'run-4', 'contract_version': RUN9_B1_CONTRACT_VERSION})


class ShadowAuditTests(unittest.TestCase):
    def test_run9_fallback_used_only_on_hash_mismatch(self):
        good = make_case(
            'C-001',
            'LGE',
            'run-4',
            [make_slice(0, [HOLE_MARKER], [make_hole('h1', frames=[0, 1])])],
        )
        bad = make_case('C-002', 'LGE', 'run-9', [make_slice(0, [HOLE_MARKER], [])])
        bad['cache_payload'] = dict(bad['cache_payload'], tampered=True)
        bad['run9_b1_payload'] = make_run9_b1_payload(
            'C-002',
            'LGE',
            [
                _run9_slice(
                    0,
                    [HOLE_MARKER, 'cmr_myo_empty'],
                    [{'b1_hole_id': 'h9', 'b1_frames': [0], 'b1_slice_span': [0]}],
                )
            ],
        )

        report = run_shadow_audit([good, bad], cohort={'positive': 1, 'control': 1})

        self.assertEqual(report['audit_mode'], 'read_only_shadow')
        self.assertEqual(report['cache_integrity']['mismatch_count'], 1)
        self.assertEqual(report['cache_mismatches'][0]['case_id'], 'C-002')
        self.assertEqual(report['run9_b1_fallback']['extractor'], 'run9_b1_contract')
        self.assertEqual(report['run9_b1_fallback']['resolved_count'], 1)
        self.assertEqual(report['run9_b1_fallback']['unrecoverable_count'], 0)
        self.assertFalse(report['run9_b1_fallback']['second_toolset_used'])
        self.assertTrue(report['key_uniqueness']['one_record_per_case_modality'])
        self.assertTrue(report['slice_index']['C-002|LGE']['addressable'])
        self.assertEqual(report['slice_index']['C-002|LGE']['n_slices'], 1)
        self.assertEqual(report['temporal_holes']['aggregate']['total'], 2)
        self.assertEqual(report['temporal_holes']['aggregate']['persistent'], 1)
        self.assertEqual(report['temporal_holes']['aggregate']['isolated'], 1)
        self.assertAlmostEqual(
            report['cooccurrence']['per_case']['C-002|LGE']['per_marker']['cmr_myo_empty']['rate'],
            1.0,
        )
        self.assertTrue(report['guardrails']['read_only'])
        self.assertFalse(report['guardrails']['labels_modified'])
        self.assertFalse(report['guardrails']['datasets_touched'])

    def test_unrecoverable_mismatch_is_reported_and_excluded(self):
        bad = make_case('C-003', 'LGE', 'run-9', [make_slice(0, [HOLE_MARKER])])
        bad['cache_payload'] = dict(bad['cache_payload'], tampered=True)

        report = run_shadow_audit([bad], cohort={'positive': 1, 'control': 0})

        self.assertEqual(report['cache_integrity']['mismatch_count'], 1)
        self.assertEqual(report['run9_b1_fallback']['resolved_count'], 0)
        self.assertEqual(report['run9_b1_fallback']['unrecoverable_count'], 1)
        self.assertEqual(report['run9_b1_fallback']['unrecoverable'][0]['case_id'], 'C-003')
        self.assertNotIn('C-003|LGE', report['slice_index'])
        self.assertEqual(report['temporal_holes']['aggregate']['total'], 0)

    def test_cohort_bound_is_enforced(self):
        with self.assertRaises(ValueError):
            run_shadow_audit([], cohort={'positive': 51, 'control': 0})
        with self.assertRaises(ValueError):
            run_shadow_audit([], cohort={'positive': 50, 'control': 11})
        report = run_shadow_audit([], cohort={'positive': 50, 'control': 10})
        self.assertEqual(report['cohort']['positive'], 50)
        self.assertEqual(report['cohort']['control'], 10)

    def test_audit_is_read_only_and_filters_foreign_runs(self):
        kept = make_case('C-001', 'LGE', 'run-4', [make_slice(0, [HOLE_MARKER])])
        skipped = make_case('C-002', 'LGE', 'run-2', [make_slice(0, [HOLE_MARKER])])
        records = [kept, skipped]
        snapshot = [dict(record) for record in records]

        report = run_shadow_audit(records, cohort={'positive': 1, 'control': 1})

        self.assertEqual(report['cache_filter']['kept'], 1)
        self.assertEqual(report['cache_filter']['skipped_count'], 1)
        self.assertEqual(report['cache_filter']['skipped_runs'], ['run-2'])
        self.assertNotIn('C-002|LGE', report['slice_index'])
        self.assertEqual(records, snapshot)
        self.assertEqual(sorted(report['runs_reused']), sorted(set(report['runs_reused'])))

    def test_empty_audit_reports_zeroed_metrics(self):
        report = run_shadow_audit([], cohort={'positive': 0, 'control': 0})
        self.assertEqual(report['cache_integrity']['checked'], 0)
        self.assertEqual(report['temporal_holes']['aggregate']['total'], 0)
        self.assertEqual(report['temporal_holes']['aggregate']['persistent_ratio'], 0.0)
        self.assertEqual(report['cooccurrence']['overall']['hole_marker_slices'], 0)
        self.assertEqual(report['cooccurrence']['overall']['per_marker']['cmr_myo_empty']['rate'], 0.0)
        self.assertEqual(report['slice_index'], {})


if __name__ == '__main__':
    unittest.main()
