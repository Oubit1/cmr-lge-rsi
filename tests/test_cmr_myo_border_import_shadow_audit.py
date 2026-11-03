'''Offline unit tests for the cmr_myo_touches_image_border import shadow audit.

All fixtures are synthetic and injected in memory; the suite runs without
datasets, patient images, credentials, network access or production caches.
'''

from __future__ import annotations

import copy
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.cmr_myo_border_import_shadow_audit import (
    B1_EXTRACTOR_NAME,
    CACHE_EXTRACTOR_NAME,
    CACHE_RUN_IDS,
    CachedCaseSummary,
    DEFAULT_CONTROL_BUDGET,
    DEFAULT_POSITIVE_BUDGET,
    MAX_COHORT_SIZE,
    build_shadow_audit_report,
    coerce_record,
    cooccurrence_summary,
    select_cohort,
    single_record_compliance,
    slice_index_profile,
)


def _record(
    case_id,
    *,
    modality='LGE',
    run_id='run-4',
    border=True,
    declared_hash='hash',
    payload_hash=None,
    extractor='cache',
    small_area=False,
    excess_components=False,
    missing_temporal_hole=False,
    marked_slice_indices=(),
):
    return CachedCaseSummary(
        case_id=case_id,
        modality=modality,
        run_id=run_id,
        declared_hash=declared_hash,
        payload_hash=declared_hash if payload_hash is None else payload_hash,
        touches_image_border=border,
        small_area=small_area,
        excess_components=excess_components,
        missing_temporal_hole=missing_temporal_hole,
        marked_slice_indices=tuple(marked_slice_indices),
        extractor=extractor,
    )


class CohortBoundingTests(unittest.TestCase):
    def test_budget_is_bounded_and_deterministic(self):
        records = [_record('P%03d' % index) for index in range(70)]
        records += [
            _record('C%03d' % index, border=False, run_id='run-5')
            for index in range(20)
        ]
        report = build_shadow_audit_report(records)
        cohort = report['cohort']
        self.assertEqual(cohort['positives_requested'], DEFAULT_POSITIVE_BUDGET)
        self.assertEqual(cohort['positives_selected'], DEFAULT_POSITIVE_BUDGET)
        self.assertEqual(cohort['controls_selected'], DEFAULT_CONTROL_BUDGET)
        self.assertEqual(cohort['cohort_size'], MAX_COHORT_SIZE)
        self.assertLessEqual(cohort['cohort_size'], 60)
        border_ids = [
            entry['case_id']
            for entry in report['slice_index']['per_case']
            if entry['group'] == 'border'
        ]
        self.assertEqual(border_ids[0], 'P000')
        self.assertEqual(border_ids[-1], 'P049')
        self.assertEqual(report['warnings'], [])

    def test_second_record_for_same_key_does_not_inflate_cohort(self):
        records = [
            _record('P001', run_id='run-4'),
            _record('P001', run_id='run-9'),
            _record('P002'),
        ]
        positives, controls = select_cohort(records)
        self.assertEqual([record.case_id for record in positives], ['P001', 'P002'])
        self.assertEqual(controls, [])

    def test_shortfall_is_reported_as_warning(self):
        records = [_record('P%02d' % index) for index in range(4)]
        report = build_shadow_audit_report(records)
        codes = [warning['code'] for warning in report['warnings']]
        self.assertIn('fewer_positive_records_than_requested', codes)
        self.assertIn('fewer_control_records_than_requested', codes)
        self.assertEqual(report['cohort']['positives_selected'], 4)
        self.assertEqual(report['cohort']['controls_selected'], 0)


class CooccurrenceTests(unittest.TestCase):
    def test_co_flag_counts_and_shares(self):
        records = [
            _record('P001', small_area=True),
            _record('P002', excess_components=True, missing_temporal_hole=True),
            _record('C001', border=False, small_area=True),
        ]
        report = build_shadow_audit_report(records)
        border = report['cooccurrence']['border_cases']
        control = report['cooccurrence']['control_cases']

        self.assertEqual(border['group_size'], 2)
        self.assertEqual(border['analyzed_cohort_size'], 3)
        self.assertEqual(border['flags']['small_area']['count'], 1)
        self.assertEqual(border['flags']['small_area']['share_of_group'], 0.5)
        self.assertEqual(
            border['flags']['small_area']['share_of_analyzed_cohort'], 0.333333
        )
        self.assertEqual(border['flags']['excess_components']['count'], 1)
        self.assertEqual(border['flags']['missing_temporal_hole']['count'], 1)
        self.assertEqual(border['summary']['cases_with_no_co_flag'], 0)
        self.assertEqual(border['summary']['cases_with_multiple_co_flags'], 1)
        self.assertEqual(border['summary']['cases_with_all_three_co_flags'], 0)

        self.assertEqual(control['group_size'], 1)
        self.assertEqual(control['flags']['small_area']['share_of_group'], 1.0)
        self.assertEqual(
            control['flags']['small_area']['share_of_analyzed_cohort'], 0.333333
        )

    def test_three_way_co_flag_case(self):
        records = [
            _record(
                'P001',
                small_area=True,
                excess_components=True,
                missing_temporal_hole=True,
            ),
        ]
        summary = cooccurrence_summary(records, len(records))
        self.assertEqual(summary['summary']['cases_with_all_three_co_flags'], 1)
        self.assertEqual(summary['summary']['cases_with_multiple_co_flags'], 1)
        self.assertEqual(summary['summary']['cases_with_no_co_flag'], 0)

    def test_summary_without_records_is_zero_filled(self):
        summary = cooccurrence_summary([], 0)
        self.assertEqual(summary['group_size'], 0)
        self.assertEqual(summary['flags']['small_area']['share_of_group'], 0.0)
        self.assertEqual(
            summary['flags']['small_area']['share_of_analyzed_cohort'], 0.0
        )


class SliceIndexTests(unittest.TestCase):
    def test_per_case_indices_histogram_and_worst_slice(self):
        records = [
            _record('P001', marked_slice_indices=(5, 3, 3, 7)),
            _record('P002', marked_slice_indices=(2,)),
            _record('C001', border=False),
        ]
        profile = slice_index_profile(records)
        per_case = {entry['case_id']: entry for entry in profile['per_case']}
        self.assertEqual(per_case['P001']['marked_slice_indices'], [3, 5, 7])
        self.assertEqual(per_case['P001']['marked_slice_count'], 3)
        self.assertEqual(per_case['P001']['worst_slice_index'], 7)
        self.assertEqual(per_case['P002']['worst_slice_index'], 2)
        self.assertIsNone(per_case['C001']['worst_slice_index'])
        self.assertEqual(
            profile['marked_slice_index_histogram'], {2: 1, 3: 1, 5: 1, 7: 1}
        )
        self.assertEqual(profile['worst_slice_index_histogram'], {2: 1, 7: 1})
        self.assertEqual(profile['cases_with_marked_slices'], 2)
        self.assertEqual(
            profile['cases_without_marked_slices'],
            [{'case_id': 'C001', 'modality': 'LGE'}],
        )

    def test_injected_worst_slice_selector(self):
        records = [_record('P001', marked_slice_indices=(4, 9, 6))]
        profile = slice_index_profile(
            records,
            worst_slice_selector=lambda indices: min(indices) if indices else None,
        )
        self.assertEqual(profile['per_case'][0]['worst_slice_index'], 4)
        self.assertEqual(profile['worst_slice_index_histogram'], {4: 1})

    def test_report_exposes_worst_slice_selector(self):
        records = [_record('P001', marked_slice_indices=(4, 9, 6))]
        report = build_shadow_audit_report(
            records,
            worst_slice_selector=lambda indices: max(indices) if indices else None,
        )
        self.assertEqual(report['slice_index']['per_case'][0]['worst_slice_index'], 9)


class IntegrityTests(unittest.TestCase):
    def test_hash_mismatch_uses_run9_b1_and_is_excluded_from_distributions(self):
        records = [
            _record('P001', declared_hash='match', small_area=True),
            _record('P002', declared_hash='declared', payload_hash='payload'),
            _record('C001', border=False),
        ]
        report = build_shadow_audit_report(records)
        integrity = report['integrity']
        self.assertEqual(integrity['cohort_records'], 3)
        self.assertEqual(integrity['cache_hash_mismatch_count'], 1)
        self.assertEqual(integrity['cache_hash_mismatch_rate'], 0.333333)
        self.assertEqual(integrity['run9_b1_call_count'], 1)
        call = integrity['run9_b1_calls'][0]
        self.assertEqual(call['case_id'], 'P002')
        self.assertEqual(call['extractor'], B1_EXTRACTOR_NAME)
        self.assertEqual(call['extractor_run'], 'run-9')
        self.assertEqual(report['cohort']['excluded_unverified_hash'], 1)
        self.assertEqual(report['cohort']['analyzed_records'], 2)

        analyzed_case_ids = [
            entry['case_id'] for entry in report['slice_index']['per_case']
        ]
        self.assertNotIn('P002', analyzed_case_ids)
        self.assertEqual(report['cooccurrence']['border_cases']['group_size'], 1)
        self.assertEqual(
            report['cooccurrence']['border_cases']['flags']['small_area']['count'], 1
        )
        self.assertEqual(
            report['cooccurrence']['border_cases']['analyzed_cohort_size'], 2
        )

    def test_all_verified_records_give_zero_mismatch(self):
        records = [_record('P001', run_id='run-13'), _record('C001', border=False)]
        report = build_shadow_audit_report(records)
        integrity = report['integrity']
        self.assertEqual(integrity['cache_hash_mismatch_count'], 0)
        self.assertEqual(integrity['cache_hash_mismatch_rate'], 0.0)
        self.assertEqual(integrity['run9_b1_call_count'], 0)
        self.assertEqual(integrity['violations'], [])
        self.assertEqual(integrity['verified_records'], 2)

    def test_unrecognized_extractor_with_matching_hash_is_a_violation(self):
        records = [_record('P001', extractor='adhoc-parser')]
        report = build_shadow_audit_report(records)
        codes = [item['code'] for item in report['violations']]
        self.assertEqual(codes, ['unrecognized_extractor'])
        self.assertEqual(report['violations'][0]['required_extractor'], CACHE_EXTRACTOR_NAME)

    def test_hash_mismatch_resolved_by_non_b1_extractor_is_a_violation(self):
        records = [
            _record(
                'P001',
                declared_hash='declared',
                payload_hash='payload',
                extractor='adhoc-parser',
            ),
        ]
        report = build_shadow_audit_report(records)
        codes = [item['code'] for item in report['violations']]
        self.assertEqual(codes, ['hash_mismatch_resolved_by_non_b1_extractor'])
        self.assertEqual(report['violations'][0]['required_extractor'], B1_EXTRACTOR_NAME)


class ComplianceTests(unittest.TestCase):
    def test_unique_case_modality_records_are_compliant(self):
        records = [
            _record('P001', modality='LGE'),
            _record('P001', modality='T1'),
            _record('C001', border=False, modality='LGE'),
        ]
        compliance = single_record_compliance(records)
        self.assertEqual(compliance['distinct_case_modality_keys'], 3)
        self.assertEqual(compliance['keys_with_single_record'], 3)
        self.assertEqual(compliance['keys_with_multiple_records'], 0)
        self.assertEqual(compliance['single_record_compliance_rate'], 1.0)
        self.assertTrue(compliance['compliant'])
        report = build_shadow_audit_report(records)
        self.assertEqual(report['blocking_findings'], [])

    def test_duplicate_case_modality_records_are_blocking(self):
        records = [
            _record('P001', run_id='run-4'),
            _record('P001', run_id='run-9'),
        ]
        compliance = single_record_compliance(records)
        self.assertEqual(compliance['distinct_case_modality_keys'], 1)
        self.assertEqual(compliance['keys_with_multiple_records'], 1)
        self.assertEqual(compliance['single_record_compliance_rate'], 0.0)
        self.assertFalse(compliance['compliant'])
        self.assertEqual(
            compliance['duplicate_keys'],
            [{'case_id': 'P001', 'modality': 'LGE'}],
        )
        report = build_shadow_audit_report(records)
        codes = [item['code'] for item in report['blocking_findings']]
        self.assertIn('multiple_records_for_case_modality', codes)
        self.assertFalse(
            report['integrity']['single_record_compliance']['compliant']
        )


class InputHandlingTests(unittest.TestCase):
    def test_mapping_records_are_coerced(self):
        raw = {
            'case_id': 'P001',
            'modality': 'LGE',
            'run_id': 'run-18',
            'declared_hash': 'h',
            'payload_hash': 'h',
            'touches_image_border': True,
            'small_area': 1,
            'marked_slice_indices': [3, 5],
        }
        record = coerce_record(raw)
        self.assertIsInstance(record, CachedCaseSummary)
        self.assertTrue(record.small_area)
        self.assertEqual(record.marked_slice_indices, (3, 5))
        self.assertEqual(record.extractor, CACHE_EXTRACTOR_NAME)
        report = build_shadow_audit_report([raw])
        self.assertEqual(report['cohort']['positives_selected'], 1)

    def test_missing_required_mapping_field_raises(self):
        raw = {'case_id': 'P001', 'modality': 'LGE', 'run_id': 'run-4'}
        with self.assertRaises(ValueError):
            coerce_record(raw)

    def test_unsupported_record_type_raises(self):
        with self.assertRaises(TypeError):
            coerce_record(42)

    def test_negative_marked_slice_index_raises(self):
        raw = {
            'case_id': 'P001',
            'modality': 'LGE',
            'run_id': 'run-4',
            'declared_hash': 'h',
            'payload_hash': 'h',
            'touches_image_border': True,
            'marked_slice_indices': [-1],
        }
        with self.assertRaises(ValueError):
            coerce_record(raw)

    def test_empty_snapshot_raises(self):
        with self.assertRaises(ValueError):
            build_shadow_audit_report([])

    def test_inputs_are_not_mutated(self):
        raw = {
            'case_id': 'P001',
            'modality': 'LGE',
            'run_id': 'run-4',
            'declared_hash': 'h',
            'payload_hash': 'h',
            'touches_image_border': True,
            'marked_slice_indices': [3, 5],
        }
        snapshot = copy.deepcopy(raw)
        original = _record('P002', marked_slice_indices=(3, 5))
        report = build_shadow_audit_report([original, coerce_record(raw)])
        self.assertEqual(raw, snapshot)
        self.assertEqual(original.marked_slice_indices, (3, 5))
        self.assertEqual(coerce_record(original), original)
        self.assertIsInstance(report, dict)


class SafetyTests(unittest.TestCase):
    def test_report_declares_read_only_and_reused_cache_runs(self):
        records = [_record('P001'), _record('C001', border=False)]
        report = build_shadow_audit_report(records)
        self.assertEqual(report['action'], 'cmr_myo_border_import_shadow_audit')
        self.assertEqual(report['mode'], 'read_only_import_shadow')
        safety = report['safety']
        self.assertTrue(safety['read_only'])
        self.assertFalse(safety['label_mutation'])
        self.assertFalse(safety['production_rule_change'])
        self.assertFalse(safety['threshold_change'])
        self.assertFalse(safety['qc_safety_check_weakened'])
        self.assertFalse(safety['external_io'])
        population = report['population']
        self.assertEqual(population['flag'], 'cmr_myo_touches_image_border')
        self.assertEqual(population['declared_n'], 931)
        self.assertEqual(population['cache_runs_reused'], list(CACHE_RUN_IDS))
        self.assertEqual(population['extractor_run_for_hash_mismatch'], 'run-9')
        for run_id in ('run-4', 'run-5', 'run-9', 'run-13', 'run-18'):
            self.assertIn(run_id, population['cache_runs_reused'])


if __name__ == '__main__':
    unittest.main()
