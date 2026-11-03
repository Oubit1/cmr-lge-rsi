'''Offline unit tests for the bounded border-touch shadow audit.

No network, no datasets, no production state: every case payload is synthesised
in-process and every assertion is deterministic.
'''

from __future__ import annotations

import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cmr_myo_touches_border_bounded_shadow_audit import (  # noqa: E402
    CONTROL_BUDGET,
    CO_OCCURRENCE_KEYS,
    FALLBACK_SOURCE,
    FINDING_KEY,
    POSITIVE_BUDGET,
    TRUNCATION_CROP,
    TRUNCATION_FOV,
    TRUNCATION_NONE,
    WRITE_TARGETS,
    MalformedCasePayload,
    MissingExtractorError,
    classify_truncation,
    payload_hash,
    record_from_payload,
    run_bounded_shadow_audit,
    touched_sides,
    with_payload_hash,
    write_counts,
)


def _case(
    case_id,
    *,
    finding=True,
    modality='cmr',
    slice_index=4,
    n_slices=12,
    rows=10,
    cols=10,
    mask_rows=(2, 7),
    mask_cols=(2, 7),
    extra=(),
):
    '''Synthetic case payload with inline geometry.'''
    findings = []
    if finding:
        findings.append(FINDING_KEY)
    findings.extend(extra)
    return {
        'case_id': case_id,
        'modality': modality,
        'findings': findings,
        'slice_index': slice_index,
        'n_slices': n_slices,
        'rows': rows,
        'cols': cols,
        'mask_rows': list(mask_rows),
        'mask_cols': list(mask_cols),
    }


class BoundedSamplingTest(unittest.TestCase):
    def test_budget_and_determinism(self):
        candidates = [_case('pos-%03d' % i) for i in range(80)]
        candidates += [_case('ctl-%03d' % i, finding=False) for i in range(30)]
        shuffled = list(candidates)
        random.Random(7).shuffle(shuffled)

        report = run_bounded_shadow_audit(shuffled)
        self.assertEqual(len(report.positives), POSITIVE_BUDGET)
        self.assertEqual(len(report.controls), CONTROL_BUDGET)

        pos_ids = [row['case_id'] for row in report.positives]
        self.assertEqual(pos_ids, sorted(pos_ids))
        self.assertEqual(pos_ids[0], 'pos-000')
        self.assertEqual([r['case_id'] for r in report.controls][0], 'ctl-000')

        self.assertEqual(report.bounded['positives_available'], 80)
        self.assertEqual(report.bounded['controls_available'], 30)
        self.assertEqual(report.bounded['positives_audited'], POSITIVE_BUDGET)
        self.assertEqual(report.bounded['controls_audited'], CONTROL_BUDGET)
        self.assertEqual(report.completion_rate, 1.0)

        again = run_bounded_shadow_audit(list(reversed(candidates)))
        self.assertEqual([r['case_id'] for r in again.positives], pos_ids)
        self.assertEqual(
            [r['case_id'] for r in again.controls],
            [r['case_id'] for r in report.controls],
        )

    def test_small_pools_are_kept_whole(self):
        report = run_bounded_shadow_audit([_case('p-1'), _case('c-1', finding=False)])
        self.assertEqual([r['case_id'] for r in report.positives], ['p-1'])
        self.assertEqual([r['case_id'] for r in report.controls], ['c-1'])
        self.assertEqual(report.completion_rate, 1.0)

    def test_non_cmr_cases_are_not_used_as_controls(self):
        report = run_bounded_shadow_audit([_case('c-1', finding=False, modality='lge')])
        self.assertEqual(report.controls, [])
        self.assertEqual(report.control_false_positive_rate, 0.0)


class BorderGeometryTest(unittest.TestCase):
    def test_side_detection(self):
        left = record_from_payload(
            'case-left',
            {'rows': 10, 'cols': 10, 'mask_rows': [2, 7], 'mask_cols': [0, 4]},
        )
        self.assertEqual(touched_sides(left), ('left',))

        corner = record_from_payload(
            'case-corner',
            {'rows': 10, 'cols': 10, 'mask_rows': [0, 3], 'mask_cols': [6, 9]},
        )
        self.assertEqual(touched_sides(corner), ('right', 'top'))

        clean = record_from_payload(
            'case-clean',
            {'rows': 10, 'cols': 10, 'mask_rows': [2, 7], 'mask_cols': [2, 7]},
        )
        self.assertEqual(touched_sides(clean), ())

    def test_truncation_classification(self):
        crop = record_from_payload(
            'crop',
            {
                'rows': 10,
                'cols': 10,
                'mask_rows': [2, 7],
                'mask_cols': [0, 4],
                'slice_index': 4,
                'n_slices': 12,
            },
        )
        self.assertEqual(classify_truncation(crop), TRUNCATION_CROP)

        first_slice = record_from_payload(
            'fov-first',
            {
                'rows': 10,
                'cols': 10,
                'mask_rows': [2, 7],
                'mask_cols': [0, 4],
                'slice_index': 0,
                'n_slices': 12,
            },
        )
        self.assertEqual(classify_truncation(first_slice), TRUNCATION_FOV)

        last_slice = record_from_payload(
            'fov-last',
            {
                'rows': 10,
                'cols': 10,
                'mask_rows': [2, 7],
                'mask_cols': [0, 4],
                'slice_index': 11,
                'n_slices': 12,
            },
        )
        self.assertEqual(classify_truncation(last_slice), TRUNCATION_FOV)

        spanning = record_from_payload(
            'fov-span',
            {
                'rows': 10,
                'cols': 10,
                'mask_rows': [2, 7],
                'mask_cols': [0, 9],
                'slice_index': 4,
                'n_slices': 12,
            },
        )
        self.assertEqual(classify_truncation(spanning), TRUNCATION_FOV)

        clean = record_from_payload(
            'clean',
            {
                'rows': 10,
                'cols': 10,
                'mask_rows': [2, 7],
                'mask_cols': [2, 7],
                'slice_index': 4,
                'n_slices': 12,
            },
        )
        self.assertEqual(classify_truncation(clean), TRUNCATION_NONE)

    def test_malformed_payloads_rejected(self):
        with self.assertRaises(MalformedCasePayload):
            record_from_payload('bad', {'rows': 10, 'cols': 10, 'mask_rows': [2, 7]})
        with self.assertRaises(MalformedCasePayload):
            record_from_payload(
                'bad-inverted',
                {'rows': 10, 'cols': 10, 'mask_rows': [7, 2], 'mask_cols': [2, 3]},
            )


class CacheReuseAndFallbackTest(unittest.TestCase):
    def test_cache_hit_reused_and_hash_mismatch_falls_back(self):
        good = with_payload_hash(_case('c-1', mask_rows=(0, 4), mask_cols=(2, 7)))
        self.assertEqual(good['payload_hash'], payload_hash(good))

        stale = _case('c-2', mask_rows=(0, 4), mask_cols=(2, 7))
        stale['payload_hash'] = '0' * 64

        caches = {'run-5': {'c-1': good}, 'run-9': {'c-2': stale}}
        extracted = []

        def extractor(case_id):
            extracted.append(case_id)
            return _case(case_id, mask_rows=(0, 4), mask_cols=(2, 7))

        candidates = [
            {'case_id': 'c-1', 'modality': 'cmr', 'findings': [FINDING_KEY]},
            {'case_id': 'c-2', 'modality': 'cmr', 'findings': [FINDING_KEY]},
        ]
        report = run_bounded_shadow_audit(candidates, caches=caches, extractor=extractor)

        self.assertEqual(report.cache_reuse_counts['run-5'], 1)
        self.assertEqual(sum(report.cache_reuse_counts.values()), 1)
        self.assertAlmostEqual(report.cache_reuse_rate, 0.5)
        self.assertEqual(report.hash_mismatch_case_ids, ['c-2'])
        self.assertEqual(report.fallback_case_ids, ['c-2'])
        self.assertEqual(report.fallback_reasons['c-2'], 'hash_mismatch:run-9')
        self.assertEqual(report.fallback_extractor, FALLBACK_SOURCE)
        self.assertEqual(extracted, ['c-2'])
        self.assertEqual(report.inline_case_ids, [])
        self.assertEqual(report.write_counts, {target: 0 for target in WRITE_TARGETS})

    def test_cache_miss_without_extractor_raises(self):
        candidates = [{'case_id': 'c-9', 'modality': 'cmr', 'findings': [FINDING_KEY]}]
        with self.assertRaises(MissingExtractorError):
            run_bounded_shadow_audit(candidates, caches={'run-4': {}})

    def test_cache_miss_with_extractor_is_reported(self):
        def extractor(case_id):
            return _case(case_id, mask_cols=(0, 5))

        candidates = [{'case_id': 'c-9', 'modality': 'cmr', 'findings': [FINDING_KEY]}]
        report = run_bounded_shadow_audit(candidates, caches={}, extractor=extractor)
        self.assertEqual(report.cache_miss_case_ids, ['c-9'])
        self.assertEqual(report.fallback_case_ids, ['c-9'])
        self.assertEqual(report.fallback_reasons['c-9'], 'cache_miss')
        self.assertEqual(report.fallback_extractor, FALLBACK_SOURCE)
        self.assertEqual(report.positives[0]['sides'], ['left'])

    def test_inline_payload_used_without_caches(self):
        report = run_bounded_shadow_audit([_case('p-1', mask_cols=(0, 4))])
        self.assertEqual(report.inline_case_ids, ['p-1'])
        self.assertEqual(report.cache_miss_case_ids, [])
        self.assertEqual(report.fallback_case_ids, [])
        self.assertAlmostEqual(report.cache_reuse_rate, 0.0)


class AuditMetricTest(unittest.TestCase):
    def _sample(self):
        positives = []
        for i in range(POSITIVE_BUDGET):
            extra = []
            if i % 5 == 0:
                extra.append('small_area')
            if i % 10 == 0:
                extra.append('excess_components')
            positives.append(_case('p-%02d' % i, mask_cols=(0, 5), extra=tuple(extra)))

        controls = [
            _case('c-%02d' % i, finding=False, extra=('empty',) if i == 3 else ())
            for i in range(CONTROL_BUDGET)
        ]
        controls[0]['mask_rows'] = [0, 3]  # touches the top border -> control FP
        return positives + controls

    def test_control_false_positive_rate_and_co_occurrence(self):
        report = run_bounded_shadow_audit(self._sample())

        self.assertEqual(len(report.positives), POSITIVE_BUDGET)
        self.assertEqual(len(report.controls), CONTROL_BUDGET)
        self.assertEqual(report.control_false_positive_case_ids, ['c-00'])
        self.assertAlmostEqual(report.control_false_positive_rate, 0.1)

        self.assertEqual(sorted(report.co_occurrence), sorted(CO_OCCURRENCE_KEYS))
        self.assertEqual(report.co_occurrence['small_area']['positives'], 10)
        self.assertEqual(report.co_occurrence['excess_components']['positives'], 5)
        self.assertEqual(report.co_occurrence['empty']['positives'], 0)
        self.assertEqual(report.co_occurrence['empty']['controls'], 1)
        self.assertAlmostEqual(report.co_occurrence['empty']['control_rate'], 0.1)

    def test_positive_rows_report_sides_and_truncation(self):
        report = run_bounded_shadow_audit(self._sample())
        row = report.positives[0]
        self.assertEqual(row['sides'], ['left'])
        self.assertEqual(row['truncation'], TRUNCATION_CROP)
        self.assertEqual(row['slice_index'], 4)
        self.assertEqual(row['n_slices'], 12)
        self.assertEqual(report.positive_label_without_geometry_touch, [])

    def test_label_without_geometry_touch_is_reported(self):
        report = run_bounded_shadow_audit([_case('p-1', mask_rows=(2, 7), mask_cols=(2, 7))])
        self.assertEqual(report.positive_label_without_geometry_touch, ['p-1'])
        self.assertEqual(report.positives[0]['truncation'], TRUNCATION_NONE)


class ReadOnlyGuaranteeTest(unittest.TestCase):
    def test_write_counts_are_zero(self):
        self.assertEqual(write_counts(), {target: 0 for target in WRITE_TARGETS})

    def test_candidates_are_not_mutated(self):
        candidates = [_case('p-%02d' % i, mask_cols=(0, 4)) for i in range(5)]
        candidates += [_case('c-%02d' % i, finding=False) for i in range(3)]
        snapshot = copy.deepcopy(candidates)

        report = run_bounded_shadow_audit(candidates)

        self.assertEqual(candidates, snapshot)
        self.assertEqual(report.write_counts, {target: 0 for target in WRITE_TARGETS})
        self.assertEqual(set(report.to_dict()['write_counts']), set(WRITE_TARGETS))

    def test_cache_payloads_are_not_mutated(self):
        cached = with_payload_hash(_case('c-1', mask_cols=(0, 4)))
        caches = {'run-13': {'c-1': cached}}
        snapshot = copy.deepcopy(caches)

        run_bounded_shadow_audit(
            [{'case_id': 'c-1', 'modality': 'cmr', 'findings': [FINDING_KEY]}],
            caches=caches,
        )

        self.assertEqual(caches, snapshot)


if __name__ == '__main__':
    unittest.main()
