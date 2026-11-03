"""Offline unit tests for experiments.case_modality_write_dedup_gate.

Every test runs without network access, datasets or third-party packages.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.case_modality_write_dedup_gate import (  # noqa: E402
    CacheMissError,
    CaseModalityWriteGate,
    ExtractorCache,
    GateClosedError,
    MergeResult,
    ValidationFailedError,
    case_modality_key,
    collect_label_signatures,
    count_duplicate_rows,
    merge_case_modality_rows,
    normalize_slice_indices,
)


GROUP = ('C001', 'LGE')


def row(
    case_id='C001',
    modality='LGE',
    issue_type=('mi_fragmentation',),
    slice_indices=(3, 5),
    **extra
):
    record = {
        'case_id': case_id,
        'modality': modality,
        'issue_type': list(issue_type),
        'slice_indices': list(slice_indices),
    }
    record.update(extra)
    return record


class MergeTests(unittest.TestCase):
    def test_duplicate_rows_collapse_to_zero_with_full_issue_recall(self):
        existing = [row(issue_type=('mi_fragmentation',), slice_indices=(3, 5), label=1)]
        incoming = [row(issue_type=('myo_small_area',), slice_indices=(7,), label=1)]

        merged = merge_case_modality_rows(existing, incoming)

        self.assertEqual(len(merged.rows), 1)
        self.assertEqual(count_duplicate_rows(merged.rows), 0)
        self.assertEqual(merged.duplicate_rows, 0)
        self.assertEqual(merged.collapsed_incoming, 1)
        self.assertEqual(merged.collapsed_existing, 0)
        only = merged.rows[0]
        self.assertEqual(list(only['issue_type']), ['mi_fragmentation', 'myo_small_area'])
        self.assertEqual(list(only['slice_indices']), [3, 5, 7])
        self.assertEqual(case_modality_key(only), GROUP)

    def test_duplicate_existing_rows_are_collapsed_too(self):
        existing = [
            row(issue_type=('mi_fragmentation',), slice_indices=(1,)),
            row(issue_type=('myo_small_area',), slice_indices=(2,)),
        ]

        merged = merge_case_modality_rows(existing, [])

        self.assertEqual(len(merged.rows), 1)
        self.assertEqual(merged.collapsed_existing, 1)
        self.assertEqual(merged.collapsed_incoming, 0)
        self.assertEqual(list(merged.rows[0]['issue_type']), ['mi_fragmentation', 'myo_small_area'])
        self.assertEqual(list(merged.rows[0]['slice_indices']), [1, 2])

    def test_slice_index_ranges_expand_and_stay_sorted(self):
        self.assertEqual(normalize_slice_indices(('3-6', '10', 5)), (3, 4, 5, 6, 10))
        self.assertEqual(normalize_slice_indices(None), ())
        self.assertEqual(normalize_slice_indices((7, '7', ' 8 ')), (7, 8))

    def test_rows_missing_case_or_modality_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_case_modality_rows([], [{'case_id': 'C001', 'issue_type': ['a']}])
        with self.assertRaises(ValueError):
            merge_case_modality_rows([], [{'case_id': '   ', 'modality': 'LGE'}])
        with self.assertRaises(ValueError):
            merge_case_modality_rows([], [{'case_id': None, 'modality': 'LGE'}])


class LabelIntegrityTests(unittest.TestCase):
    def test_merge_keeps_first_seen_label_and_reports_conflict(self):
        existing = [row(issue_type=('mi_fragmentation',), label=1)]
        incoming = [row(issue_type=('myo_small_area',), label=0)]

        gate = CaseModalityWriteGate(name='label_gate')
        merged = gate.write(existing, incoming)
        report = gate.validate(merged, labels_before=collect_label_signatures(existing))

        self.assertEqual(merged.rows[0]['label'], 1)
        self.assertEqual(merged.label_conflicts, (GROUP,))
        self.assertEqual(report.labels_modified, 0)
        self.assertEqual(report.duplicate_rows, 0)
        self.assertEqual(report.missing_issue_types, ())
        self.assertTrue(report.passed)

    def test_label_signatures_snapshot_before_merge(self):
        existing = [row(label=1, labels=[0, 1])]
        signatures = collect_label_signatures(existing)
        self.assertIn(GROUP, signatures)
        self.assertEqual(len(signatures[GROUP]), 2)


class ExtractorCacheTests(unittest.TestCase):
    def test_valid_cache_reuses_extractor_without_building_a_new_one(self):
        calls = []

        def factory(payload):
            calls.append(payload)
            return {'extractor': payload['token']}

        cache = ExtractorCache(factory)
        payload = {'case_id': 'C001', 'modality': 'LGE', 'token': 't1'}

        first, reused_first = cache.resolve(payload)
        second, reused_second = cache.resolve(payload)

        self.assertFalse(reused_first)
        self.assertTrue(reused_second)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        stats = cache.stats
        self.assertEqual(stats.created, 1)
        self.assertEqual(stats.reused, 1)
        self.assertEqual(stats.violations, 0)
        self.assertAlmostEqual(stats.reuse_rate, 0.5)

    def test_valid_hash_flag_blocks_extractor_creation(self):
        calls = []

        def factory(payload):
            calls.append(payload)
            return object()

        cache = ExtractorCache(factory)

        with self.assertRaises(CacheMissError):
            cache.resolve({'token': 't1'}, hash_is_valid=True)

        self.assertEqual(calls, [])
        self.assertEqual(cache.stats.created, 0)
        self.assertEqual(cache.stats.violations, 1)

        cache.prime({'token': 't1'}, 'extractor')
        extractor, reused = cache.resolve({'token': 't1'}, hash_is_valid=True)
        self.assertTrue(reused)
        self.assertEqual(extractor, 'extractor')
        self.assertEqual(cache.stats.created, 0)
        self.assertEqual(cache.stats.violations, 1)


class WriteGateLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.gate = CaseModalityWriteGate(name='test_gate')
        self.existing = [row(issue_type=('mi_fragmentation',), slice_indices=(2,), label=1)]
        self.incoming = [row(issue_type=('myo_small_area',), slice_indices=(4,), label=1)]

    def test_validation_passes_then_done_closes_the_gate(self):
        merged = self.gate.write(self.existing, self.incoming)
        report = self.gate.validate(
            merged, labels_before=collect_label_signatures(self.existing)
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.duplicate_rows, 0)
        self.assertEqual(report.missing_issue_types, ())
        self.assertEqual(report.labels_modified, 0)
        self.assertEqual(report.cache_violations, 0)
        self.assertEqual(report.extractors_created, 0)

        self.gate.mark_done(report)
        self.assertTrue(self.gate.is_done)
        self.assertEqual(self.gate.state, 'DONE')

        with self.assertRaises(GateClosedError):
            self.gate.write(self.existing, self.incoming)
        with self.assertRaises(GateClosedError):
            self.gate.resolve_extractor({'token': 't1'})
        self.assertEqual(self.gate.rejected_proposals, 2)

    def test_mark_done_refuses_failing_validation(self):
        self.gate.write(self.existing, [])
        tampered = MergeResult(rows=(dict(self.existing[0]), dict(self.existing[0])))

        report = self.gate.validate(tampered)

        self.assertEqual(report.duplicate_rows, 1)
        self.assertFalse(report.passed)
        with self.assertRaises(ValidationFailedError):
            self.gate.mark_done(report)
        self.assertFalse(self.gate.is_done)

    def test_validation_flags_dropped_issue_types(self):
        merged = merge_case_modality_rows(self.existing, self.incoming)
        tampered_row = dict(merged.rows[0])
        tampered_row['issue_type'] = ['mi_fragmentation']
        tampered = MergeResult(
            rows=(tampered_row,),
            issue_types=dict(merged.issue_types),
            labels=dict(merged.labels),
        )

        report = self.gate.validate(tampered)

        self.assertFalse(report.issue_types_complete)
        self.assertEqual(len(report.missing_issue_types), 1)
        self.assertIn('myo_small_area', report.missing_issue_types[0])
        self.assertFalse(report.passed)

    def test_gate_extractor_cache_is_reused(self):
        payload = {'case_id': 'C001', 'modality': 'LGE', 'token': 't'}
        calls = []

        def factory(value):
            calls.append(value)
            return {'x': 1}

        _first, reused_first = self.gate.resolve_extractor(payload, factory=factory)
        _second, reused_second = self.gate.resolve_extractor(payload)

        self.assertFalse(reused_first)
        self.assertTrue(reused_second)
        self.assertEqual(len(calls), 1)

        self.gate.write(self.existing, self.incoming)
        report = self.gate.validate(
            labels_before=collect_label_signatures(self.existing)
        )

        self.assertEqual(report.extractors_created, 1)
        self.assertAlmostEqual(report.cache_reuse_rate, 0.5)
        self.assertEqual(report.cache_violations, 0)
        self.assertTrue(report.passed)


if __name__ == '__main__':
    unittest.main()
