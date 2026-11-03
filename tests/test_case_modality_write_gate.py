"""Offline tests for experiments/case_modality_write_gate.py.

Covers the three acceptance metrics of the write-enforcing de-duplication gate:

* duplicate ``(case_id, modality)`` write records drop to zero and every
  rejected row is retained in the audit log;
* with a valid cached hash no new extractor is constructed (at most one build
  per key, only on hash mismatch);
* merged issue recall is not lower than any single incoming row and the review
  queue does not grow.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.case_modality_write_gate import (  # noqa: E402
    CaseModalityWriteGate,
    cache_hash_is_valid,
    case_modality_key,
    issue_index_map,
    merge_issue_index_maps,
    normalize_issue_types,
    normalize_slice_indices,
    run_write_gate,
)


def assert_map_covers(test, merged, incoming):
    '''Assert every issue/slice of ``incoming`` survives inside ``merged``.'''
    for issue, indices in incoming.items():
        test.assertIn(issue, merged)
        for index in indices:
            test.assertIn(index, merged[issue])


class KeyAndPayloadTests(unittest.TestCase):

    def test_key_is_normalised(self):
        self.assertEqual(case_modality_key({'case_id': ' C01 ', 'modality': 'LGE'}), ('C01', 'lge'))

    def test_key_requires_both_fields(self):
        with self.assertRaises(KeyError):
            case_modality_key({'case_id': 'C01'})
        with self.assertRaises(KeyError):
            case_modality_key({'case_id': '   ', 'modality': 'lge'})

    def test_normalizers_deduplicate(self):
        self.assertEqual(normalize_issue_types(['a', 'a', None, 'b']), ('a', 'b'))
        self.assertEqual(normalize_slice_indices([3, 1, 3, '2']), (1, 2, 3))
        self.assertEqual(normalize_slice_indices('4, 5,4'), (4, 5))

    def test_issue_index_map_accepts_legacy_shapes(self):
        self.assertEqual(
            issue_index_map({'issue_type': 'a', 'slice_indices': [2, 1]}),
            {'a': (1, 2)},
        )
        self.assertEqual(
            issue_index_map({'issues': [{'issue_type': 'b', 'slice_indices': [7]}]}),
            {'b': (7,)},
        )
        self.assertEqual(
            issue_index_map({'issue_slice_indices': {'c': [9, 9]}}),
            {'c': (9,)},
        )
        self.assertEqual(issue_index_map({}), {})

    def test_merge_preserves_all_indices(self):
        merged = merge_issue_index_maps({'a': (1, 2)}, {'a': (2, 3), 'b': (8,)})
        self.assertEqual(merged, {'a': (1, 2, 3), 'b': (8,)})

    def test_cache_hash_is_valid_only_on_equality(self):
        self.assertTrue(cache_hash_is_valid('abc', 'abc'))
        self.assertFalse(cache_hash_is_valid('abc', 'def'))
        self.assertFalse(cache_hash_is_valid(None, 'abc'))
        self.assertFalse(cache_hash_is_valid('abc', None))
        self.assertFalse(cache_hash_is_valid('', ''))


class DuplicateRejectionTests(unittest.TestCase):

    def test_duplicate_write_is_rejected_and_audited(self):
        gate = CaseModalityWriteGate()
        first = gate.write(
            {'case_id': 'C001', 'modality': 'LGE', 'issue_type': 'myo_small_area', 'slice_indices': [3, 1]}
        )
        second = gate.write(
            {'case_id': 'C001', 'modality': 'lge', 'issue_type': 'myo_small_area', 'slice_indices': [1, 2]}
        )
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, CaseModalityWriteGate.DUPLICATE_REASON)
        self.assertEqual(gate.queue_size(), 1)
        self.assertEqual(gate.rejected_rows, 1)
        self.assertEqual(gate.duplicate_rejections, 1)
        log = gate.audit_log
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['reason'], CaseModalityWriteGate.DUPLICATE_REASON)
        self.assertEqual(log[0]['row']['slice_indices'], [1, 2])
        self.assertEqual(log[0]['key'], ['C001', 'lge'])

    def test_no_duplicate_records_are_created(self):
        gate = CaseModalityWriteGate()
        for _ in range(5):
            gate.write({'case_id': 'C42', 'modality': 'MYO', 'issue_type': 'x'})
        self.assertEqual(gate.queue_size(), 1)
        self.assertEqual(gate.duplicate_rejections, 4)
        self.assertEqual(len(gate.audit_log), 4)
        self.assertEqual(gate.records[('C42', 'myo')].accepted_rows, 1)
        self.assertEqual(gate.records[('C42', 'myo')].rejected_rows, 4)

    def test_audit_log_is_isolated_from_later_mutation(self):
        gate = CaseModalityWriteGate()
        row = {'case_id': 'C7', 'modality': 'LGE', 'issue_type': 'a', 'slice_indices': [1]}
        gate.write(dict(row))
        gate.write(row)
        row['slice_indices'].append(99)
        row['case_id'] = 'MUTATED'
        log = gate.audit_log
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]['row']['slice_indices'], [1])
        self.assertEqual(log[0]['row']['case_id'], 'C7')

    def test_missing_key_row_is_rejected_and_audited(self):
        gate = CaseModalityWriteGate()
        decision = gate.write({'modality': 'LGE', 'issue_type': 'a'})
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, CaseModalityWriteGate.MALFORMED_REASON)
        self.assertEqual(gate.queue_size(), 0)
        self.assertEqual(gate.rejected_rows, 1)
        self.assertEqual(len(gate.audit_log), 1)


class RecallAndQueueTests(unittest.TestCase):

    def test_issue_types_merge_and_slice_indices_are_preserved(self):
        gate = CaseModalityWriteGate()
        gate.write({'case_id': 'C9', 'modality': 'MYO', 'issue_types': ['a'], 'slice_indices': [5, 4]})
        gate.write({'case_id': 'C9', 'modality': 'myo', 'issues': [{'issue_type': 'b', 'slice_indices': [8]}]})
        gate.write({'case_id': 'C9', 'modality': 'MYO', 'issue_slice_indices': {'a': [4, 7]}})
        record = gate.records[('C9', 'myo')]
        self.assertEqual(record.issue_slice_indices, {'a': (4, 5, 7), 'b': (8,)})
        self.assertEqual(record.issue_types, ('a', 'b'))
        self.assertEqual(gate.queue_size(), 1)

    def test_recall_never_drops_below_any_single_row(self):
        gate = CaseModalityWriteGate()
        rows = [
            {'case_id': 'C10', 'modality': 'LGE', 'issue_type': 'mi_multi', 'slice_indices': [2]},
            {'case_id': 'C10', 'modality': 'LGE', 'issue_type': 'mi_multi', 'slice_indices': [2, 4]},
            {'case_id': 'C10', 'modality': 'LGE', 'issue_type': 'mi_frag', 'slice_indices': [6]},
        ]
        for row in rows:
            gate.write(row)
        merged = gate.records[('C10', 'lge')].recall_view()
        for row in rows:
            assert_map_covers(self, merged, issue_index_map(row))
        self.assertEqual(gate.queue_size(), 1)

    def test_run_write_gate_summary(self):
        rows = [
            {'case_id': 'C1', 'modality': 'LGE', 'issue_type': 'a', 'slice_indices': [1]},
            {'case_id': 'C1', 'modality': 'LGE', 'issue_type': 'a', 'slice_indices': [1, 2]},
            {'case_id': 'C2', 'modality': 'LGE', 'issue_type': 'b', 'slice_indices': [3]},
            {'case_id': 'C2', 'modality': 'lge', 'issue_type': 'c', 'slice_indices': [4]},
        ]
        summary = run_write_gate(rows)
        self.assertEqual(summary['record_count'], 2)
        self.assertEqual(summary['duplicate_rejections'], 2)
        self.assertEqual(len(summary['audit_log']), 2)
        self.assertEqual(summary['records'][('C2', 'lge')]['issue_slice_indices'], {'b': [3], 'c': [4]})


class ExtractorCacheTests(unittest.TestCase):

    def test_valid_cache_hash_forbids_new_extractor(self):
        calls = []

        def factory(key, fresh):
            calls.append((key, fresh))
            return {'key': key}

        gate = CaseModalityWriteGate(cache_hashes={('C1', 'lge'): 'h1'}, extractor_factory=factory)
        first = gate.write({'case_id': 'C1', 'modality': 'LGE'}, fresh_hash='h1')
        second = gate.write({'case_id': 'C1', 'modality': 'LGE'}, fresh_hash='h1')
        self.assertEqual(calls, [])
        self.assertEqual(gate.extractor_builds, 0)
        self.assertTrue(first.extractor_refused)
        self.assertTrue(second.extractor_refused)
        self.assertTrue(gate.records[('C1', 'lge')].cache_valid)

    def test_hash_mismatch_builds_extractor_once_per_key(self):
        calls = []

        def factory(key, fresh):
            calls.append((key, fresh))
            return {'key': key}

        gate = CaseModalityWriteGate(cache_hashes={('C2', 'lge'): 'old'}, extractor_factory=factory)
        first = gate.write({'case_id': 'C2', 'modality': 'LGE'}, fresh_hash='new')
        second = gate.write({'case_id': 'C2', 'modality': 'LGE'}, fresh_hash='new')
        self.assertEqual(len(calls), 1)
        self.assertEqual(gate.extractor_builds, 1)
        self.assertTrue(first.extractor_built)
        self.assertFalse(second.extractor_built)
        self.assertEqual(second.record.extractor, first.record.extractor)
        self.assertTrue(second.record.cache_valid)

    def test_build_cap_per_key_with_repeated_mismatch(self):
        calls = []

        def factory(key, fresh):
            calls.append(key)
            return object()

        gate = CaseModalityWriteGate(extractor_factory=factory)
        for _ in range(4):
            gate.write({'case_id': 'C3', 'modality': 'MYO'}, fresh_hash='hx')
        self.assertEqual(len(calls), 1)
        self.assertEqual(gate.extractor_builds, 1)

    def test_no_factory_means_no_build(self):
        gate = CaseModalityWriteGate()
        decision = gate.write({'case_id': 'C4', 'modality': 'MYO'}, fresh_hash='h')
        self.assertFalse(decision.extractor_built)
        self.assertEqual(gate.extractor_builds, 0)


if __name__ == '__main__':
    unittest.main()
