'''Offline unit tests for experiments/case_modality_dedup_hard_gate.py.

No network, no datasets, no patient data: every case here operates on synthetic
case ids and in-memory records only.
'''

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.case_modality_dedup_hard_gate import (
    ACTION_CANONICAL,
    ACTION_MERGED,
    ACTION_REJECTED,
    DEFAULT_CONVERGENCE_WINDOW,
    EXTRACTOR_CREATE,
    EXTRACTOR_REUSE,
    STATE_DONE,
    STATE_OPEN,
    STATE_VALIDATED,
    CaseModalityDedupHardGate,
    clean_run_streak,
    convergence_verdict,
    dedup_invariants,
    dedup_key,
    dedup_proposals_in_history,
    dedup_records,
    duplicate_record_count,
    is_valid_hash,
    issue_type_coverage,
    normalize_cache_hash,
    normalize_issue_types,
    normalize_modality,
    normalize_slice_indices,
    proposal_kinds,
)


class NormalizationTests(unittest.TestCase):
    def test_dedup_key_normalizes_modality(self):
        self.assertEqual(dedup_key(' case-1 ', 'lge'), ('case-1', 'LGE'))

    def test_dedup_key_requires_fields(self):
        with self.assertRaises(ValueError):
            dedup_key('', 'LGE')
        with self.assertRaises(ValueError):
            normalize_modality(None)

    def test_issue_types_trim_dedupe_and_keep_order(self):
        self.assertEqual(normalize_issue_types([' a ', 'b', 'a', None, '']), ('a', 'b'))
        self.assertEqual(normalize_issue_types('solo'), ('solo',))
        self.assertEqual(normalize_issue_types(None), ())

    def test_slice_indices_sorted_unique(self):
        self.assertEqual(normalize_slice_indices([3, 1, 2, 2, '4']), (1, 2, 3, 4))
        self.assertEqual(normalize_slice_indices(None), ())
        self.assertEqual(normalize_slice_indices([True, 'x', 2.0]), (2,))

    def test_slice_zero_is_preserved(self):
        self.assertEqual(normalize_slice_indices(0), (0,))

    def test_hash_validation(self):
        self.assertTrue(is_valid_hash('abcdef0123456789'))
        self.assertFalse(is_valid_hash(''))
        self.assertFalse(is_valid_hash('zzzzzzzz'))
        self.assertFalse(is_valid_hash(None))
        self.assertEqual(normalize_cache_hash(' ABCDEF0123456789 '), 'abcdef0123456789')
        self.assertIsNone(normalize_cache_hash('not-a-hash'))


class WriteSideDedupTests(unittest.TestCase):
    def test_first_row_is_canonical(self):
        gate = CaseModalityDedupHardGate()
        decision = gate.admit('case-1', 'LGE', ['a'], [1])
        self.assertEqual(decision.action, ACTION_CANONICAL)
        self.assertTrue(decision.canonical)
        self.assertFalse(decision.merged)
        self.assertEqual(decision.duplicate_records_after, 0)
        self.assertEqual(gate.state('case-1', 'LGE'), STATE_OPEN)

    def test_duplicates_merge_and_duplicate_count_is_zero(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('case-1', 'LGE', ['a', 'b'], [1])
        second = gate.admit('case-1', 'LGE', ['b', 'c'], [1, 2])
        third = gate.admit('case-1', 'LGE', ['d'], [3])
        self.assertEqual(second.action, ACTION_MERGED)
        self.assertEqual(third.action, ACTION_MERGED)
        self.assertEqual(gate.duplicate_record_count(), 0)
        self.assertEqual(len(gate.rows()), 1)
        record = gate.record('case-1', 'LGE')
        self.assertEqual(record['issue_types'], ('a', 'b', 'c', 'd'))
        self.assertEqual(record['slice_indices'], (1, 2, 3))

    def test_coverage_never_shrinks_when_merging(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('case-1', 'LGE', ['a', 'b'], [1])
        before = gate.issue_type_coverage()
        gate.admit('case-1', 'LGE', ['b', 'c'], [2])
        after = gate.issue_type_coverage()
        self.assertTrue(before.issubset(after))
        self.assertEqual(sorted(after), ['a', 'b', 'c'])

    def test_admit_many_folds_repeated_records(self):
        gate = CaseModalityDedupHardGate()
        decisions = gate.admit_many([
            {'case_id': 'c1', 'modality': 'lge', 'issue_type': 'a', 'slice_index': 0},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_type': 'b', 'slice_index': 1},
            {'case_id': 'c2', 'modality': 'MYO', 'issue_type': 'z'},
        ])
        self.assertEqual(
            [d.action for d in decisions],
            [ACTION_CANONICAL, ACTION_MERGED, ACTION_CANONICAL],
        )
        self.assertEqual(gate.duplicate_record_count(), 0)
        self.assertEqual(gate.record('c1', 'lge')['slice_indices'], (0, 1))
        self.assertEqual(gate.record('c1', 'lge')['issue_types'], ('a', 'b'))

    def test_replaying_the_same_batch_stays_at_zero_duplicates(self):
        gate = CaseModalityDedupHardGate()
        batch = [
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['a'], 'slice_indices': [1]},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['b'], 'slice_indices': [2]},
        ]
        gate.admit_many(batch)
        gate.admit_many(batch)
        self.assertEqual(gate.duplicate_record_count(), 0)
        self.assertEqual(len(gate.rows()), 1)
        self.assertEqual(sorted(gate.issue_type_coverage()), ['a', 'b'])

    def test_distinct_keys_stay_distinct(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'])
        gate.admit('c1', 'MYO', ['a'])
        gate.admit('c2', 'LGE', ['a'])
        self.assertEqual(len(gate.rows()), 3)
        self.assertEqual(gate.duplicate_record_count(), 0)


class ExtractorCacheGuardTests(unittest.TestCase):
    def test_valid_hash_blocks_new_extractor(self):
        gate = CaseModalityDedupHardGate()
        gate.register_cache_hash('c1', 'LGE', 'deadbeefdeadbeef')
        first = gate.resolve_extractor('c1', 'lge')
        second = gate.resolve_extractor('c1', 'LGE', 'deadbeefdeadbeef')
        self.assertEqual(first.action, EXTRACTOR_REUSE)
        self.assertEqual(second.action, EXTRACTOR_REUSE)
        self.assertTrue(first.reuse)
        self.assertFalse(first.created)
        self.assertEqual(gate.extractors_created, 0)
        self.assertEqual(gate.cache_reuses, 2)

    def test_invalid_hash_is_not_cached(self):
        gate = CaseModalityDedupHardGate()
        self.assertFalse(gate.register_cache_hash('c1', 'LGE', 'nope'))
        self.assertIsNone(gate.cache_hash('c1', 'LGE'))
        decision = gate.resolve_extractor('c1', 'LGE', 'nope')
        self.assertEqual(decision.action, EXTRACTOR_CREATE)
        self.assertTrue(decision.created)
        self.assertEqual(gate.extractors_created, 1)

    def test_invalid_hash_does_not_replace_valid_cache(self):
        gate = CaseModalityDedupHardGate()
        gate.register_cache_hash('c1', 'LGE', 'deadbeefdeadbeef')
        self.assertFalse(gate.register_cache_hash('c1', 'LGE', ''))
        self.assertEqual(gate.cache_hash('c1', 'LGE'), 'deadbeefdeadbeef')
        self.assertEqual(gate.resolve_extractor('c1', 'LGE').action, EXTRACTOR_REUSE)

    def test_cache_is_per_key(self):
        gate = CaseModalityDedupHardGate()
        gate.register_cache_hash('c1', 'LGE', 'deadbeefdeadbeef')
        self.assertEqual(gate.resolve_extractor('c1', 'LGE').action, EXTRACTOR_REUSE)
        self.assertEqual(gate.resolve_extractor('c1', 'MYO').action, EXTRACTOR_CREATE)
        self.assertEqual(gate.extractors_created, 1)


class DoneTerminalTests(unittest.TestCase):
    def _done_gate(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'], [1])
        self.assertTrue(gate.mark_validated('c1', 'LGE', True))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_VALIDATED)
        self.assertTrue(gate.close('c1', 'LGE'))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_DONE)
        return gate

    def test_validation_then_done(self):
        gate = self._done_gate()
        plan = gate.write_plan('c1', 'LGE')
        self.assertFalse(plan['accepts_writes'])
        self.assertEqual(plan['state'], STATE_DONE)

    def test_close_requires_validation(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'])
        self.assertFalse(gate.close('c1', 'LGE'))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_OPEN)

    def test_close_is_idempotent(self):
        gate = self._done_gate()
        self.assertFalse(gate.close('c1', 'LGE'))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_DONE)

    def test_done_rejects_further_writes(self):
        gate = self._done_gate()
        decision = gate.admit('c1', 'LGE', ['b'], [2])
        self.assertEqual(decision.action, ACTION_REJECTED)
        self.assertTrue(decision.closed)
        self.assertEqual(gate.rejected_writes, 1)
        self.assertEqual(gate.record('c1', 'LGE')['issue_types'], ('a',))

    def test_done_suppresses_dedup_proposal(self):
        gate = self._done_gate()
        proposal = gate.propose_dedup('c1', 'LGE')
        self.assertFalse(proposal.issued)
        self.assertEqual(gate.proposals_issued, 0)
        self.assertEqual(gate.proposals_suppressed, 1)

    def test_proposal_issued_at_most_once_before_done(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'], [1])
        gate.admit('c1', 'LGE', ['b'], [2])
        first = gate.propose_dedup('c1', 'LGE')
        second = gate.propose_dedup('c1', 'LGE')
        self.assertTrue(first.issued)
        self.assertFalse(second.issued)
        self.assertEqual(gate.proposals_issued, 1)

    def test_no_proposal_without_merged_duplicates(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'], [1])
        proposal = gate.propose_dedup('c1', 'LGE')
        self.assertFalse(proposal.issued)
        self.assertEqual(gate.proposals_issued, 0)

    def test_reopen_requires_reason_and_is_explicit(self):
        gate = self._done_gate()
        self.assertFalse(gate.reopen('c1', 'LGE', '   '))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_DONE)
        self.assertTrue(gate.reopen('c1', 'LGE', 'new evidence arrived'))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_OPEN)
        self.assertEqual(gate.reopens, 1)
        self.assertEqual(gate.admit('c1', 'LGE', ['b'], [2]).action, ACTION_MERGED)

    def test_failed_validation_keeps_key_open(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'])
        self.assertFalse(gate.mark_validated('c1', 'LGE', False))
        self.assertEqual(gate.state('c1', 'LGE'), STATE_OPEN)


class HistoryConvergenceTests(unittest.TestCase):
    def test_proposal_kinds_normalizes_dedup_markers(self):
        self.assertEqual(proposal_kinds([{'kind': 'dedup'}]), ('dedup',))
        self.assertEqual(proposal_kinds([{'type': 'DEDUP_ROLLUP'}]), ('dedup',))
        self.assertEqual(proposal_kinds([{'kind': 'routing'}]), ('routing',))
        self.assertEqual(proposal_kinds([]), ())

    def test_dedup_proposals_in_history(self):
        history = [
            {'proposals': [{'kind': 'dedup'}, {'kind': 'routing'}]},
            {'proposals': [{'kind': 'dedup'}]},
            {'proposals': []},
        ]
        self.assertEqual(dedup_proposals_in_history(history), 2)

    def test_three_clean_runs_converge(self):
        history = [
            {'proposals': [{'kind': 'dedup'}]},
            {'proposals': []},
            {'proposals': []},
            {'proposals': []},
        ]
        self.assertEqual(clean_run_streak(history), 3)
        verdict = convergence_verdict(history)
        self.assertTrue(verdict['converged'])
        self.assertEqual(verdict['action'], 'STOP_CLASS')
        self.assertFalse(verdict['escalate_to_human'])
        self.assertEqual(verdict['window'], DEFAULT_CONVERGENCE_WINDOW)

    def test_two_clean_runs_do_not_converge(self):
        history = [{'proposals': [{'kind': 'dedup'}]}, {'proposals': []}, {'proposals': []}]
        verdict = convergence_verdict(history)
        self.assertFalse(verdict['converged'])
        self.assertEqual(verdict['action'], 'CONTINUE')
        self.assertTrue(verdict['escalate_to_human'])

    def test_residual_dedup_breaks_streak(self):
        history = [
            {'proposals': []},
            {'proposals': []},
            {'proposals': [{'kind': 'dedup'}]},
            {'proposals': []},
        ]
        self.assertEqual(clean_run_streak(history), 1)
        self.assertFalse(convergence_verdict(history)['converged'])

    def test_window_guard(self):
        with self.assertRaises(ValueError):
            convergence_verdict([], window=0)

    def test_suppressed_proposals_are_not_counted(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'])
        gate.mark_validated('c1', 'LGE', True)
        gate.close('c1', 'LGE')
        suppressed = gate.propose_dedup('c1', 'LGE')
        history = [{'proposals': [suppressed]}]
        self.assertEqual(dedup_proposals_in_history(history), 0)
        self.assertEqual(clean_run_streak(history), 1)

    def test_done_gate_produces_clean_history(self):
        gate = CaseModalityDedupHardGate()
        history = []
        gate.admit('c1', 'LGE', ['a'], [1])
        gate.admit('c1', 'LGE', ['b'], [2])
        gate.mark_validated('c1', 'LGE', True)
        gate.close('c1', 'LGE')
        for _ in range(DEFAULT_CONVERGENCE_WINDOW):
            history.append({'proposals': [gate.propose_dedup('c1', 'LGE')]})
        self.assertEqual(dedup_proposals_in_history(history), 0)
        self.assertTrue(convergence_verdict(history)['converged'])


class RecordRollupTests(unittest.TestCase):
    def test_duplicate_record_count_module_level(self):
        records = [
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['a']},
            {'case_id': 'c1', 'modality': 'lge', 'issue_types': ['b']},
            {'case_id': 'c2', 'modality': 'MYO', 'issue_types': ['a']},
        ]
        self.assertEqual(duplicate_record_count(records), 1)

    def test_dedup_records_folds_originals(self):
        records = [
            {'case_id': 'c1', 'modality': 'lge', 'issue_type': 'a', 'slice_index': 3},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_type': 'b', 'slice_index': 1},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['a', 'c'],
             'slice_indices': [1, 5]},
        ]
        merged = dedup_records(records)
        self.assertEqual(len(merged), 1)
        entry = merged[('c1', 'LGE')]
        self.assertEqual(entry['records'], 3)
        self.assertEqual(entry['issue_types'], ['a', 'b', 'c'])
        self.assertEqual(entry['slice_indices'], [1, 3, 5])
        self.assertEqual(duplicate_record_count(records), 2)

    def test_issue_type_coverage_module_level(self):
        records = [
            {'case_id': 'c1', 'modality': 'LGE', 'issue_type': 'a'},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['b']},
        ]
        self.assertEqual(sorted(issue_type_coverage(records)), ['a', 'b'])

    def test_dedup_invariants_report_no_duplicates_after_gate(self):
        gate = CaseModalityDedupHardGate()
        gate.admit('c1', 'LGE', ['a'], [1])
        gate.admit('c1', 'LGE', ['b'], [2])
        report = dedup_invariants(gate.rows())
        self.assertTrue(report['no_duplicates'])
        self.assertEqual(report['duplicate_records'], 0)
        self.assertEqual(report['unique_keys'], 1)
        self.assertEqual(sorted(report['issue_type_coverage']), ['a', 'b'])
        self.assertEqual(report['slice_indices'][('c1', 'LGE')], (1, 2))

    def test_dedup_invariants_flags_raw_duplicates(self):
        raw = [
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['a']},
            {'case_id': 'c1', 'modality': 'LGE', 'issue_types': ['a']},
        ]
        report = dedup_invariants(raw)
        self.assertFalse(report['no_duplicates'])
        self.assertEqual(report['duplicate_records'], 1)


if __name__ == '__main__':
    unittest.main()
