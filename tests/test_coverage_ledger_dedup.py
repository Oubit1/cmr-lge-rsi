import unittest

from experiments.coverage_ledger_dedup import (
    CoverageLedger,
    DuplicateCaseModalityError,
    WriteRecord,
    merge_case_modality,
    merge_case_modality_strict,
    replay_runs,
)


class CoverageLedgerTests(unittest.TestCase):
    def test_marks_done_and_skips_only_unchanged_exact_hit(self):
        ledger = CoverageLedger()
        ledger.mark_done('myo_empty', 'LGE', 'hash1', 'v1')

        self.assertTrue(ledger.should_skip_audit('myo_empty', 'LGE', 'hash1', 'v1'))
        self.assertFalse(ledger.should_skip_audit('myo_empty', 'LGE', 'hash2', 'v1'))
        self.assertFalse(ledger.should_skip_audit('myo_empty', 'LGE', 'hash1', 'v2'))
        self.assertFalse(ledger.should_skip_audit('myo_small', 'LGE', 'hash1', 'v1'))
        self.assertFalse(ledger.should_skip_audit('myo_empty', 'LGE', 'hash1', 'v1 '))

    def test_propose_returns_record_once_then_skips(self):
        ledger = CoverageLedger()
        first = ledger.propose_audit('myo_empty', 'LGE', 'hash1', 'v1')
        second = ledger.propose_audit('myo_empty', 'LGE', 'hash1', 'v1')

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertTrue(ledger.is_covered('myo_empty', 'LGE', 'hash1', 'v1'))

    def test_validation_blocks_new_proposals(self):
        ledger = CoverageLedger()
        ledger.mark_validated()

        self.assertIsNone(ledger.propose_audit('myo_empty', 'LGE', 'hash1', 'v1'))
        self.assertFalse(ledger.is_covered('myo_empty', 'LGE', 'hash1', 'v1'))
        self.assertTrue(ledger.validated)


class WriteDedupTests(unittest.TestCase):
    def test_merge_preserves_slice_index_and_unions_issue_types(self):
        records = [
            WriteRecord('case1', 'LGE', ('myo_empty',), 3),
            WriteRecord('case1', 'LGE', ('myo_small',), 5),
            WriteRecord('case2', 'LGE', ('myo_empty',), 7),
        ]

        merged, rejected = merge_case_modality(records)

        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(merged), 2)

        case1 = [r for r in merged if r.case_id == 'case1'][0]
        self.assertEqual(case1.slice_index, 3)
        self.assertEqual(case1.issue_types, ('myo_empty', 'myo_small'))

        case2 = [r for r in merged if r.case_id == 'case2'][0]
        self.assertEqual(case2.slice_index, 7)
        self.assertEqual(case2.issue_types, ('myo_empty',))

    def test_merged_output_has_no_duplicate_case_modality(self):
        records = [
            WriteRecord('case1', 'LGE', ('a',), 1),
            WriteRecord('case1', 'LGE', ('b',), 2),
            WriteRecord('case1', 'LGE', ('c',), 3),
            WriteRecord('case2', 'LGE', ('a',), 4),
        ]

        merged, rejected = merge_case_modality(records)

        self.assertEqual(len(merged), 2)
        self.assertEqual(len(rejected), 2)
        keys = [(r.case_id, r.modality) for r in merged]
        self.assertEqual(len(keys), len(set(keys)))

    def test_strict_merge_raises_on_duplicate(self):
        records = [
            WriteRecord('case1', 'LGE', ('a',), 1),
            WriteRecord('case1', 'LGE', ('b',), 2),
        ]

        with self.assertRaises(DuplicateCaseModalityError):
            merge_case_modality_strict(records)

    def test_strict_merge_accepts_unique_rows(self):
        records = [
            WriteRecord('case1', 'LGE', ('a',), 1),
            WriteRecord('case2', 'LGE', ('b',), 2),
        ]

        merged = merge_case_modality_strict(records)

        self.assertEqual(len(merged), 2)

    def test_issue_type_recall_does_not_drop_after_merge(self):
        records = [
            WriteRecord('case1', 'LGE', ('a',), 1),
            WriteRecord('case1', 'LGE', ('b',), 2),
            WriteRecord('case1', 'LGE', ('c',), 3),
        ]

        merged, rejected = merge_case_modality(records)

        original_issue_types = set()
        for record in records:
            original_issue_types.update(record.issue_types)

        merged_issue_types = set()
        for record in merged:
            merged_issue_types.update(record.issue_types)

        self.assertEqual(original_issue_types, merged_issue_types)
        self.assertEqual(len(rejected), 2)


class ReplayTests(unittest.TestCase):
    def test_replay_skips_at_least_five_covered_classes_on_run38_45(self):
        ledger = CoverageLedger()

        runs = [
            {'issue_type': 'myo_empty', 'modality': 'LGE', 'cache_hash': 'h1', 'rule_version': 'v1'},
            {'issue_type': 'myo_small', 'modality': 'LGE', 'cache_hash': 'h2', 'rule_version': 'v1'},
            {'issue_type': 'myo_border', 'modality': 'LGE', 'cache_hash': 'h3', 'rule_version': 'v1'},
            {'issue_type': 'lge_mi_multi', 'modality': 'LGE', 'cache_hash': 'h4', 'rule_version': 'v1'},
            {'issue_type': 'lge_mi_frag', 'modality': 'LGE', 'cache_hash': 'h5', 'rule_version': 'v1'},
            {'issue_type': 'myo_excess', 'modality': 'LGE', 'cache_hash': 'h6', 'rule_version': 'v1'},
            {'issue_type': 'myo_temporal', 'modality': 'LGE', 'cache_hash': 'h7', 'rule_version': 'v1'},
            {'issue_type': 'centroid_drift', 'modality': 'LGE', 'cache_hash': 'h8', 'rule_version': 'v1'},
        ]

        for run in runs[:5]:
            ledger.mark_done(run['issue_type'], run['modality'], run['cache_hash'], run['rule_version'])

        result = replay_runs(runs, ledger)

        self.assertGreaterEqual(result['skipped'], 5)
        self.assertEqual(result['skipped'], 5)
        self.assertEqual(len(result['skipped_keys']), 5)
        self.assertEqual(result['proposed'], 3)

    def test_replay_does_not_propose_after_validation(self):
        ledger = CoverageLedger()
        ledger.mark_validated()

        runs = [
            {'issue_type': 'myo_empty', 'modality': 'LGE', 'cache_hash': 'h1', 'rule_version': 'v1'},
        ]

        result = replay_runs(runs, ledger)

        self.assertEqual(result['skipped'], 0)
        self.assertEqual(result['proposed'], 0)


if __name__ == '__main__':
    unittest.main()
