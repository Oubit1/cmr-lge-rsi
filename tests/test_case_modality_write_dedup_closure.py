'''Offline unit tests for case-modality write dedup closure.'''

import unittest

from experiments.case_modality_write_dedup_closure import (
    RULE_VERSION,
    CaseModalityWriteDedupClosure,
    apply_case_modality_write_dedup_gate,
    canonical_fingerprint,
    validate_gate_result,
)


class CaseModalityWriteDedupClosureTests(unittest.TestCase):
    def _rows(self):
        return [
            {'case_id': 'case-1', 'modality': 'LGE', 'issue_type': 'myo_excess_components', 'slice_index': 3},
            {'case_id': 'case-1', 'modality': 'lge', 'issue_type': 'myo_small_area', 'slice_index': 4},
            {'case_id': 'case-2', 'modality': 'LGE', 'issue_type': ['myo_empty', 'myo_excess_components'], 'slice_index': [1, 2]},
            {'case_id': 'case-2', 'modality': 'lge', 'issue_type': 'myo_empty', 'slice_index': 2},
        ]

    def test_duplicate_rows_are_rejected_and_issue_types_and_slices_are_merged(self):
        result = apply_case_modality_write_dedup_gate(self._rows())
        self.assertEqual(result.metrics['duplicate_rows_rejected'], 2)
        self.assertEqual(result.metrics['unique_input_keys'], 2)
        self.assertEqual(result.metrics['unique_output_keys'], 2)
        self.assertEqual(result.metrics['unique_key_loss'], 0)
        by_key = {(r['case_id'], r['modality'].lower()): r for r in result.deduped}
        case_1 = by_key[('case-1', 'lge')]
        self.assertEqual(case_1['issue_type'], ('myo_excess_components', 'myo_small_area'))
        self.assertEqual(case_1['issue_types'], ('myo_excess_components', 'myo_small_area'))
        self.assertEqual(case_1['slice_index'], (3, 4))
        self.assertEqual(case_1['slice_indices'], (3, 4))
        case_2 = by_key[('case-2', 'lge')]
        self.assertEqual(case_2['issue_type'], ('myo_empty', 'myo_excess_components'))
        self.assertEqual(case_2['slice_index'], (1, 2))
        self.assertEqual(result.metrics['positive_issue_type_retention'], 1.0)
        self.assertEqual(result.metrics['slice_index_retention'], 1.0)
        validation = validate_gate_result(result)
        self.assertTrue(validation.passed, validation.failures)

    def test_invalid_rows_are_rejected_without_crashing(self):
        result = apply_case_modality_write_dedup_gate([
            {'case_id': '', 'modality': 'LGE'},
            {'case_id': 'case-3'},
            {'case_id': 'case-3', 'modality': 'LGE', 'issue_type': 'ok', 'slice_index': 0},
        ])
        reasons = [r.reason for r in result.rejected]
        self.assertIn('missing_case_id_or_modality', reasons)
        self.assertEqual(result.metrics['invalid_rows_rejected'], 2)
        self.assertEqual(result.metrics['unique_key_loss'], 0)
        self.assertFalse(validate_gate_result(result).passed)

    def test_validation_marks_done_and_unchanged_hash_or_version_does_not_repropose(self):
        closure = CaseModalityWriteDedupClosure()
        first = closure.evaluate(self._rows())
        self.assertEqual(first.status, 'APPLIED')
        self.assertTrue(first.done_marked)
        self.assertIsNotNone(first.proposal)
        self.assertTrue(closure.done)
        second = closure.evaluate(self._rows())
        self.assertEqual(second.status, 'NOOP')
        self.assertTrue(second.skipped)
        self.assertIsNone(second.proposal)
        changed_version = closure.evaluate(self._rows(), rule_version='v2')
        self.assertEqual(changed_version.status, 'APPLIED')
        self.assertIsNotNone(changed_version.proposal)

    def test_changed_cache_hash_reproposes(self):
        closure = CaseModalityWriteDedupClosure()
        first = closure.evaluate(self._rows(), cache_hash='hash-a')
        self.assertEqual(first.status, 'APPLIED')
        second = closure.evaluate(self._rows(), cache_hash='hash-b')
        self.assertEqual(second.status, 'APPLIED')
        self.assertIsNotNone(second.proposal)
        third = closure.evaluate(self._rows(), cache_hash='hash-b')
        self.assertEqual(third.status, 'NOOP')
        self.assertIsNone(third.proposal)

    def test_fingerprint_is_stable_for_equivalent_issue_type_order(self):
        rows_a = [
            {'case_id': 'c', 'modality': 'LGE', 'issue_type': ['b', 'a'], 'slice_index': [2, 1]},
        ]
        rows_b = [
            {'case_id': 'c', 'modality': 'lge', 'issue_type': ['a', 'b'], 'slice_index': [1, 2]},
        ]
        self.assertEqual(canonical_fingerprint(rows_a), canonical_fingerprint(rows_b))

    def test_rule_version_constant_is_non_empty(self):
        self.assertTrue(RULE_VERSION)


if __name__ == '__main__':
    unittest.main()
