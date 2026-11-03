'''Offline unit tests for the audit coverage ledger and write de-dup gate.'''

import unittest

from experiments.audit_coverage_ledger import (
    ACCEPTANCE_THRESHOLDS,
    DEFAULT_PIPELINE,
    ISOLATED_PIPELINE,
    AuditCoverageLedger,
    DedupWriteGate,
    cache_hash,
    merge_issue_types,
    retention_rates,
)


AUDITED_CLASSES = (
    'cmr_myo_small_area',
    'cmr_myo_excess_components',
    'cmr_myo_missing_temporal_hole',
    'lge_mi_fragmentation',
    'lge_myo_topology',
    'cmr_centroid_drift',
    'cmr_myo_border_contact',
    'cmr_myo_touches_border',
    'review_centroid_outlier',
)


def _row(slice_index, issue_type, case_id='c1', modality='LGE'):
    return {
        'case_id': case_id,
        'modality': modality,
        'slice_index': slice_index,
        'issue_type': issue_type,
    }


class HelperTests(unittest.TestCase):
    def test_cache_hash_is_deterministic_and_order_insensitive(self):
        self.assertEqual(cache_hash({'a': 1, 'b': 2}), cache_hash({'b': 2, 'a': 1}))
        self.assertNotEqual(cache_hash({'a': 1}), cache_hash({'a': 2}))
        self.assertEqual(len(cache_hash('anything')), 64)

    def test_merge_issue_types_dedups_and_keeps_first_seen_order(self):
        self.assertEqual(merge_issue_types('a', ['b', 'a'], None), ('a', 'b'))
        self.assertEqual(merge_issue_types(), ())
        self.assertEqual(merge_issue_types([]), ())
        self.assertEqual(merge_issue_types(['x'], ['y'], {'issue_type': 'x'}), ('x', 'y'))


class WriteGateTests(unittest.TestCase):
    def test_gate_is_pass_through_outside_isolated_pipeline(self):
        gate = DedupWriteGate(DEFAULT_PIPELINE)
        rows = [_row(3, 'scar'), _row(3, 'edema')]
        result = gate.write(rows)
        self.assertFalse(result.armed)
        self.assertEqual(len(result.written), 2)
        self.assertEqual(result.blocked_rows, 0)
        self.assertEqual(result.duplicates_written, 0)
        self.assertEqual(gate.duplicate_write_rate, 0.0)
        rates = retention_rates(rows, result)
        self.assertEqual(rates['slice_index_retention'], 1.0)
        self.assertEqual(rates['issue_type_merge_retention'], 1.0)

    def test_gate_arms_only_on_isolated_pipeline(self):
        self.assertTrue(DedupWriteGate(ISOLATED_PIPELINE).armed)
        self.assertFalse(DedupWriteGate(DEFAULT_PIPELINE).armed)
        self.assertFalse(DedupWriteGate('staging').armed)

    def test_armed_gate_merges_duplicate_key_and_keeps_everything(self):
        gate = DedupWriteGate(ISOLATED_PIPELINE)
        rows = [
            _row(3, 'scar'),
            _row(3, 'edema'),
            _row(4, 'scar'),
            _row(4, 'scar'),
        ]
        result = gate.write(rows)
        self.assertTrue(result.armed)
        self.assertEqual(len(result.written), 2)
        self.assertEqual(result.written[0]['issue_type'], ['scar', 'edema'])
        self.assertEqual(result.written[1]['issue_type'], ['scar'])
        self.assertEqual(result.blocked_rows, 2)
        self.assertEqual(result.merged_rows, 2)
        self.assertEqual(result.slice_indices, (3, 4))
        self.assertEqual(result.issue_types, ('scar', 'edema'))
        self.assertEqual(result.duplicate_write_rate, 0.0)
        self.assertEqual(gate.duplicate_write_rate, 0.0)

        rates = retention_rates(rows, result)
        self.assertEqual(rates['slice_index_retention'], 1.0)
        self.assertEqual(rates['issue_type_merge_retention'], 1.0)

    def test_armed_gate_separates_different_case_and_modality(self):
        gate = DedupWriteGate(ISOLATED_PIPELINE)
        rows = [
            _row(3, 'scar', case_id='c1', modality='LGE'),
            _row(3, 'scar', case_id='c1', modality='CINE'),
            _row(3, 'scar', case_id='c2', modality='LGE'),
        ]
        result = gate.write(rows)
        self.assertEqual(len(result.written), 3)
        self.assertEqual(result.blocked_rows, 0)

    def test_armed_gate_uses_fallbacks_for_case_and_modality(self):
        gate = DedupWriteGate(ISOLATED_PIPELINE)
        rows = [
            {'slice_index': 1, 'issue_type': 'scar'},
            {'slice_index': 1, 'issue_type': 'edema'},
        ]
        result = gate.write(rows, case_id='c9', modality='LGE')
        self.assertEqual(len(result.written), 1)
        self.assertEqual(result.written[0]['case_id'], 'c9')
        self.assertEqual(result.written[0]['modality'], 'LGE')
        self.assertEqual(result.written[0]['issue_type'], ['scar', 'edema'])


class LedgerTests(unittest.TestCase):
    def test_declared_class_is_skipped_and_ratio_tracks_hits(self):
        ledger = AuditCoverageLedger(ISOLATED_PIPELINE)
        for name in AUDITED_CLASSES:
            ledger.declare_audited_class(name)

        hits = [ledger.skip_audited_class(name) for name in AUDITED_CLASSES]
        self.assertTrue(all(hits))
        # re-auditing an already covered class is still a hit, not new work
        repeats = [ledger.skip_audited_class(name) for name in AUDITED_CLASSES]
        self.assertTrue(all(repeats))
        # a class that was never audited must not be reported as covered
        self.assertFalse(ledger.skip_audited_class('brand_new_class'))

        ratio = ledger.skip_ratio('class')
        self.assertGreaterEqual(ratio, ACCEPTANCE_THRESHOLDS['covered_class_skip_ratio'])
        self.assertAlmostEqual(ratio, 18.0 / 19.0, places=6)
        self.assertEqual(ledger.metrics()['covered_class_skip_ratio'], ratio)

    def test_cache_hash_roundtrip_and_mismatch(self):
        ledger = AuditCoverageLedger()
        digest = ledger.record_cache_hash('rollup', {'rows': 3})
        self.assertEqual(digest, cache_hash({'rows': 3}))
        self.assertEqual(ledger.lookup_cache_hash('rollup'), digest)
        self.assertTrue(ledger.skip_cached('rollup', {'rows': 3}))
        self.assertFalse(ledger.skip_cached('rollup', {'rows': 4}))
        self.assertFalse(ledger.skip_cached('never_recorded'))
        self.assertEqual(ledger.skip_ratio('hash'), 1.0 / 3.0)

    def test_cache_hash_hit_is_content_addressed(self):
        ledger = AuditCoverageLedger()
        digest = ledger.record_cache_hash('rollup-key', {'v': 1})
        # the same content under a different key is still a hit
        self.assertTrue(ledger.skip_cached('other-key', {'v': 1}))
        self.assertEqual(ledger.lookup_cache_hash('other-key'), None)
        self.assertTrue(ledger.has_cache_hash(digest))
        # different content stays a miss even with a matching key grammar
        self.assertFalse(ledger.skip_cached('rollup-key', {'v': 2}))
        self.assertEqual(ledger.skip_ratio('hash'), 1.0 / 2.0)

    def test_mark_done_requires_verification(self):
        ledger = AuditCoverageLedger()
        self.assertFalse(ledger.mark_done('gate-8', verified=False))
        self.assertFalse(ledger.is_closed('gate-8'))
        outcome = ledger.propose('gate-8', issue_type='scar')
        self.assertFalse(outcome['skipped'])
        self.assertIsNotNone(outcome['proposal'])

        self.assertTrue(ledger.mark_done('gate-8', verified=True, evidence={'run': 1}))
        self.assertTrue(ledger.is_closed('gate-8'))
        self.assertTrue(ledger.is_done('gate-8'))

    def test_closed_item_is_never_re_proposed(self):
        ledger = AuditCoverageLedger()
        ledger.mark_done('gate-7', verified=True, evidence={'metric': 'ok'})
        first = ledger.propose('gate-7', issue_type='scar')
        second = ledger.propose('gate-7', issue_type='edema')
        for outcome in (first, second):
            self.assertTrue(outcome['skipped'])
            self.assertEqual(outcome['reason'], 'closed_gate')
            self.assertIsNone(outcome['proposal'])
        self.assertEqual(ledger.closed_repeat_attempts, 2)
        self.assertEqual(ledger.metrics()['closed_repeat_proposals'], 0)
        self.assertEqual(ledger.closed_repeat_proposals, 0)

    def test_proposal_reasons_for_covered_class_and_cache_hit(self):
        ledger = AuditCoverageLedger(ISOLATED_PIPELINE)
        ledger.declare_audited_class('cmr_myo_small_area')
        ledger.record_cache_hash('rollup-key', {'v': 1})

        covered = ledger.propose('cmr_myo_small_area', issue_type='small_area')
        self.assertTrue(covered['skipped'])
        self.assertEqual(covered['reason'], 'covered_class')

        cached = ledger.propose('cmr_other', payload={'v': 1})
        self.assertTrue(cached['skipped'])
        self.assertEqual(cached['reason'], 'cache_hit')

        fresh = ledger.propose('cmr_new', issue_type=['scar', 'scar'], case_id='c1', modality='LGE')
        self.assertFalse(fresh['skipped'])
        self.assertEqual(fresh['proposal']['issue_type'], ['scar'])

    def test_snapshot_records_classes_gates_and_hashes(self):
        ledger = AuditCoverageLedger(ISOLATED_PIPELINE)
        ledger.declare_audited_class('cmr_myo_small_area')
        ledger.mark_done('gate-1', verified=True)
        ledger.record_cache_hash('k', {'v': 1})
        snap = ledger.snapshot()
        self.assertEqual(len(snap['audited_classes']), 1)
        self.assertEqual(len(snap['closed_gates']), 1)
        self.assertIn('k', snap['cache_hashes'])
        self.assertTrue(snap['write_gate_armed'])

    def test_end_to_end_flow_meets_acceptance(self):
        ledger = AuditCoverageLedger(ISOLATED_PIPELINE)
        for name in AUDITED_CLASSES:
            ledger.declare_audited_class(name)
        for name in AUDITED_CLASSES:
            self.assertTrue(ledger.skip_audited_class(name))
        self.assertFalse(ledger.skip_audited_class('unknown_class'))

        rows = [
            _row(1, 'scar'),
            _row(1, 'edema'),
            _row(1, 'scar'),
            _row(2, 'scar'),
            _row(2, 'scar'),
        ]
        result = ledger.write_rows(rows)
        self.assertEqual(len(result.written), 2)
        self.assertEqual(result.duplicate_write_rate, 0.0)

        self.assertTrue(ledger.mark_done('cmr_myo_small_area_gate', verified=True))
        self.assertTrue(ledger.propose('cmr_myo_small_area_gate')['skipped'])

        metrics = ledger.metrics()
        self.assertEqual(metrics['duplicate_write_rate'], 0.0)
        self.assertEqual(metrics['slice_index_retention'], 1.0)
        self.assertEqual(metrics['issue_type_merge_retention'], 1.0)
        self.assertEqual(metrics['closed_repeat_proposals'], 0)
        self.assertGreaterEqual(
            metrics['covered_class_skip_ratio'],
            ACCEPTANCE_THRESHOLDS['covered_class_skip_ratio'],
        )

        report = ledger.acceptance_report()
        self.assertEqual(set(report), {
            'duplicate_write_rate',
            'slice_index_retention',
            'issue_type_merge_retention',
            'closed_repeat_proposals',
            'covered_class_skip_ratio',
        })
        for name, entry in report.items():
            self.assertTrue(entry['passed'], msg=name)
        self.assertTrue(ledger.meets_acceptance())

    def test_default_pipeline_does_not_arm_gate(self):
        ledger = AuditCoverageLedger()
        self.assertFalse(ledger.gate.armed)
        self.assertFalse(ledger.metrics()['write_gate_armed'])
        rows = [_row(1, 'scar'), _row(1, 'edema')]
        result = ledger.write_rows(rows)
        self.assertEqual(len(result.written), 2)
        self.assertEqual(result.duplicate_write_rate, 0.0)


if __name__ == '__main__':
    unittest.main()
