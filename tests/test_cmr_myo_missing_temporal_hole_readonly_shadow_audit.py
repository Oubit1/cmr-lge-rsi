'''Offline unit tests for the bounded read-only missing-temporal-hole audit.

No datasets, no network and no cached artifacts are required: every record is
constructed in-process, so the suite runs anywhere.
'''

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cmr_myo_missing_temporal_hole_readonly_shadow_audit import (  # noqa: E402
    ALLOWED_HASH_MISMATCH_EXTRACTORS,
    AUDIT_POSITIVE_TARGET,
    CACHE_HIT_MIN,
    CACHED_RUN_IDS,
    CONSISTENCY_MIN,
    CONTROL_TARGET,
    FPR_MAX,
    PPV_MIN,
    READ_ONLY,
    RUN9_B1_EXTRACTOR,
    CacheStats,
    CaseRecord,
    HashMismatchRegistry,
    SliceRecord,
    compute_consistency,
    compute_control_fpr,
    compute_ppv,
    is_reused_run,
    reextract_on_hash_mismatch,
    run_shadow_audit,
    validate_reused_run,
)


def make_positives(total=30, true_positive=24):
    '''Audit positives: all flagged, the first ``true_positive`` are truly flagged.'''
    return [
        CaseRecord(
            case_id='pos-{0:03d}'.format(index),
            flag=True,
            shadow_reference=index < true_positive,
            cohort='positive',
        )
        for index in range(total)
    ]


def make_controls(total=10, false_positive=2):
    '''cmr controls: the first ``false_positive`` are incorrectly flagged.'''
    return [
        CaseRecord(
            case_id='ctl-{0:03d}'.format(index),
            flag=index < false_positive,
            shadow_reference=False,
            cohort='control',
        )
        for index in range(total)
    ]


def make_slices(total=10, agreed=9):
    '''Slice-level records: the first ``agreed`` agree with shadow continuity.'''
    return [
        SliceRecord(
            case_id='pos-{0:03d}'.format(index),
            slice_id='s-{0:03d}'.format(index),
            flag=True,
            shadow_continuity_support=index < agreed,
        )
        for index in range(total)
    ]


def stats_with(hits, misses):
    stats = CacheStats()
    for hit in [True] * hits + [False] * misses:
        stats.register_access(hit)
    return stats


class ReadOnlyContractTest(unittest.TestCase):
    def test_read_only_flag(self):
        self.assertTrue(READ_ONLY)

    def test_threshold_constants(self):
        self.assertEqual(PPV_MIN, 0.7)
        self.assertEqual(FPR_MAX, 0.3)
        self.assertEqual(CONSISTENCY_MIN, 0.8)
        self.assertEqual(CACHE_HIT_MIN, 0.9)

    def test_cohort_targets(self):
        self.assertEqual(AUDIT_POSITIVE_TARGET, 30)
        self.assertEqual(CONTROL_TARGET, 10)


class CachedRunTest(unittest.TestCase):
    def test_all_cached_runs_are_allowed(self):
        for run_id in CACHED_RUN_IDS:
            self.assertTrue(is_reused_run(run_id))
            validate_reused_run(run_id)

    def test_expected_cached_run_ids(self):
        self.assertEqual(
            set(CACHED_RUN_IDS),
            {
                'run-4',
                'run-5',
                'run-9',
                'run-13',
                'run-14',
                'run-15',
                'run-16',
                'run-17',
                'run-18',
            },
        )

    def test_uncached_run_is_rejected(self):
        for run_id in ('run-1', 'run-3', 'run-19', 'run-9b', ''):
            self.assertFalse(is_reused_run(run_id))
            with self.assertRaises(ValueError):
                validate_reused_run(run_id)


class MetricMathTest(unittest.TestCase):
    def test_ppv(self):
        self.assertAlmostEqual(compute_ppv(make_positives(30, 24)), 0.8, places=9)

    def test_ppv_without_flagged_cases_is_zero(self):
        records = [CaseRecord('p0', False, True, 'positive')]
        self.assertEqual(compute_ppv(records), 0.0)

    def test_ppv_counts_only_flagged_cases(self):
        records = [
            CaseRecord('p0', True, True, 'positive'),
            CaseRecord('p1', True, False, 'positive'),
            CaseRecord('p2', False, True, 'positive'),
        ]
        self.assertAlmostEqual(compute_ppv(records), 0.5, places=9)

    def test_control_fpr(self):
        self.assertAlmostEqual(compute_control_fpr(make_controls(10, 2)), 0.2, places=9)

    def test_control_fpr_empty_is_zero(self):
        self.assertEqual(compute_control_fpr([]), 0.0)

    def test_consistency(self):
        self.assertAlmostEqual(compute_consistency(make_slices(10, 9)), 0.9, places=9)

    def test_consistency_empty_is_zero(self):
        self.assertEqual(compute_consistency([]), 0.0)

    def test_consistency_counts_both_directions(self):
        records = [
            SliceRecord('c0', 's0', True, True),
            SliceRecord('c0', 's1', True, False),
            SliceRecord('c0', 's2', False, False),
        ]
        self.assertAlmostEqual(compute_consistency(records), 2.0 / 3.0, places=9)


class CacheStatsTest(unittest.TestCase):
    def test_hit_rate(self):
        stats = stats_with(9, 1)
        self.assertEqual(stats.accesses, 10)
        self.assertAlmostEqual(stats.hit_rate, 0.9, places=9)

    def test_hit_rate_without_accesses_is_zero(self):
        self.assertEqual(CacheStats().hit_rate, 0.0)

    def test_merge_sums_counts(self):
        left = CacheStats(hits=5, misses=1)
        left.merge(CacheStats(hits=4, misses=0))
        self.assertEqual((left.hits, left.misses), (9, 1))
        self.assertAlmostEqual(left.hit_rate, 0.9, places=9)


class HashMismatchTest(unittest.TestCase):
    def test_run9_b1_is_the_only_allowed_extractor(self):
        self.assertEqual(ALLOWED_HASH_MISMATCH_EXTRACTORS, {RUN9_B1_EXTRACTOR})

    def test_registry_accepts_run9_b1(self):
        registry = HashMismatchRegistry()
        record = registry.register('c0', 's0', 'aaa', 'bbb', RUN9_B1_EXTRACTOR)
        self.assertEqual(record.extractor_id, RUN9_B1_EXTRACTOR)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.case_ids(), ('c0',))

    def test_registry_defaults_to_run9_b1(self):
        registry = HashMismatchRegistry()
        record = registry.register('c1', 's1', 'aaa', 'bbb')
        self.assertEqual(record.extractor_id, RUN9_B1_EXTRACTOR)

    def test_registry_rejects_other_extractors(self):
        registry = HashMismatchRegistry()
        for extractor in ('run-4:B1', 'run-13:legacy', 'run-9:B2', ''):
            with self.assertRaises(ValueError):
                registry.register('c0', 's0', 'aaa', 'bbb', extractor)
        self.assertEqual(len(registry), 0)

    def test_registry_is_append_only(self):
        registry = HashMismatchRegistry()
        first = registry.register('c0', 's0', 'aaa', 'bbb')
        second = registry.register('c1', 's1', 'ccc', 'ddd')
        self.assertEqual(list(registry), [first, second])

    def test_reextract_skips_matching_hashes(self):
        registry = HashMismatchRegistry()
        self.assertIsNone(
            reextract_on_hash_mismatch(registry, 'c0', 's0', 'aaa', 'aaa')
        )
        self.assertEqual(len(registry), 0)

    def test_reextract_registers_mismatch_via_run9_b1(self):
        registry = HashMismatchRegistry()
        record = reextract_on_hash_mismatch(registry, 'c0', 's0', 'aaa', 'bbb')
        self.assertIsNotNone(record)
        self.assertEqual(record.extractor_id, RUN9_B1_EXTRACTOR)
        self.assertEqual(len(registry), 1)

    def test_reextract_rejects_foreign_extractor(self):
        registry = HashMismatchRegistry()
        with self.assertRaises(ValueError):
            reextract_on_hash_mismatch(
                registry, 'c0', 's0', 'aaa', 'bbb', 'run-13:legacy'
            )
        self.assertEqual(len(registry), 0)


class AuditReportTest(unittest.TestCase):
    def test_passing_report(self):
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 2),
            make_slices(10, 9),
            cache_stats=stats_with(9, 1),
        )
        self.assertTrue(report.passes())
        self.assertEqual(report.failures(), ())
        self.assertEqual(report.positive_count, 30)
        self.assertEqual(report.control_count, 10)
        self.assertEqual(report.slice_count, 10)

    def test_boundary_values_pass(self):
        report = run_shadow_audit(
            make_positives(30, 21),
            make_controls(10, 3),
            make_slices(10, 8),
            cache_stats=stats_with(9, 1),
        )
        self.assertAlmostEqual(report.ppv, 0.7, places=9)
        self.assertAlmostEqual(report.control_fpr, 0.3, places=9)
        self.assertAlmostEqual(report.consistency, 0.8, places=9)
        self.assertTrue(report.passes())

    def test_low_ppv_fails(self):
        report = run_shadow_audit(
            make_positives(30, 18),
            make_controls(10, 2),
            make_slices(10, 9),
            cache_stats=stats_with(9, 1),
        )
        self.assertIn('ppv', report.failures())
        self.assertFalse(report.passes())

    def test_high_control_fpr_fails(self):
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 4),
            make_slices(10, 9),
            cache_stats=stats_with(9, 1),
        )
        self.assertIn('control_fpr', report.failures())

    def test_low_consistency_fails(self):
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 2),
            make_slices(10, 7),
            cache_stats=stats_with(9, 1),
        )
        self.assertIn('consistency', report.failures())

    def test_low_cache_hit_rate_fails(self):
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 2),
            make_slices(10, 9),
            cache_stats=stats_with(8, 2),
        )
        self.assertIn('cache_hit_rate', report.failures())

    def test_hash_mismatches_are_reported_and_routed(self):
        registry = HashMismatchRegistry()
        reextract_on_hash_mismatch(registry, 'pos-000', 's-000', 'aaa', 'bbb')
        reextract_on_hash_mismatch(registry, 'pos-001', 's-001', 'ccc', 'ddd')
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 2),
            make_slices(10, 9),
            cache_stats=stats_with(9, 1),
            registry=registry,
        )
        self.assertEqual(report.hash_mismatch_count, 2)
        self.assertEqual(report.hash_mismatch_extractors, (RUN9_B1_EXTRACTOR,))
        self.assertTrue(report.hash_mismatch_routing_ok())
        self.assertTrue(report.passes())

    def test_default_registry_is_empty(self):
        report = run_shadow_audit(
            make_positives(30, 24),
            make_controls(10, 2),
            make_slices(10, 9),
            cache_stats=stats_with(9, 1),
        )
        self.assertEqual(report.hash_mismatch_count, 0)
        self.assertTrue(report.hash_mismatch_routing_ok())

    def test_empty_inputs_are_not_a_pass(self):
        report = run_shadow_audit([], [], [], cache_stats=CacheStats())
        self.assertFalse(report.passes())
        self.assertEqual(report.hash_mismatch_count, 0)


class CaseRecordValidationTest(unittest.TestCase):
    def test_valid_cohorts(self):
        self.assertEqual(CaseRecord('c0', True, True, 'control').cohort, 'control')
        self.assertEqual(CaseRecord('c0', True, True, 'positive').cohort, 'positive')

    def test_default_cohort_is_positive(self):
        self.assertEqual(CaseRecord('c0', True, True).cohort, 'positive')

    def test_invalid_cohort_raises(self):
        with self.assertRaises(ValueError):
            CaseRecord('c0', True, True, 'audit')


if __name__ == '__main__':
    unittest.main()
