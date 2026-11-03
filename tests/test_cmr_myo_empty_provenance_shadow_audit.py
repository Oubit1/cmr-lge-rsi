"""Offline unit tests for the cmr_myo_empty provenance shadow audit.

Fully synthetic: no dataset, no patient image and no runtime database is read.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cmr_myo_empty_provenance_shadow_audit import (  # noqa: E402
    ALLOWED_EXTRACTOR,
    CO_OCCURRENCE_ISSUES,
    EMPTY_ISSUE,
    SEGMENTS,
    STRATUM_TOTAL,
    SingleImportExtractorFactory,
    classify_slice,
    cooccurrence_summary,
    dedupe_records,
    dedup_pass_rate,
    empty_slice_distribution,
    run_cache_digest,
    run_shadow_audit,
    same_slice_cooccurrence,
    select_cohort,
    slice_counts_from_records,
    temporal_continuity,
    temporal_continuity_summary,
    verify_cache_digests,
)


def _digester(record):
    payload = [
        str(record.get('run_id')),
        str(record.get('case_id')),
        str(record.get('modality')),
        str(record.get('issue')),
        str(record.get('slice_index')),
    ]
    return 'digest:' + '|'.join(payload)


def _fixture_records():
    """Replay of the verified run-4/5/9 caches for the empty stratum slice.

    30 positive cases: the run-4 cache flags the basal/mid half of the stack
    (slices 0-5) as empty and the run-5 cache flags the mid/apical half
    (slices 6-11), so the dedup union covers the whole stack with no
    empty<->non-empty transition, while still testing duplicate collapse,
    slice-index unioning and run attribution.  10 control cases carry no
    ``empty`` issue at all.
    """
    records = []
    for index in range(30):
        case_id = 'pos_{0:03d}'.format(index)
        for run_id, slices in (
            (4, range(0, 6)),
            (5, range(6, 12)),
        ):
            for slice_index in slices:
                records.append(
                    {
                        'case_id': case_id,
                        'modality': 'LGE',
                        'issue': EMPTY_ISSUE,
                        'slice_index': slice_index,
                        'run_id': run_id,
                        'n_slices': 12,
                    }
                )
        records.append(
            {
                'case_id': case_id,
                'modality': 'LGE',
                'issue': 'small_area',
                'slice_index': index % 6,
                'run_id': 9,
                'n_slices': 12,
            }
        )
    for index in range(10):
        case_id = 'ctl_{0:03d}'.format(index)
        records.append(
            {
                'case_id': case_id,
                'modality': 'LGE',
                'issue': 'centroid_drift',
                'slice_index': index % 12,
                'run_id': 9,
                'n_slices': 12,
            }
        )
    return records


def _verified_digests(records):
    """Run-level digests as stored by the verified cache summary."""
    grouped = {}
    for record in records:
        run_id = record.get('run_id')
        if run_id is not None:
            grouped.setdefault(run_id, []).append(record)
    return {
        run_id: run_cache_digest(run_records, _digester)
        for run_id, run_records in grouped.items()
    }


class DedupTests(unittest.TestCase):

    def test_one_record_per_case_modality(self):
        records = _fixture_records()
        merged = dedupe_records(records)
        keys = [(entry['case_id'], entry['modality']) for entry in merged]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(merged), 40)

    def test_issue_union_keeps_slice_indices(self):
        records = [
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 3,
                'run_id': 4,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 7,
                'run_id': 5,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': 'small_area',
                'slice_index': 7,
                'run_id': 5,
            },
        ]
        merged = dedupe_records(records)
        self.assertEqual(len(merged), 1)
        entry = merged[0]
        self.assertEqual(entry['issues'][EMPTY_ISSUE], [3, 7])
        self.assertEqual(entry['issues']['small_area'], [7])
        self.assertEqual(entry['runs'], [4, 5])
        self.assertEqual(dedup_pass_rate(records, merged), 1.0)

    def test_fixture_union_covers_whole_stack(self):
        merged = dedupe_records(_fixture_records())
        positives = [e for e in merged if e['case_id'].startswith('pos_')]
        self.assertEqual(len(positives), 30)
        for entry in positives:
            self.assertEqual(entry['issues'][EMPTY_ISSUE], list(range(12)))
            self.assertEqual(entry['runs'], [4, 5, 9])

    def test_duplicate_slice_index_is_not_repeated(self):
        records = [
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 3,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 3,
            },
        ]
        entry = dedupe_records(records)[0]
        self.assertEqual(entry['issues'][EMPTY_ISSUE], [3])

    def test_pass_rate_is_one_on_fixture(self):
        records = _fixture_records()
        merged = dedupe_records(records)
        self.assertEqual(dedup_pass_rate(records, merged), 1.0)

    def test_pass_rate_detects_missing_output(self):
        records = _fixture_records()
        merged = dedupe_records(records)[:-1]
        self.assertEqual(dedup_pass_rate(records, merged), 0.0)


class CohortTests(unittest.TestCase):

    def test_cohort_is_30_positive_10_control(self):
        cohort = select_cohort(_fixture_records())
        self.assertEqual(cohort['cohort_size'], 40)
        self.assertEqual(len(cohort['positives']), 30)
        self.assertEqual(len(cohort['controls']), 10)
        self.assertTrue(cohort['positives'][0].startswith('pos_'))
        self.assertTrue(cohort['controls'][0].startswith('ctl_'))

    def test_cohort_rejects_impossible_size(self):
        with self.assertRaises(ValueError):
            select_cohort(_fixture_records(), n_positive=100, n_control=10)

    def test_stratum_total_unchanged(self):
        self.assertEqual(STRATUM_TOTAL, 106)


class HashReuseTests(unittest.TestCase):

    def test_valid_digests_create_no_extractor(self):
        records = _fixture_records()
        factory = SingleImportExtractorFactory()
        report = verify_cache_digests(
            records, _digester, _verified_digests(records), factory
        )
        self.assertEqual(report['mismatch_count'], 0)
        self.assertEqual(report['reuse_rate'], 1.0)
        self.assertTrue(report['reuse_accepted'])
        self.assertEqual(report['new_extractor_count'], 0)
        self.assertEqual(factory.new_extractor_count, 0)
        self.assertEqual(factory.requested_kinds, [])
        self.assertEqual(report['checked_runs'], 3)
        self.assertEqual(report['reused_runs'], 3)
        self.assertEqual(report['run_reuse_rate'], 1.0)
        self.assertEqual(report['record_reuse_rate'], 1.0)

    def test_run_digest_is_order_independent(self):
        records = _fixture_records()
        run_four = [r for r in records if r.get('run_id') == 4]
        self.assertEqual(
            run_cache_digest(run_four, _digester),
            run_cache_digest(list(reversed(run_four)), _digester),
        )

    def test_mismatch_reuses_single_run9_b1_extractor(self):
        records = _fixture_records()
        verified = _verified_digests(records)
        verified[9] = 'digest:stale'
        factory = SingleImportExtractorFactory()
        report = verify_cache_digests(records, _digester, verified, factory)
        self.assertGreater(report['mismatch_count'], 0)
        self.assertEqual(report['mismatch_count'], 1)
        self.assertEqual(report['reuse_rate'], 0.0)
        self.assertFalse(report['reuse_accepted'])
        self.assertEqual(report['reused_runs'], 2)
        self.assertGreater(report['mismatch_record_count'], 0)
        self.assertEqual(report['new_extractor_count'], 1)
        self.assertEqual(factory.requested_kinds, [ALLOWED_EXTRACTOR])

    def test_many_mismatches_still_request_one_extractor(self):
        records = _fixture_records()
        verified = _verified_digests(records)
        for run_id in (4, 5, 9):
            verified[run_id] = 'digest:stale'
        factory = SingleImportExtractorFactory()
        report = verify_cache_digests(records, _digester, verified, factory)
        self.assertEqual(report['mismatch_count'], 3)
        self.assertEqual(report['reuse_rate'], 0.0)
        self.assertEqual(report['reused_runs'], 0)
        self.assertEqual(factory.new_extractor_count, 1)
        self.assertEqual(factory.requested_kinds, [ALLOWED_EXTRACTOR])

    def test_second_toolset_is_refused(self):
        factory = SingleImportExtractorFactory()
        factory.get(ALLOWED_EXTRACTOR)
        with self.assertRaises(PermissionError):
            factory.get('some_other_toolset')
        self.assertEqual(factory.new_extractor_count, 1)


class DistributionTests(unittest.TestCase):

    def test_classify_slice_segments(self):
        self.assertEqual(classify_slice(0, 12), 'basal')
        self.assertEqual(classify_slice(3, 12), 'basal')
        self.assertEqual(classify_slice(4, 12), 'mid')
        self.assertEqual(classify_slice(7, 12), 'mid')
        self.assertEqual(classify_slice(8, 12), 'apical')
        self.assertEqual(classify_slice(11, 12), 'apical')

    def test_classify_slice_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            classify_slice(12, 12)
        with self.assertRaises(ValueError):
            classify_slice(0, 0)

    def test_slice_counts_from_records(self):
        records = [
            {'case_id': 'c1', 'modality': 'LGE', 'n_slices': 10},
            {'case_id': 'c1', 'modality': 'LGE', 'n_slices': 12},
        ]
        counts = slice_counts_from_records(records, 12)
        self.assertEqual(counts[('c1', 'LGE')], 12)

    def test_empty_distribution_covers_three_segments(self):
        records = _fixture_records()
        merged = dedupe_records(records)
        slice_counts = slice_counts_from_records(records, 12)
        distribution = empty_slice_distribution(merged, slice_counts)
        self.assertEqual(set(distribution['counts']), set(SEGMENTS))
        self.assertGreater(distribution['total_empty_slices'], 0)
        self.assertAlmostEqual(sum(distribution['shares'].values()), 1.0)
        self.assertEqual(len(distribution['per_case']), 30)


class ContinuityTests(unittest.TestCase):

    def test_single_contiguous_block_is_continuous(self):
        record = {
            'case_id': 'c1',
            'modality': 'LGE',
            'issues': {EMPTY_ISSUE: [4, 5, 6]},
        }
        item = temporal_continuity(record, 12)
        self.assertEqual(item['transitions'], 2)
        self.assertAlmostEqual(item['continuity'], 1.0 - 2.0 / 11.0)

    def test_scattered_empty_slices_have_many_transitions(self):
        record = {
            'case_id': 'c1',
            'modality': 'LGE',
            'issues': {EMPTY_ISSUE: [0, 2, 4, 6, 8, 10]},
        }
        item = temporal_continuity(record, 12)
        self.assertEqual(item['transitions'], 11)
        self.assertAlmostEqual(item['continuity'], 0.0)

    def test_summary_aggregates_records_with_empty(self):
        records = _fixture_records()
        merged = dedupe_records(records)
        slice_counts = slice_counts_from_records(records, 12)
        summary = temporal_continuity_summary(merged, slice_counts)
        self.assertEqual(summary['records'], 40)
        self.assertEqual(summary['records_with_empty'], 30)
        self.assertEqual(summary['total_transitions'], 0)
        self.assertAlmostEqual(summary['mean_continuity'], 1.0)
        self.assertTrue(summary['is_continuous'])


class CooccurrenceTests(unittest.TestCase):

    def test_cooccurrence_rates_on_shared_slice(self):
        records = [
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 2,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': EMPTY_ISSUE,
                'slice_index': 5,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': 'small_area',
                'slice_index': 5,
            },
            {
                'case_id': 'c1',
                'modality': 'LGE',
                'issue': 'touches_image_border',
                'slice_index': 5,
            },
        ]
        merged = dedupe_records(records)
        stats = same_slice_cooccurrence(merged)
        self.assertAlmostEqual(stats['small_area']['rate'], 0.5)
        self.assertAlmostEqual(stats['touches_image_border']['rate'], 0.5)
        self.assertEqual(stats['small_area']['cooccurring_slices'], 1)
        self.assertEqual(stats['missing_temporal_hole']['rate'], 0.0)
        self.assertEqual(stats['centroid_drift']['rate'], 0.0)

    def test_cooccurrence_summary_has_all_issues(self):
        records = _fixture_records()
        merged = dedupe_records(records)
        summary = cooccurrence_summary(merged)
        self.assertEqual(set(summary), set(CO_OCCURRENCE_ISSUES))


class EndToEndTests(unittest.TestCase):

    def test_audit_returns_all_metrics(self):
        records = _fixture_records()
        report = run_shadow_audit(
            records, _digester, _verified_digests(records), default_n_slices=12
        )
        self.assertEqual(report['stratum'], 'cmr_myo_empty')
        self.assertEqual(report['stratum_total'], STRATUM_TOTAL)
        self.assertEqual(report['dedup']['dedup_pass_rate'], 1.0)
        self.assertEqual(report['hash_reuse']['mismatch_count'], 0)
        self.assertEqual(report['hash_reuse']['new_extractor_count'], 0)
        self.assertEqual(report['hash_reuse']['reuse_rate'], 1.0)
        self.assertEqual(report['hash_reuse']['reused_runs'], 3)
        self.assertEqual(report['cohort']['cohort_size'], 40)
        self.assertEqual(
            set(report['empty_slice_distribution']['counts']), set(SEGMENTS)
        )
        self.assertEqual(
            set(report['cooccurrence']), set(CO_OCCURRENCE_ISSUES)
        )
        self.assertTrue(report['read_only'])
        self.assertEqual(report['extractor']['new_extractor_count'], 0)
        self.assertEqual(report['extractor']['requested_kinds'], [])

    def test_audit_does_not_mutate_inputs(self):
        records = _fixture_records()
        before = json.dumps(records, sort_keys=True)
        run_shadow_audit(
            records, _digester, _verified_digests(records), default_n_slices=12
        )
        self.assertEqual(json.dumps(records, sort_keys=True), before)

    def test_audit_repairs_mismatch_with_one_extractor(self):
        records = _fixture_records()
        verified = _verified_digests(records)
        verified[4] = 'digest:stale'
        report = run_shadow_audit(
            records, _digester, verified, default_n_slices=12
        )
        self.assertGreater(report['hash_reuse']['mismatch_count'], 0)
        self.assertEqual(report['hash_reuse']['reuse_rate'], 0.0)
        self.assertEqual(report['hash_reuse']['new_extractor_count'], 1)
        self.assertEqual(report['extractor']['requested_kinds'], [ALLOWED_EXTRACTOR])
        self.assertEqual(report['dedup']['dedup_pass_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
