import unittest

from experiments.cmr_myo_small_area_import_shadow_audit import (
    B1_EXTRACTOR_NAME,
    SliceRecord,
    build_import_shadow_audit,
    cache_hash_and_b1_summary,
    dedup_gate_summary,
    flagged_slice_indices_addressable,
    positive_slice_support_summary,
    records_from_cache_summaries,
    worst_slice_per_case,
)


def rec(case_id, modality, slice_index, label='cmr_myo_small_area', flags=None, cached_hash=None, live_hash=None, source_run='run-9', extractor=None):
    return SliceRecord(
        case_id=case_id,
        modality=modality,
        slice_index=slice_index,
        label=label,
        flags=flags or {},
        cached_hash=cached_hash,
        live_hash=live_hash,
        source_run=source_run,
        extractor=extractor,
    )


class PositiveSupportSummaryTest(unittest.TestCase):
    def test_no_same_slice_support_ratio_and_cooccurrence(self):
        records = [
            rec('c1', 'LGE', 1, flags={}),
            rec('c1', 'LGE', 2, flags={'excess_components': True}),
            rec('c2', 'LGE', 3, flags={'myo_empty': True, 'centroid_drift': True}),
            rec('c3', 'LGE', 4, label='control', flags={'excess_components': True}),
        ]
        summary = positive_slice_support_summary(records)
        self.assertEqual(summary['positive_slice_count'], 3)
        self.assertEqual(summary['unsupported_positive_slice_count'], 1)
        self.assertAlmostEqual(summary['unsupported_positive_slice_ratio'], 1 / 3)
        self.assertEqual(summary['same_slice_cooccurrence_counts']['excess_components'], 1)
        self.assertEqual(summary['same_slice_cooccurrence_counts']['myo_empty'], 1)
        self.assertEqual(summary['same_slice_cooccurrence_counts']['centroid_drift'], 1)
        self.assertEqual(summary['same_slice_cooccurrence_counts']['touches_image_border'], 0)


class WorstSliceTest(unittest.TestCase):
    def test_worst_slice_and_addressability(self):
        records = [
            rec('c1', 'LGE', 1, flags={'excess_components': True}),
            rec('c1', 'LGE', 2, flags={'myo_empty': True}),
            rec('c1', 'LGE', 3, flags={'excess_components': True, 'touches_image_border': True, 'missing_temporal_hole': True}),
            rec('c2', 'LGE', 5, flags={'centroid_drift': True, 'layer_area_jump': True}),
            rec('c2', 'LGE', 6, label='control', flags={}),
        ]
        worst = worst_slice_per_case(records)
        self.assertEqual(worst['c1']['worst_slice_index'], 2)
        self.assertEqual(worst['c1']['worst_slice_severity'], 'critical')
        self.assertEqual(worst['c2']['worst_slice_index'], 5)
        self.assertEqual(worst['c2']['worst_slice_severity'], 'medium')
        addressability = flagged_slice_indices_addressable(records)
        self.assertTrue(addressability['addressable'])
        self.assertEqual(addressability['failures'], {})


class CacheB1Test(unittest.TestCase):
    def test_hash_mismatch_counts_and_b1_only(self):
        records = [
            rec('c1', 'LGE', 1, cached_hash='a', live_hash='a', source_run='run-4'),
            rec('c1', 'LGE', 2, cached_hash='a', live_hash='b', source_run='run-9', extractor=B1_EXTRACTOR_NAME),
            rec('c2', 'LGE', 3, cached_hash='x', live_hash='y', source_run='run-13', extractor='other'),
            rec('c2', 'LGE', 4, source_run='run-99'),
        ]
        summary = cache_hash_and_b1_summary(records)
        self.assertEqual(summary['cache_hash_mismatch_count'], 2)
        self.assertEqual(summary['b1_extractor_call_count'], 1)
        self.assertEqual(summary['non_b1_on_mismatch_count'], 1)
        self.assertEqual(summary['disallowed_cache_run_count'], 1)
        self.assertEqual(summary['b1_extractor_run'], 'run-9')


class DedupGateTest(unittest.TestCase):
    def test_exactly_one_record_per_case_modality(self):
        records = [
            rec('c1', 'LGE', 1),
            rec('c1', 'LGE', 2),
            rec('c2', 'LGE', 1),
            rec('c2', 'LGE', 2),
        ]
        summary = dedup_gate_summary(records)
        self.assertEqual(summary['group_count'], 2)
        self.assertEqual(summary['passed_group_count'], 0)
        self.assertEqual(summary['dedup_gate_pass_rate'], 0.0)
        self.assertFalse(summary['exactly_one_per_case_modality'])

    def test_dedup_pass_rate_one(self):
        records = [
            rec('c1', 'LGE', 1),
            rec('c2', 'LGE', 1),
            rec('c3', 'LGE', 1),
        ]
        summary = dedup_gate_summary(records)
        self.assertEqual(summary['dedup_gate_pass_rate'], 1.0)
        self.assertTrue(summary['exactly_one_per_case_modality'])


class CacheSummaryImportTest(unittest.TestCase):
    def test_only_allowed_runs_imported(self):
        summaries = [
            {'case_id': 'c1', 'modality': 'LGE', 'slice_index': 1, 'source_run': 'run-4', 'flags': {'excess_components': True}},
            {'case_id': 'c2', 'modality': 'LGE', 'slice_index': 2, 'source_run': 'run-99', 'flags': {}},
        ]
        records = records_from_cache_summaries(summaries)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].case_id, 'c1')


class BuildAuditTest(unittest.TestCase):
    def test_build_audit_shape(self):
        records = [rec('c1', 'LGE', 1, flags={'myo_empty': True})]
        audit = build_import_shadow_audit(records)
        self.assertEqual(audit['target_label'], 'cmr_myo_small_area')
        self.assertIn('positive_slice_support', audit)
        self.assertIn('worst_slice_per_case', audit)
        self.assertIn('cache_hash_and_b1', audit)
        self.assertIn('dedup_gate', audit)


if __name__ == '__main__':
    unittest.main()
