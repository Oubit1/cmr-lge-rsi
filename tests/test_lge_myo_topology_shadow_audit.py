import json
import os
import tempfile
import unittest

from experiments.lge_myo_topology_shadow_audit import (
    FALLBACK_RUN9_B1,
    audit_topology_shadow,
    load_cache_summaries,
)


def _slice(index, multi=False, high=False):
    return {
        'slice_index': index,
        'mi_multiple_components': multi,
        'mi_fraction_high': high,
    }


def _row(case_id, stratum, run_id=9, cache_hit=True, expected_hash='h', observed_hash='h', slices=None):
    return {
        'case_id': case_id,
        'modality': 'LGE',
        'stratum': stratum,
        'run_id': run_id,
        'cache_hit': cache_hit,
        'expected_hash': expected_hash,
        'observed_hash': observed_hash,
        'slices': slices if slices is not None else [],
    }


class LgeMyoTopologyShadowAuditTest(unittest.TestCase):
    def test_cooccurrence_and_case_level_support_difference(self):
        rows = [
            _row('P1', 'positive', slices=[_slice(0, True, False), _slice(1, True, True)]),
            _row('P2', 'positive', slices=[_slice(0, False, False)]),
            _row('C1', 'control', slices=[_slice(0, True, True), _slice(1, False, False)]),
            _row('C2', 'control', slices=[_slice(0, False, False)]),
        ]
        report = audit_topology_shadow(rows, positive_n=2, control_n=2)
        self.assertAlmostEqual(report['cooccurrence']['positive_mi_multiple_components_rate'], 2 / 3)
        self.assertAlmostEqual(report['cooccurrence']['control_mi_multiple_components_rate'], 1 / 3)
        self.assertAlmostEqual(report['cooccurrence']['difference'], 1 / 3)
        self.assertAlmostEqual(report['case_level_mi_fraction_high_support']['positive_support_rate'], 0.5)
        self.assertAlmostEqual(report['case_level_mi_fraction_high_support']['control_support_rate'], 0.5)
        self.assertAlmostEqual(report['case_level_mi_fraction_high_support']['difference'], 0.0)

    def test_cache_reuse_and_hash_mismatch_fallback_reporting(self):
        rows = []
        for i in range(9):
            rows.append(
                _row(
                    f'P{i}',
                    'positive',
                    run_id=9,
                    cache_hit=True,
                    expected_hash='abc',
                    observed_hash='abc',
                    slices=[_slice(0)],
                )
            )
        rows.append(
            _row(
                'P9',
                'positive',
                run_id=9,
                cache_hit=True,
                expected_hash='abc',
                observed_hash='def',
                slices=[_slice(0)],
            )
        )
        rows.append(
            _row(
                'C0',
                'control',
                run_id=9,
                cache_hit=False,
                expected_hash='abc',
                observed_hash='abc',
                slices=[_slice(0)],
            )
        )
        rows.append(
            _row(
                'C1',
                'control',
                run_id=7,
                cache_hit=True,
                expected_hash='abc',
                observed_hash='abc',
                slices=[_slice(0)],
            )
        )
        report = audit_topology_shadow(rows)
        self.assertAlmostEqual(report['cache_reuse']['reuse_rate'], 9 / 12)
        self.assertEqual(report['cache_reuse']['hash_mismatch_fallbacks'], 1)
        self.assertEqual(report['cache_reuse']['fallback_reasons']['hash_mismatch'], 1)
        self.assertEqual(report['cache_reuse']['fallback_reasons']['cache_miss'], 1)
        self.assertEqual(report['cache_reuse']['fallback_reasons']['run_not_in_allowed_cache_set'], 1)
        self.assertEqual(report['cache_reuse']['fallback_events'][0]['fallback_extractor'], FALLBACK_RUN9_B1)

    def test_duplicate_rows_and_slice_index_addressability(self):
        rows = [
            _row('P1', 'positive', slices=[_slice(0)]),
            _row('P1', 'positive', slices=[_slice(1)]),
            _row(
                'P2',
                'positive',
                slices=[
                    {
                        'slice_index': 'bad',
                        'mi_multiple_components': False,
                        'mi_fraction_high': False,
                    }
                ],
            ),
        ]
        report = audit_topology_shadow(rows, positive_n=2, control_n=0)
        self.assertEqual(report['counts']['duplicate_rows'], 1)
        self.assertEqual(report['counts']['unique_case_modality_rows'], 2)
        self.assertEqual(report['addressability']['slice_index_unaddressable'], 1)
        self.assertFalse(report['addressability']['all_slices_addressable'])
        self.assertFalse(report['uniqueness_ok'])

    def test_load_cache_summaries_jsonl_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cache.jsonl')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(_row('P1', 'positive', slices=[_slice(0)])) + '\n')
                handle.write(json.dumps(_row('C1', 'control', slices=[_slice(0)])) + '\n')
            loaded = load_cache_summaries(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]['case_id'], 'P1')


if __name__ == '__main__':
    unittest.main()
