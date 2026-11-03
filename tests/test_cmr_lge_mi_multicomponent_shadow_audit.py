'''Offline unit tests for the bounded LGE mi_multiple_components shadow audit.

No dataset, network, cache file or runtime database is touched: every case is
synthetic and built in memory.
'''

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cmr_lge_mi_multicomponent_shadow_audit import (  # noqa: E402
    B1_RUN,
    CACHE_RUNS,
    FRAGMENTED_LABEL,
    FRAGMENTED_SHARE_TARGET,
    TRUE_MULTIFOCAL_LABEL,
    build_cache_index,
    classify_multi_component,
    dedup_gate,
    lookup_cached,
    run_shadow_audit,
    select_cohort,
)


def _entry(case_id, modality, run, summary, sha='sha-match'):
    return {
        'case_id': case_id,
        'modality': modality,
        'run': run,
        'sha256': sha,
        'summary': summary,
    }


def _summary(areas, worst, topology='compact', mi_high=False):
    return {
        'areas_mm2': list(areas),
        'worst_slice_components': worst,
        'myo_topology': topology,
        'mi_fraction_high': mi_high,
    }


def _case(case_id, modality, cohort, sha='sha-match'):
    payload = {'case_id': case_id, 'modality': modality, 'cohort': cohort}
    if sha is not None:
        payload['expected_sha256'] = sha
    return payload


def _boom(*args, **kwargs):
    raise AssertionError('run-9 B1 extractor must not be called here')


class ConstantsTest(unittest.TestCase):

    def test_reuse_runs_and_b1_contract(self):
        self.assertEqual(
            list(CACHE_RUNS),
            ['run-4', 'run-5', 'run-9', 'run-13', 'run-14',
             'run-15', 'run-16', 'run-17', 'run-18'],
        )
        self.assertEqual(B1_RUN, 'run-9')
        self.assertEqual(FRAGMENTED_SHARE_TARGET, 0.60)


class ClassifyTest(unittest.TestCase):

    def test_strata(self):
        self.assertEqual(classify_multi_component([]), 'no_component')
        self.assertEqual(classify_multi_component([12.0]), 'single_lesion')
        self.assertEqual(classify_multi_component([30.0, 25.0]), TRUE_MULTIFOCAL_LABEL)
        # small satellites collapse into a single true lesion -> pseudo
        self.assertEqual(classify_multi_component([30.0, 2.0, 3.0]), FRAGMENTED_LABEL)
        # no component reaches the area floor
        self.assertEqual(classify_multi_component([2.0, 1.0]), FRAGMENTED_LABEL)
        # many similarly sized small pieces, none individually dominant
        self.assertEqual(classify_multi_component([9.0] * 10), FRAGMENTED_LABEL)

    def test_thresholds_are_parameters(self):
        self.assertEqual(
            classify_multi_component([30.0, 2.0, 3.0], min_area_mm2=1.0),
            TRUE_MULTIFOCAL_LABEL,
        )


class DedupTest(unittest.TestCase):

    def test_duplicate_case_modality_is_blocked(self):
        records = [
            {'case_id': 'c1', 'modality': 'LGE'},
            {'case_id': 'c1', 'modality': 'LGE'},
            {'case_id': 'c1', 'modality': 'T2'},
            {'case_id': 'c2', 'modality': 'LGE'},
        ]
        accepted, blocked = dedup_gate(records)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]['reason'], 'duplicate_case_modality')
        self.assertEqual(
            (blocked[0]['case_id'], blocked[0]['modality']),
            ('c1', 'LGE'),
        )


class CacheIndexTest(unittest.TestCase):

    def test_lookup_prefers_declared_run_order(self):
        entries = [
            _entry('c1', 'LGE', 'run-18', {'areas_mm2': [1.0]}),
            _entry('c1', 'LGE', 'run-4', {'areas_mm2': [2.0]}),
        ]
        index = build_cache_index(entries)
        hit = lookup_cached(index, 'c1', 'LGE')
        self.assertEqual(hit['run'], 'run-4')
        self.assertIsNone(lookup_cached(index, 'c1', 'T2'))

    def test_unknown_run_is_still_usable(self):
        index = build_cache_index([_entry('c2', 'LGE', 'run-99', {'areas_mm2': [3.0]})])
        self.assertEqual(lookup_cached(index, 'c2', 'LGE')['run'], 'run-99')


class AuditTest(unittest.TestCase):

    def _matched_cohort(self):
        cases, entries = [], []
        for i in range(50):
            case_id = 'p%02d' % i
            cases.append(_case(case_id, 'LGE', 'positive'))
            entries.append(_entry(
                case_id, 'LGE', 'run-4', _summary([20.0, 12.0], 2, mi_high=True)
            ))
        for i in range(10):
            case_id = 'c%02d' % i
            cases.append(_case(case_id, 'LGE', 'control'))
            entries.append(_entry(
                case_id, 'LGE', 'run-13', _summary([15.0], 1, topology=None)
            ))
        return cases, entries

    def test_cache_hits_do_not_call_b1(self):
        cases, entries = self._matched_cohort()
        report = run_shadow_audit(cases, entries, b1_extractor=_boom)
        self.assertTrue(report['read_only'])
        self.assertEqual(report['cache']['hits'], 60)
        self.assertEqual(report['cache']['misses'], 0)
        self.assertEqual(report['cache']['hit_rate'], 1.0)
        self.assertEqual(report['cache']['b1_triggers'], 0)
        self.assertEqual(report['b1_fallbacks'], [])
        self.assertEqual(
            report['strata']['positive']['classification_counts'],
            {TRUE_MULTIFOCAL_LABEL: 50},
        )
        self.assertFalse(report['acceptance']['fragmented_share_met'])
        self.assertEqual(report['components']['positive_median_components'], 2)
        self.assertEqual(
            report['components']['positive_worst_slice_distribution'],
            {'2': 50},
        )
        self.assertEqual(report['cooccurrence']['myo_topology_rate'], 1.0)
        self.assertEqual(report['cooccurrence']['mi_fraction_high_rate'], 1.0)

    def test_hash_mismatch_uses_only_b1_and_is_reported(self):
        case = _case('p1', 'LGE', 'positive', sha='sha-new')
        entry = _entry('p1', 'LGE', 'run-4', _summary([20.0, 18.0], 2), sha='sha-old')
        calls = []

        def extractor(case_id, modality):
            calls.append((case_id, modality))
            return _summary([20.0, 18.0], 2)

        report = run_shadow_audit([case], [entry], b1_extractor=extractor)
        self.assertEqual(calls, [('p1', 'LGE')])
        self.assertEqual(report['cache']['hits'], 0)
        self.assertEqual(report['cache']['hash_mismatches'], 1)
        self.assertEqual(report['cache']['b1_triggers'], 1)
        self.assertEqual(report['b1_fallbacks'][0]['reason'], 'hash_mismatch')
        self.assertEqual(report['b1_fallbacks'][0]['run'], B1_RUN)
        self.assertEqual(report['b1_fallbacks'][0]['extractor'], 'B1/' + B1_RUN)
        self.assertTrue(report['b1_fallbacks'][0]['resolved'])
        self.assertEqual(report['cases'][0]['source'], 'b1_hash_mismatch')

    def test_missing_recorded_digest_is_treated_as_mismatch(self):
        case = _case('p2', 'LGE', 'positive', sha='sha-expected')
        entry = {
            'case_id': 'p2',
            'modality': 'LGE',
            'run': 'run-4',
            'summary': _summary([9.0, 8.0], 2),
        }
        report = run_shadow_audit(
            [case], [entry], b1_extractor=lambda c, m: _summary([9.0, 8.0], 2)
        )
        self.assertEqual(report['cache']['hash_mismatches'], 1)
        self.assertEqual(report['cache']['b1_triggers'], 1)
        self.assertEqual(report['cases'][0]['classification'], TRUE_MULTIFOCAL_LABEL)

    def test_unverified_hit_is_used_when_caller_supplies_no_digest(self):
        case = {'case_id': 'p3', 'modality': 'LGE', 'cohort': 'positive'}
        entry = _entry('p3', 'LGE', 'run-5', _summary([30.0, 26.0], 2))
        report = run_shadow_audit([case], [entry], b1_extractor=_boom)
        self.assertEqual(report['cases'][0]['source'], 'cache_unverified')
        self.assertEqual(report['cache']['hits'], 1)
        self.assertEqual(report['cache']['unverified_hits'], 1)
        self.assertEqual(report['cache']['b1_triggers'], 0)

    def test_cache_miss_without_extractor_is_unresolved(self):
        case = _case('p9', 'LGE', 'positive', sha=None)
        report = run_shadow_audit([case], [])
        self.assertEqual(report['cache']['b1_triggers'], 0)
        self.assertEqual(report['cache']['unresolved'], 1)
        self.assertEqual(report['cases'][0]['source'], 'unresolved_cache_miss')
        self.assertIsNone(report['cases'][0]['classification'])
        self.assertEqual(report['strata']['positive']['classified'], 0)
        self.assertIsNone(report['strata']['positive']['fragmented_share'])
        self.assertFalse(report['strata']['positive']['meets_target'])

    def test_dedup_gate_blocks_duplicate_records_in_audit(self):
        cases = [
            _case('p1', 'LGE', 'positive'),
            _case('p1', 'LGE', 'positive'),
        ]
        entries = [_entry('p1', 'LGE', 'run-4', _summary([20.0, 18.0], 2))]
        report = run_shadow_audit(cases, entries, b1_extractor=_boom)
        self.assertEqual(report['counts']['dedup_blocked'], 1)
        self.assertEqual(report['cache']['dedup_blocked_duplicate_case_modality'], 1)
        self.assertEqual(report['counts']['accepted'], 1)
        self.assertEqual(len(report['cases']), 1)

    def test_cohort_bounds_are_enforced(self):
        positives = [_case('p%03d' % i, 'LGE', 'positive', sha=None) for i in range(51)]
        with self.assertRaises(ValueError):
            run_shadow_audit(positives, [])
        report = run_shadow_audit(positives, [], strict_bounds=False)
        self.assertEqual(report['counts']['accepted'], 51)
        self.assertFalse(report['cohort_bounds']['strict'])

    def test_select_cohort_is_deterministic_and_bounded(self):
        cases = [_case('p%02d' % i, 'LGE', 'positive', sha=None) for i in (9, 1, 5, 3, 7, 11)]
        cases += [_case('c%d' % i, 'LGE', 'control', sha=None) for i in range(4)]
        selected = select_cohort(cases, positives=3, controls=2)
        self.assertEqual(
            [case['case_id'] for case in selected],
            ['p01', 'p03', 'p05', 'c0', 'c1'],
        )

    def test_stratified_share_components_and_cooccurrence(self):
        positives = [
            ('p1', _summary([40.0, 2.0, 1.0], 3, topology='compact')),
            ('p2', _summary([9.0] * 10, 10, topology='compact')),
            ('p3', _summary([2.0, 1.0], 2, topology=None)),
            ('p4', _summary([30.0, 25.0], 2, topology='compact', mi_high=True)),
            ('p5', _summary([18.0, 14.0, 3.0], 3, topology=None, mi_high=True)),
        ]
        cases, entries = [], []
        for case_id, summary in positives:
            cases.append(_case(case_id, 'LGE', 'positive'))
            entries.append(_entry(case_id, 'LGE', 'run-4', summary))
        for case_id in ('c1', 'c2'):
            cases.append(_case(case_id, 'LGE', 'control'))
            entries.append(_entry(
                case_id, 'LGE', 'run-9', _summary([15.0], 1, topology=None)
            ))
        report = run_shadow_audit(cases, entries)
        stratum = report['strata']['positive']
        self.assertEqual(stratum['fragmented_pseudo'], 3)
        self.assertEqual(stratum['classified'], 5)
        self.assertEqual(stratum['fragmented_share'], 0.6)
        self.assertTrue(stratum['meets_target'])
        self.assertTrue(report['acceptance']['fragmented_share_met'])
        self.assertEqual(report['components']['positive_median_components'], 3)
        self.assertEqual(
            report['components']['positive_worst_slice_distribution'],
            {'2': 2, '3': 2, '10': 1},
        )
        self.assertEqual(report['cooccurrence']['myo_topology_rate'], 0.6)
        self.assertEqual(report['cooccurrence']['mi_fraction_high_rate'], 0.4)
        self.assertEqual(report['cooccurrence']['fragmented_myo_topology_rate'], 0.6667)
        self.assertEqual(report['cooccurrence']['fragmented_mi_fraction_high_rate'], 0.0)
        self.assertEqual(
            report['strata']['control']['classification_counts'],
            {'single_lesion': 2},
        )

    def test_audit_is_read_only_and_json_serialisable(self):
        cases = [_case('p1', 'LGE', 'positive')]
        entries = [_entry('p1', 'LGE', 'run-4', _summary([30.0, 26.0], 2))]
        cases_before = copy.deepcopy(cases)
        entries_before = copy.deepcopy(entries)
        report = run_shadow_audit(cases, entries)
        self.assertEqual(cases, cases_before)
        self.assertEqual(entries, entries_before)
        self.assertTrue(report['read_only'])
        self.assertEqual(json.loads(json.dumps(report))['counts']['accepted'], 1)
        self.assertEqual(report['b1_run'], B1_RUN)
        self.assertEqual(
            report['runs_reused'],
            list(CACHE_RUNS),
        )


if __name__ == '__main__':
    unittest.main()
