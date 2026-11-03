import importlib.util
import pathlib
import sys
import unittest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / 'experiments' / 'cmr_myo_empty_readonly_shadow_audit.py'
_SPEC = importlib.util.spec_from_file_location('cmr_myo_empty_readonly_shadow_audit', _MODULE_PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)


class ReadOnlyShadowAuditTests(unittest.TestCase):
    def _case(self, case_id, label, slice_index, temporal_index, payload=b'payload'):
        return _AUDIT.CaseRecord(
            case_id=case_id,
            label=label,
            slice_index=slice_index,
            temporal_index=temporal_index,
            payload=payload,
        )

    def test_cache_hit_reuses_record_without_extractor(self):
        case = self._case('p-hit', _AUDIT.POSITIVE_LABEL, 3, 7)
        record = _AUDIT.CacheRecord(
            run_id=4,
            case_id='p-hit',
            slice_index=3,
            temporal_index=7,
            payload_hash=_AUDIT.payload_hash(case.payload),
            classification='pass',
            features={'temporal_hole': 1},
        )

        def extractor(_case):
            raise AssertionError('extractor must not be called on a valid cache hit')

        finding = _AUDIT.audit_case(case, {case.key: record}, extractor)
        self.assertEqual(finding.route, 'cache_hit')
        self.assertTrue(finding.cache_hit)
        self.assertFalse(finding.hash_mismatch)
        self.assertEqual(finding.classification, 'pass')

    def test_hash_mismatch_uses_registered_run9_b1_extractor_and_registers_event(self):
        case = self._case('p-mismatch', _AUDIT.POSITIVE_LABEL, 5, 2, b'payload-1')
        record = _AUDIT.CacheRecord(
            run_id=13,
            case_id='p-mismatch',
            slice_index=5,
            temporal_index=2,
            payload_hash='not-the-real-hash',
            classification='pass',
            features={},
        )
        calls = []

        def extractor(_case):
            calls.append(_case.case_id)
            return {'classification': 'failure', 'features': {'small_area': True, 'border': 1}}

        finding = _AUDIT.audit_case(case, {case.key: record}, extractor)
        self.assertEqual(finding.route, 'cache_hash_mismatch_run9_b1')
        self.assertTrue(finding.hash_mismatch)
        self.assertFalse(finding.cache_hit)
        self.assertEqual(calls, ['p-mismatch'])
        self.assertEqual(finding.events[0]['registered_extractor_run'], 9)
        self.assertEqual(finding.events[0]['registered_extractor_variant'], 'B1')
        self.assertEqual(finding.events[0]['cache_run_id'], 13)
        self.assertTrue(finding.events[0]['read_only'])

    def test_cache_miss_registers_and_uses_read_only_fallback(self):
        case = self._case('p-miss', _AUDIT.POSITIVE_LABEL, 1, 1)

        def extractor(_case):
            return {'classification': 'pass', 'features': {}}

        finding = _AUDIT.audit_case(case, {}, extractor)
        self.assertEqual(finding.route, 'cache_miss_run9_b1')
        self.assertFalse(finding.hash_mismatch)
        self.assertEqual(finding.events[0]['event'], 'cmr_myo_empty_shadow_audit.cache_miss')

    def test_summary_reports_completion_cache_and_rate_intervals(self):
        cases = [
            self._case('p1', _AUDIT.POSITIVE_LABEL, 1, 0),
            self._case('p2', _AUDIT.POSITIVE_LABEL, 1, 1),
            self._case('p3', _AUDIT.POSITIVE_LABEL, 2, 2),
            self._case('p4', _AUDIT.POSITIVE_LABEL, 2, 3),
            self._case('c1', _AUDIT.CONTROL_LABEL, 1, 0),
            self._case('c2', _AUDIT.CONTROL_LABEL, 2, 1),
        ]

        def extractor(case):
            if case.case_id == 'p1':
                return {'classification': 'failure', 'features': {'small_area': True, 'border': 1}}
            return {'classification': 'pass', 'features': {}}

        summary = _AUDIT.audit_cases(cases, {}, extractor)
        self.assertEqual(summary.completion_rate, 1.0)
        self.assertEqual(summary.cache_hits, 0)
        self.assertEqual(summary.cache_misses, 6)
        self.assertEqual(summary.hash_mismatches, 0)
        self.assertEqual(summary.cache_miss_rate, 1.0)
        self.assertEqual(summary.hash_mismatch_rate, 0.0)
        self.assertEqual(summary.positive_true_failure_rate, 0.25)
        lower, upper = summary.positive_true_failure_interval
        self.assertLess(lower, 0.25)
        self.assertGreater(upper, 0.25)
        self.assertEqual(summary.control_false_positive_rate, 0.0)
        self.assertEqual(summary.control_false_positive_interval[0], 0.0)
        self.assertGreater(summary.control_false_positive_interval[1], 0.0)

    def test_cooccurrence_rates_are_reported(self):
        cases = [
            self._case('p1', _AUDIT.POSITIVE_LABEL, 1, 0),
            self._case('p2', _AUDIT.POSITIVE_LABEL, 1, 1),
        ]

        def extractor(case):
            if case.case_id == 'p1':
                return {
                    'classification': 'failure',
                    'features': {
                        'small_area': True,
                        'excess_components': 2,
                        'border': True,
                        'temporal_hole': 0,
                        'centroid_drift': 1,
                    },
                }
            return {'classification': 'pass', 'features': {}}

        summary = _AUDIT.audit_cases(cases, {}, extractor)
        self.assertEqual(summary.cooccurrence_rates['small_area']['positive_rate'], 0.5)
        self.assertEqual(summary.cooccurrence_rates['excess_components']['positive_rate'], 0.5)
        self.assertEqual(summary.cooccurrence_rates['border']['positive_rate'], 0.5)
        self.assertEqual(summary.cooccurrence_rates['temporal_hole']['positive_rate'], 0.0)
        self.assertEqual(summary.cooccurrence_rates['centroid_drift']['positive_rate'], 0.5)
        self.assertEqual(summary.cooccurrence_rates['small_area']['control_rate'], 0.0)

    def test_stratification_by_slice_and_temporal_neighbor(self):
        cases = [
            self._case('a', _AUDIT.POSITIVE_LABEL, 2, 0),
            self._case('b', _AUDIT.POSITIVE_LABEL, 2, 1),
            self._case('c', _AUDIT.CONTROL_LABEL, 3, 1),
        ]
        strata = _AUDIT.stratify_cases(cases)
        self.assertEqual(strata['by_slice']['slice_2'], ['a', 'b'])
        self.assertEqual(strata['by_slice']['slice_3'], ['c'])
        self.assertEqual(strata['by_temporal_neighbor']['temporal_neighbor_0_1'], ['a'])
        self.assertEqual(strata['by_temporal_neighbor']['temporal_neighbor_1_2'], ['b', 'c'])

    def test_completion_acceptance_can_be_enforced(self):
        cases = [self._case('p1', _AUDIT.POSITIVE_LABEL, 1, 0)]
        summary = _AUDIT.audit_cases(cases, {}, extractor=None, enforce_completion=False)
        self.assertEqual(summary.completion_rate, 0.0)
        with self.assertRaises(ValueError):
            _AUDIT.audit_cases(cases, {}, extractor=None, enforce_completion=True)
        with self.assertRaises(ValueError):
            _AUDIT.check_completion_acceptance(summary)


if __name__ == '__main__':
    unittest.main()
