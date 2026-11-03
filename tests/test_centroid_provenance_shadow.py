import unittest
import builtins
from experiments.centroid_provenance_shadow import (
    validate_digests, sample_cases, apply_noop_arm, apply_largest_component_retention_arm,
    apply_temporal_continuity_arm, compute_persistence_rate, compute_centroid_displacement_delta,
    attribute_outliers, deduplicate_evidence, run_shadow_analysis, REQUIRED_CONFIG_HASH
)

class TestCentroidProvenanceShadow(unittest.TestCase):
    def setUp(self):
        self.valid_digests = {
            'case_001': {
                'modality': 'CMR',
                'slices': [
                    {'id': 1, 'review_centroid_outlier': True, 'x': 1.0, 'y': 2.0, 'z': 3.0, 'attribution_category': 'small-area'},
                    {'id': 2, 'review_centroid_outlier': False, 'x': 1.1, 'y': 2.1, 'z': 3.1}
                ]
            },
            'case_002': {
                'modality': 'LGE',
                'slices': [
                    {'id': 1, 'review_centroid_outlier': True, 'x': 4.0, 'y': 5.0, 'z': 6.0, 'attribution_category': 'border-touching'},
                    {'id': 2, 'review_centroid_outlier': True, 'x': 4.1, 'y': 5.1, 'z': 6.1, 'attribution_category': 'temporal-drift'}
                ]
            }
        }
        self.invalid_digests = {"bad": "data"}

    def test_validate_digests_valid(self):
        self.assertTrue(validate_digests(self.valid_digests))

    def test_validate_digests_invalid(self):
        self.assertFalse(validate_digests(self.invalid_digests))

    def test_sample_cases(self):
        cases = sample_cases(self.valid_digests)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]['stratum'], 'CMR')
        self.assertEqual(cases[1]['stratum'], 'LGE')
        self.assertEqual(len(cases[0]['outlier_slices']), 1)
        self.assertEqual(len(cases[1]['outlier_slices']), 2)

    def test_apply_noop_arm(self):
        cases = sample_cases(self.valid_digests)
        noop = apply_noop_arm(cases)
        self.assertEqual(noop, cases)

    def test_apply_largest_component_retention_arm(self):
        cases = sample_cases(self.valid_digests)
        ret = apply_largest_component_retention_arm(cases)
        for c in ret:
            for s in c['outlier_slices']:
                self.assertIn('retained', s)

    def test_apply_temporal_continuity_arm(self):
        cases = sample_cases(self.valid_digests)
        cont = apply_temporal_continuity_arm(cases)
        for c in cont:
            for s in c['outlier_slices']:
                self.assertIn('continuous', s)

    def test_compute_persistence_rate(self):
        before = [True, False, True]
        after = [True, True, True]
        p = compute_persistence_rate(before, after)
        self.assertAlmostEqual(p, 2/3)

    def test_compute_centroid_displacement_delta(self):
        cb = [(0,0,0), (1,1,1)]
        ca = [(1,1,1), (2,2,2)]
        d = compute_centroid_displacement_delta(cb, ca)
        self.assertAlmostEqual(d, 1.732, places=2)

    def test_attribute_outliers(self):
        flags = [
            {'attribution_category': 'small-area'},
            {'attribution_category': 'border-touching'},
            {'attribution_category': 'small-area'}
        ]
        attr = attribute_outliers(flags)
        self.assertAlmostEqual(attr['small-area'], 2/3)
        self.assertAlmostEqual(attr['border-touching'], 1/3)
        self.assertAlmostEqual(attr['unexplained-residual'], 0.0)

    def test_deduplicate_evidence(self):
        payload = [
            {'case_id': 'c1', 'modality': 'CMR', 'slice_idx': 0},
            {'case_id': 'c1', 'modality': 'CMR', 'slice_idx': 1},
            {'case_id': 'c2', 'modality': 'LGE', 'slice_idx': 0}
        ]
        deduped = deduplicate_evidence(payload, max_records=2)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]['case_id'], 'c1')
        self.assertEqual(deduped[1]['case_id'], 'c2')

    def test_run_shadow_analysis(self):
        results = run_shadow_analysis(self.valid_digests, REQUIRED_CONFIG_HASH)
        self.assertIn('CMR', results)
        self.assertIn('LGE', results)
        self.assertIn('persistence_rates', results['CMR'])
        self.assertIn('attribution_fractions', results['CMR'])
        self.assertLessEqual(results['CMR']['evidence_dedup_count'], 20)

    def test_config_hash_mismatch_stops_execution(self):
        with self.assertRaises(ValueError) as context:
            run_shadow_analysis(self.valid_digests, "invalid_hash")
        self.assertIn("mismatch", str(context.exception).lower())

    def test_read_only_enforcement(self):
        original_open = builtins.open
        open_called = False
        def mock_open(*args, **kwargs):
            nonlocal open_called
            open_called = True
            raise IOError("Production write attempt detected!")
        try:
            builtins.open = mock_open
            run_shadow_analysis(self.valid_digests, REQUIRED_CONFIG_HASH)
            self.assertFalse(open_called, "Module attempted to write to production storage.")
        finally:
            builtins.open = original_open

if __name__ == '__main__':
    unittest.main()
