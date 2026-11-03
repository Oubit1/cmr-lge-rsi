import unittest
import numpy as np
from experiments.lge_mi_fraction_high_cooccurrence_stratum import compute_stratum_metrics, apply_shadow_arms

class TestLgeMiFractionHighCooccurrenceStratum(unittest.TestCase):
    def setUp(self):
        self.mock_lge = np.zeros((10, 10), dtype=np.uint8)
        self.mock_lge[3:7, 3:7] = 1
        self.mock_cmr = np.zeros((10, 10), dtype=np.uint8)
        self.mock_cmr[2:8, 2:8] = 1
        self.sample_data = [
            {
                'id': 'case_1',
                'lge_mask': self.mock_lge.tolist(),
                'cmr_mask': self.mock_cmr.tolist(),
                'has_myocardial_topology': True,
                'is_control': False,
                'mi_fraction_high': True
            },
            {
                'id': 'case_2',
                'lge_mask': self.mock_lge.tolist(),
                'cmr_mask': self.mock_cmr.tolist(),
                'has_myocardial_topology': False,
                'is_control': False,
                'mi_fraction_high': True
            },
            {
                'id': 'ctrl_1',
                'lge_mask': np.zeros((10, 10), dtype=np.uint8).tolist(),
                'cmr_mask': self.mock_cmr.tolist(),
                'has_myocardial_topology': False,
                'is_control': True,
                'mi_fraction_high': False
            }
        ]

    def test_stratum_metrics_structure(self):
        metrics = compute_stratum_metrics(self.sample_data)
        self.assertIn('total_cases', metrics)
        self.assertEqual(metrics['total_cases'], 3)
        self.assertIn('stratified_counts', metrics)
        self.assertEqual(metrics['stratified_counts']['with_topo'], 1)
        self.assertEqual(metrics['stratified_counts']['without_topo'], 1)
        self.assertEqual(metrics['stratified_counts']['controls'], 1)
        self.assertIn('review_queue', metrics)
        self.assertIsInstance(metrics['review_queue'], list)

    def test_shadow_arms_apply(self):
        lge = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.uint8)
        cmr = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
        res = apply_shadow_arms(lge, cmr)
        self.assertIn('noop', res)
        self.assertIn('close_merge', res)
        self.assertIn('ring_break', res)
        self.assertTrue(np.array_equal(res['noop'], lge))

    def test_zero_production_writes(self):
        import inspect
        sig = inspect.signature(compute_stratum_metrics)
        params = list(sig.parameters.keys())
        self.assertNotIn('write_handle', params)
        self.assertNotIn('db_conn', params)

    def test_control_case_handling(self):
        ctrl_data = [{'id': 'c1', 'lge_mask': [[0]*10 for _ in range(10)], 'cmr_mask': [[1]*10 for _ in range(10)], 'has_myocardial_topology': False, 'is_control': True, 'mi_fraction_high': False}]
        m = compute_stratum_metrics(ctrl_data)
        self.assertEqual(m['stratified_counts']['controls'], 1)
        self.assertEqual(m['flag_rates']['original'], 0.0)

if __name__ == '__main__':
    unittest.main()
