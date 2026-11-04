"""Offline interface tests using synthetic, in-memory NIfTI snapshots.

Fixtures test behavior, not clinical accuracy or controller fitness.
Run: python -B -m unittest discover -s tests -p test_lge_candidate_generator_acceptance.py
"""
import copy
import io
import json
import os
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from scripts import lge_candidate_generator as generator


class CandidateAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.images = {}
        self.saved = []
        self.affine = np.diag([1.5, 1.5, 4.0, 1.0])
        self.output = os.path.abspath('tests/candidate_outputs')
        self.profile = {'targets': {'lge_myo': {
            'n': 4859, 'metrics': {'holes': {'q01': 1.0, 'q99': 1.0}}}}}
        patches = [
            patch.object(generator.nib, 'load', side_effect=self.images.__getitem__),
            patch.object(generator.nib, 'save', side_effect=lambda img, path: self.saved.append((img, path))),
            patch.object(generator.os.path, 'isfile', side_effect=lambda p: p in self.images),
            patch.object(generator.os, 'makedirs'),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def ring(self):
        mask = np.zeros((20, 20), dtype=np.int16)
        mask[4:16, 4:16] = 1
        mask[7:13, 7:13] = 0
        return mask

    def record(self, mask, name='synthetic', aliases=False, stage='lge_myo'):
        arrays = {'image': np.ones(mask.shape, dtype=np.int16), 'label': mask,
                  'mi' if stage == 'lge_myo' else 'myo': np.zeros_like(mask)}
        paths = {}
        for role, array in arrays.items():
            path = os.path.abspath(f'tests/synthetic_snapshots/{name}_{role}.nii.gz')
            self.images[path] = nib.Nifti1Image(array.copy(), self.affine)
            paths[role] = path
        rec = {'case_id': name, 'issue_codes': [], 'measurements': {}}
        if aliases:
            for role, path in paths.items():
                alias = 'mask' if role == 'label' else role
                rec[f'snapshot_{alias}_path'] = path
        else:
            rec['paths'] = paths
        return rec

    def generate(self, records, stage='lge_myo', calibrated=True):
        before = {p: np.asarray(img.dataobj).copy() for p, img in self.images.items()}
        data = {'workflow_stage': stage, 'records': records}
        if calibrated:
            data['calibration_profile'] = self.profile
        result = generator.generate_candidates(data, self.output)
        for path, array in before.items():
            np.testing.assert_array_equal(array, np.asarray(self.images[path].dataobj))
        self.assertEqual(len(result), len(records))
        return result

    def test_legacy_single_component_acceptance_without_ring_prior(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[2:8, 2:8] = 1
        rec = self.record(mask, aliases=True)
        self.assertEqual(self.generate([rec], calibrated=False)[0]['decision'], 'accept_existing')
        self.profile['targets']['lge_myo']['metrics']['holes']['q01'] = 0
        self.assertEqual(self.generate([rec])[0]['decision'], 'accept_existing')
        self.assertEqual(self.saved, [])

    def test_clean_ring_is_accepted_for_supported_geometries_and_aliases(self):
        base = self.ring()
        masks = [base, base[:, :, None], base[None, :, :], np.stack([base, base], axis=2)]
        for index, mask in enumerate(masks):
            with self.subTest(shape=mask.shape):
                rec = self.record(mask, f'clean_{index}', aliases=bool(index % 2))
                result = self.generate([rec])[0]
                self.assertEqual(result['decision'], 'accept_existing')
                self.assertIsNone(result['candidate_filename'])
        self.assertEqual(self.saved, [])

    def test_calibrated_filled_cavity_and_open_ring_abstain_without_repair(self):
        filled = self.ring()
        filled[7:13, 7:13] = 1
        opened = self.ring()
        opened[4:8, 9:11] = 0
        for index, mask in enumerate([filled, opened]):
            result = self.generate([self.record(mask, f'defect_{index}')])[0]
            self.assertEqual(result['decision'], 'needs_review')
            self.assertIn('cavity_review:holes=0', result['reason'])
        self.assertEqual(self.saved, [])

    def test_additional_hole_and_multislice_defect_abstain(self):
        extra = self.ring()
        extra[5, 5] = 0
        stack = np.stack([self.ring(), np.ones((20, 20), dtype=np.int16)], axis=2)
        for index, mask in enumerate([extra, stack]):
            result = self.generate([self.record(mask, f'topology_{index}')])[0]
            self.assertEqual(result['decision'], 'needs_review')

    def test_parent_largest_component_repair_is_preserved_for_both_stages(self):
        for stage in ('lge_myo', 'lge_mi'):
            for depth in (None, 1, 2):
                mask = self.ring()
                mask[0, 0] = 1
                if depth is not None:
                    mask = np.repeat(mask[:, :, None], depth, axis=2)
                rec = self.record(mask, f'{stage}_{depth}', stage=stage)
                rec['measurements']['myo_components'] = 2
                result = self.generate([rec], stage)[0]
                self.assertEqual(result['decision'], 'repair')
                self.assertIn('measured_myo_components=2', result['reason'])
                image, path = self.saved[-1]
                expected = mask.copy()
                expected[0, 0] = 0
                np.testing.assert_array_equal(np.asarray(image.dataobj), expected)
                np.testing.assert_array_equal(image.affine, self.affine)
                self.assertEqual(image.shape, mask.shape)
                self.assertEqual(set(np.unique(image.dataobj)), {0, 1})
                self.assertFalse(os.path.isabs(result['candidate_filename']))
                self.assertEqual(os.path.dirname(path), self.output)

    def test_mi_single_component_behavior_is_preserved(self):
        rec = self.record(np.ones((8, 8), dtype=np.int16), stage='lge_mi')
        self.assertEqual(self.generate([rec], 'lge_mi')[0]['decision'], 'accept_existing')

    def test_unsupported_inputs_do_not_abort_batch(self):
        bad = self.record(np.ones((3, 3, 2, 2), dtype=np.int16), 'four_dimensional')
        mismatch = self.record(self.ring(), 'mismatch')
        pair_path = mismatch['paths']['mi']
        self.images[pair_path] = nib.Nifti1Image(np.zeros((2, 2), dtype=np.int16), self.affine)
        clean = self.record(self.ring(), 'valid')
        result = self.generate([{}, bad, mismatch, None, clean])
        self.assertEqual([r['decision'] for r in result], ['needs_review'] * 4 + ['accept_existing'])
        self.assertEqual(generator.generate_candidates(None, self.output), [])
        self.assertEqual(self.saved, [])

    def test_candidate_cannot_overwrite_any_snapshot(self):
        mask = self.ring()
        mask[0, 0] = 1
        rec = self.record(mask)
        collision = os.path.join(self.output, 'candidate_000000_repaired.nii.gz')
        self.images[collision] = self.images[rec['paths']['label']]
        rec['paths']['label'] = collision
        result = self.generate([rec])[0]
        self.assertEqual(result['decision'], 'needs_review')
        self.assertIn('snapshot_boundary', result['reason'])
        self.assertEqual(self.saved, [])

    def test_cli_retains_json_list_contract(self):
        rec = self.record(self.ring())
        payload = {'workflow_stage': 'lge_myo', 'records': [rec],
                   'calibration_profile': self.profile, 'output_dir': self.output}
        stdout = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(payload))), patch('sys.stdout', stdout):
            generator.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result[0]['case_id'], rec['case_id'])
        self.assertEqual(result[0]['decision'], 'accept_existing')

    def test_measurement_report_is_pure_and_metrics_remain_unavailable(self):
        data = {'workflow_stage': 'lge_myo', 'frozen_split': {'status': 'UNAVAILABLE'},
                'records': [{'case_id': 'synthetic', 'measurements': {'myo_components': 2},
                             'score': 0.8, 'flags': ['myo_topology'], 'subscores': {'area': 0.7}}]}
        before = copy.deepcopy(data)
        with patch.object(generator, 'generate_candidates') as generate:
            report = generator.run_evaluation(data)
            generate.assert_not_called()
        self.assertEqual(data, before)
        self.assertEqual(report['frozen_split']['status'], 'UNAVAILABLE')
        self.assertEqual(report['holdout_excluded'], 0)
        self.assertIsNone(report['metrics']['defect_false_acceptance'])
        for key in ('score', 'flags', 'subscores', 'measurements'):
            self.assertEqual(report['records'][0][key], data['records'][0][key])
        report['records'][0]['flags'].append('report_only')
        self.assertEqual(data, before)
        self.assertEqual(self.saved, [])


if __name__ == '__main__':
    unittest.main()
