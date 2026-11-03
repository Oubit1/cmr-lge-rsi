"""Inference planning never needs to activate an untested candidate."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from app import segmentation_registry as registry
from scripts.segment_from_registry import predict_command


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(self.writable)
        self.vendor = self.root / 'vendor'
        self.folder = self.vendor / 'automated_annotation_of_lge'
        (self.folder / 'lge_seg').mkdir(parents=True)
        (self.folder / 'main.py').write_text('# synthetic entry')
        architecture = self.folder / 'lge_seg/model.py'
        architecture.write_text('# synthetic architecture')
        self.registry = self.root / 'registry'
        self.inputs = self.root / 'images'
        self.inputs.mkdir()
        self.files = {'architecture': architecture}
        for name in ('weights', 'config', 'dataset_manifest'):
            self.files[name] = self.root / name
            self.files[name].write_text('{}' if name == 'config' else 'synthetic ' + name)
        self.parent = registry.register_baseline(self.registry, target='lge_myo', owner='custodian', **self.files)
        sources = {p.relative_to(self.folder).as_posix(): self.sha(p) for p in self.folder.rglob('*.py')}
        receipt = {'status': 'SUCCEEDED', 'training_source': 'training', 'owner_role': 'lge_myo_specialist',
                   'attempt_id': 'synthetic', 'metrics': {},
                   'artifact_hashes': {k: self.sha(p) for k, p in self.files.items()},
                   'vendor_sources_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True,
                       separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()}
        self.candidate = registry.register_candidate(self.registry, target='lge_myo', owner='owner',
                            parent_id=self.parent['id'], training_record=receipt, **self.files)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def writable(self):
        for folder, _, files in os.walk(self.root):
            for name in files:
                (Path(folder) / name).chmod(0o600)

    def plan(self, **kwargs):
        return predict_command(self.registry, self.vendor, kwargs.pop('target', 'lge_myo'), self.inputs,
                               kwargs.pop('output', self.root / 'predictions'),
                               model_id=self.candidate['id'], **kwargs)

    def test_inactive_candidate_is_evaluated_without_switching_baseline(self):
        registry.select_baseline(self.registry, self.parent['id'])
        command, receipt = self.plan()
        self.assertEqual(receipt['model_id'], self.candidate['id'])
        self.assertFalse(receipt['production_labels_written'])
        self.assertEqual(registry.resolve_active(self.registry, 'lge_myo')['candidate_id'], self.parent['id'])
        self.assertIn('--config', command)
        self.assertLess(command.index('--config'), command.index('predict'))
        self.assertFalse((self.root / 'predictions').exists())

    def test_changed_non_model_vendor_file_is_rejected(self):
        (self.folder / 'main.py').write_text('# changed inference preprocessing')
        with self.assertRaisesRegex(ValueError, 'vendor_sources_changed'):
            self.plan()

    def test_wrong_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'model_target_mismatch'):
            self.plan(target='cmr_myo')

    def test_output_cannot_overlap_inputs(self):
        for path in (self.inputs, self.inputs / 'nested', self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'new_separate_output'):
                self.plan(output=path)

    def test_candidate_artifact_corruption_is_detected(self):
        model = registry.resolve_candidate(self.registry, self.candidate['id'])
        path = Path(model['artifacts']['weights']['path'])
        path.chmod(0o600)
        path.write_text('changed weights')
        with self.assertRaisesRegex(registry.RegistryError, 'artifact hash/size mismatch'):
            self.plan()


if __name__ == '__main__':
    unittest.main()
