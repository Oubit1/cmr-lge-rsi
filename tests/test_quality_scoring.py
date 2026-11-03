from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from app.quality_scoring import (
    PairSpec,
    add_interslice_context,
    build_profile,
    extract_pair,
    load_specs,
    score_record,
)


class QualityScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        yy, xx = np.ogrid[:32, :32]
        radius = np.sqrt((xx - 16) ** 2 + (yy - 16) ** 2)
        self.ring = (radius <= 10) & (radius >= 6)
        image = np.zeros((32, 32, 3), dtype=np.float32)
        image[self.ring, :] = 80
        image[radius < 6, :] = 30
        label = np.repeat(self.ring[..., None], 3, axis=-1).astype(np.uint8)
        self.image = self.root / "image.nii.gz"
        self.label = self.root / "label.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(self.image))
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(self.label))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_extracts_clean_temporal_cmr_pair(self) -> None:
        record = extract_pair(PairSpec(
            dataset="test", tier="gold", target="cmr_myo", image=self.image,
            label=self.label, myo=None, relative_id="case/slice.nii.gz",
            case_id="test:case", slice_position=1, reviewed=False,
        ))
        self.assertTrue(record["valid"])
        self.assertEqual(record["frames"], 3)
        self.assertAlmostEqual(record["temporal_dice"], 1.0)
        self.assertEqual(record["components"], 1.0)

    def test_sparse_phase_protocol_does_not_treat_unlabelled_frames_as_failures(self) -> None:
        label = np.zeros((32, 32, 10), dtype=np.uint8)
        label[..., 0] = self.ring
        sparse_path = self.root / "sparse.nii.gz"
        image_path = self.root / "sparse_image.nii.gz"
        image = np.repeat(np.asarray(nib.load(str(self.image)).dataobj)[..., :1], 10, axis=-1)
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(sparse_path))
        record = extract_pair(PairSpec(
            dataset="test", tier="gold", target="cmr_myo", image=image_path,
            label=sparse_path, myo=None, relative_id="sparse.nii.gz",
            case_id="test:sparse", slice_position=1, reviewed=False,
        ))
        self.assertTrue(record["sparse_phase"])
        self.assertEqual(record["annotated_frames"], 1)
        profile = build_profile([record])
        scored = score_record(record, profile)
        self.assertNotIn("mostly_empty_label", scored["flags"])
        self.assertGreaterEqual(scored["score"], 90)

    def test_mi_outside_myo_is_penalized_and_flagged(self) -> None:
        clean = {
            "valid": True, "dataset": "test", "tier": "gold", "target": "lge_mi",
            "binary_fraction": 1.0, "finite_fraction": 1.0, "affine_match": 1.0,
            "dynamic_range": 100.0, "largest_component_fraction": 1.0,
            "extra_after_two_fraction": 0.0, "tiny_component_fraction": 0.0,
            "holes": 0.0, "components": 1.0, "boundary_gradient_ratio": 2.0,
            "area": 30.0, "mi_containment": 1.0, "mi_signal_z": 3.0,
            "interslice_score": 100.0, "border_frame_fraction": 0.0,
            "empty_frame_fraction": 0.0, "reviewed": False,
        }
        profile = build_profile([clean])
        good = score_record(clean, profile)
        bad = score_record({**clean, "mi_containment": .5}, profile)
        self.assertGreater(good["score"], bad["score"])
        self.assertIn("mi_outside_myo", bad["flags"])

    def test_sandwich_gap_is_low_continuity(self) -> None:
        records = [
            {"valid": True, "target": "lge_mi", "case_id": "x", "slice_position": 1, "area": 20},
            {"valid": True, "target": "lge_mi", "case_id": "x", "slice_position": 2, "area": 0},
            {"valid": True, "target": "lge_mi", "case_id": "x", "slice_position": 3, "area": 18},
        ]
        add_interslice_context(records)
        self.assertTrue(records[1]["sandwich_gap"])
        self.assertLessEqual(records[1]["interslice_score"], 20)
        self.assertTrue(records[0]["boundary_slice"])

    def test_registered_paths_are_rejected(self) -> None:
        config = self.root / "config.yaml"
        config.write_text(
            "datasets:\n  - name: bad\n    targets:\n      cmr_myo:\n"
            f"        image_dir: {self.root}/images\n"
            f"        label_dir: {self.root}/labels_warpped\n",
            encoding="utf-8",
        )
        (self.root / "images").mkdir()
        (self.root / "labels_warpped").mkdir()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            load_specs(config)


if __name__ == "__main__":
    unittest.main()
