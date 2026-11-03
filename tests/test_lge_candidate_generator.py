"""Tests for scripts/lge_candidate_generator.py – offline, no external data."""

from __future__ import annotations

import os
import tempfile
import unittest

import nibabel as nib
import numpy as np

from scripts.lge_candidate_generator import (
    _connected_component_count,
    _largest_component,
    _structure_for_ndim,
    generate_candidates,
)


class TestStructureForNdim(unittest.TestCase):
    def test_2d_returns_3x3(self):
        s = _structure_for_ndim(2)
        self.assertEqual(s.shape, (3, 3))
        self.assertTrue(s.dtype == bool)

    def test_3d_returns_3x3x3(self):
        s = _structure_for_ndim(3)
        self.assertEqual(s.shape, (3, 3, 3))
        self.assertTrue(s.dtype == bool)


class TestConnectedComponentCount(unittest.TestCase):
    def test_2d_single_component(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[2:5, 2:5] = 1
        self.assertEqual(_connected_component_count(mask > 0), 1)

    def test_2d_two_components(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[1:3, 1:3] = 1
        mask[6:8, 6:8] = 1
        self.assertEqual(_connected_component_count(mask > 0), 2)

    def test_3d_singleton_single_component(self):
        mask = np.zeros((1, 10, 10), dtype=np.int16)
        mask[0, 2:5, 2:5] = 1
        self.assertEqual(_connected_component_count(mask > 0), 1)

    def test_3d_singleton_two_components(self):
        mask = np.zeros((1, 10, 10), dtype=np.int16)
        mask[0, 1:3, 1:3] = 1
        mask[0, 6:8, 6:8] = 1
        self.assertEqual(_connected_component_count(mask > 0), 2)

    def test_3d_multislice_two_components(self):
        mask = np.zeros((3, 10, 10), dtype=np.int16)
        mask[0, 1:3, 1:3] = 1
        mask[2, 6:8, 6:8] = 1
        self.assertEqual(_connected_component_count(mask > 0), 2)


class TestLargestComponent(unittest.TestCase):
    def test_2d_keeps_largest(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[1:4, 1:4] = 1  # 3x3 = 9 pixels
        mask[6:8, 6:8] = 1  # 2x2 = 4 pixels
        result = _largest_component(mask > 0)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(result[1:4, 1:4], expected))
        self.assertFalse(np.any(result[6:8, 6:8]))

    def test_3d_singleton_keeps_largest(self):
        mask = np.zeros((1, 10, 10), dtype=np.int16)
        mask[0, 1:4, 1:4] = 1  # 9 pixels
        mask[0, 6:8, 6:8] = 1  # 4 pixels
        result = _largest_component(mask > 0)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(result[0, 1:4, 1:4], expected))
        self.assertFalse(np.any(result[0, 6:8, 6:8]))

    def test_3d_multislice_keeps_largest(self):
        mask = np.zeros((3, 10, 10), dtype=np.int16)
        mask[0, 1:4, 1:4] = 1  # 9 pixels
        mask[2, 6:8, 6:8] = 1  # 4 pixels
        result = _largest_component(mask > 0)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(result[0, 1:4, 1:4], expected))
        self.assertFalse(np.any(result[2, 6:8, 6:8]))


class TestGenerateCandidates(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_nifti(self, name, data):
        path = os.path.join(self.tmpdir, name)
        img = nib.Nifti1Image(data.astype(np.int16), np.eye(4))
        nib.save(img, path)
        return path

    def test_missing_paths_needs_review(self):
        data = {"records": [{"case_id": "c1"}]}
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "needs_review")
        self.assertEqual(results[0]["reason"], "missing_snapshot_paths")

    def test_file_not_found_needs_review(self):
        data = {
            "records": [
                {
                    "case_id": "c1",
                    "snapshot_image_path": "/nonexistent/img.nii.gz",
                    "snapshot_mask_path": "/nonexistent/mask.nii.gz",
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "needs_review")
        self.assertIn("snapshot_file_not_found", results[0]["reason"])

    def test_single_component_accept_existing(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[2:5, 2:5] = 1
        image = np.zeros((10, 10), dtype=np.int16)
        img_path = self._write_nifti("img.nii.gz", image)
        msk_path = self._write_nifti("msk.nii.gz", mask)
        data = {
            "records": [
                {
                    "case_id": "c1",
                    "snapshot_image_path": img_path,
                    "snapshot_mask_path": msk_path,
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "accept_existing")
        self.assertIsNone(results[0]["candidate_filename"])

    def test_multiple_components_repair(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        mask[1:4, 1:4] = 1
        mask[6:8, 6:8] = 1
        image = np.zeros((10, 10), dtype=np.int16)
        img_path = self._write_nifti("img.nii.gz", image)
        msk_path = self._write_nifti("msk.nii.gz", mask)
        data = {
            "records": [
                {
                    "case_id": "c1",
                    "snapshot_image_path": img_path,
                    "snapshot_mask_path": msk_path,
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "repair")
        self.assertIsNotNone(results[0]["candidate_filename"])
        cand_path = os.path.join(self.tmpdir, results[0]["candidate_filename"])
        self.assertTrue(os.path.isfile(cand_path))
        cand = np.asarray(nib.load(cand_path).dataobj)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(cand[1:4, 1:4], expected))
        self.assertFalse(np.any(cand[6:8, 6:8]))

    def test_empty_mask_needs_review(self):
        mask = np.zeros((10, 10), dtype=np.int16)
        image = np.zeros((10, 10), dtype=np.int16)
        img_path = self._write_nifti("img.nii.gz", image)
        msk_path = self._write_nifti("msk.nii.gz", mask)
        data = {
            "records": [
                {
                    "case_id": "c1",
                    "snapshot_image_path": img_path,
                    "snapshot_mask_path": msk_path,
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "needs_review")
        self.assertEqual(results[0]["reason"], "empty_mask")

    def test_3d_singleton_repair(self):
        mask = np.zeros((1, 10, 10), dtype=np.int16)
        mask[0, 1:4, 1:4] = 1
        mask[0, 6:8, 6:8] = 1
        image = np.zeros((1, 10, 10), dtype=np.int16)
        img_path = self._write_nifti("img3d.nii.gz", image)
        msk_path = self._write_nifti("msk3d.nii.gz", mask)
        data = {
            "records": [
                {
                    "case_id": "c3d",
                    "snapshot_image_path": img_path,
                    "snapshot_mask_path": msk_path,
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "repair")
        cand_path = os.path.join(self.tmpdir, results[0]["candidate_filename"])
        self.assertTrue(os.path.isfile(cand_path))
        cand = np.asarray(nib.load(cand_path).dataobj)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(cand[0, 1:4, 1:4], expected))
        self.assertFalse(np.any(cand[0, 6:8, 6:8]))

    def test_3d_multislice_repair(self):
        mask = np.zeros((3, 10, 10), dtype=np.int16)
        mask[0, 1:4, 1:4] = 1
        mask[2, 6:8, 6:8] = 1
        image = np.zeros((3, 10, 10), dtype=np.int16)
        img_path = self._write_nifti("img3dms.nii.gz", image)
        msk_path = self._write_nifti("msk3dms.nii.gz", mask)
        data = {
            "records": [
                {
                    "case_id": "c3dms",
                    "snapshot_image_path": img_path,
                    "snapshot_mask_path": msk_path,
                }
            ]
        }
        results = generate_candidates(data, self.tmpdir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["decision"], "repair")
        cand_path = os.path.join(self.tmpdir, results[0]["candidate_filename"])
        self.assertTrue(os.path.isfile(cand_path))
        cand = np.asarray(nib.load(cand_path).dataobj)
        expected = np.ones((3, 3), dtype=np.int16)
        self.assertTrue(np.array_equal(cand[0, 1:4, 1:4], expected))
        self.assertFalse(np.any(cand[2, 6:8, 6:8]))


if __name__ == "__main__":
    unittest.main()
