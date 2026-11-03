from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from scripts.validate_batch_candidates import (
    candidate_context_reasons,
    identity_matches,
    validate_pair,
)


class CandidateValidationTests(unittest.TestCase):
    def test_manifest_identity_must_still_match_reported_pool_index(self):
        current = {"patient": "P002", "volume": "scan.nii.gz", "frame": 7}
        self.assertTrue(identity_matches(
            {"patient": "P002", "volume": "scan.nii.gz", "frame": 7}, current
        ))
        self.assertFalse(identity_matches(
            {"patient": "P001", "volume": "scan.nii.gz", "frame": 7}, current
        ))
        self.assertFalse(identity_matches(
            {"patient": "P002", "volume": "scan.nii.gz", "frame": 6}, current
        ))

    def test_candidate_context_rejects_stale_identity_source_and_frame(self):
        current = {"patient": "P002", "volume": "scan.nii.gz", "frame": 7}
        reasons, frame = candidate_context_reasons(
            {
                "patient": "P001", "volume": "scan.nii.gz", "frame": 7,
                "source_label": "/dataset/labels/cmr/wrong.nii.gz",
            },
            current,
            Path("/dataset"),
            Path("/dataset/labels/cmr/P002/scan.nii.gz"),
            Path("000123_P002_k5.npz"),
        )
        self.assertEqual(frame, 5)
        self.assertEqual(reasons, [
            "stale_manifest_index", "stale_source_path", "stale_candidate_frame",
        ])

    def test_candidate_context_accepts_current_mapping(self):
        source = Path("/dataset/labels/cmr/P002/scan.nii.gz")
        reasons, frame = candidate_context_reasons(
            {
                "patient": "P002", "volume": "scan.nii.gz", "frame": 7,
                "source_label": str(source),
            },
            {"patient": "P002", "volume": "scan.nii.gz", "frame": 7},
            Path("/dataset"),
            source,
            Path("000123_P002_k6.npz"),
        )
        self.assertEqual((reasons, frame), ([], 6))

    def test_small_detached_component_removal_passes(self):
        original = np.zeros((32, 32), dtype=bool)
        original[8:24, 8:16] = True
        original[8:24, 18:26] = True
        original[2:4, 2:4] = True
        fixed = original.copy()
        fixed[2:4, 2:4] = False
        passed, reasons, _ = validate_pair(original, fixed, ["remove_small_components"])
        self.assertTrue(passed, reasons)

    def test_main_component_removal_fails(self):
        original = np.zeros((32, 32), dtype=bool)
        original[8:24, 8:16] = True
        original[8:24, 18:26] = True
        original[2:4, 2:4] = True
        fixed = np.zeros_like(original)
        fixed[2:4, 2:4] = True
        passed, reasons, _ = validate_pair(original, fixed, ["remove_small_components"])
        self.assertFalse(passed)
        self.assertIn("dominant_component_modified", reasons)

    def test_removing_second_of_two_components_is_rejected(self):
        original = np.zeros((32, 32), dtype=bool)
        original[8:24, 8:16] = True
        original[8:24, 18:22] = True
        fixed = original.copy()
        fixed[8:24, 18:22] = False
        passed, reasons, _ = validate_pair(original, fixed, ["remove_small_components"])
        self.assertFalse(passed)
        self.assertIn("two_component_anatomy_is_legal", reasons)

    def test_internal_hole_restore_passes(self):
        original = np.zeros((32, 32), dtype=bool)
        yy, xx = np.ogrid[:32, :32]
        original[(xx - 16) ** 2 + (yy - 16) ** 2 <= 10 ** 2] = True
        fixed = original.copy()
        fixed[(xx - 16) ** 2 + (yy - 16) ** 2 <= 5 ** 2] = False
        self.assertTrue(np.array_equal(ndi.binary_fill_holes(original),
                                       ndi.binary_fill_holes(fixed)))
        passed, reasons, _ = validate_pair(original, fixed, ["restore_temporal_hole"])
        self.assertTrue(passed, reasons)


if __name__ == "__main__":
    unittest.main()
