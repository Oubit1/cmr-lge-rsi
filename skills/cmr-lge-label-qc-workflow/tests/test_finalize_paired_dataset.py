from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/finalize_paired_dataset.py"
SPEC = importlib.util.spec_from_file_location("finalize_paired_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FinalizePairedDatasetTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> dict[str, Path]:
        roots = {
            "cmr_image": root / "images/cmr",
            "lge_image": root / "images/lge",
            "cmr_myo": root / "labels/cmr_checked/cmr_Myo_mask",
            "lge_myo": root / "labels/lge_checked/lge_Myo_labels",
            "lge_mi": root / "labels/lge_checked/lge_MI_labels",
            "lge_myo_original": root / "labels/lge_original/lge_Myo_labels",
            "lge_mi_original": root / "labels/lge_original/lge_MI_labels",
        }
        for key in ("P/1.nii.gz", "P/2.nii.gz"):
            for name, directory in roots.items():
                path = directory / key
                path.parent.mkdir(parents=True, exist_ok=True)
                shape = (8, 8, 2) if name in {"cmr_image", "cmr_myo"} else (8, 8)
                nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4)), str(path))
        issue = root / "labels/check/cmr_boundary_issue_slices/issue_manifest.csv"
        issue.parent.mkdir(parents=True)
        with issue.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient", "volume", "issues"])
            writer.writeheader()
            writer.writerow({"patient": "P", "volume": "1.nii.gz", "issues": "boundary"})
        return roots

    def test_checked_is_primary_and_original_is_tracked_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            layout, selected = MODULE.choose_layout(root, "checked")
            self.assertEqual(selected, "checked")
            self.assertIn("labels/lge_checked", layout["lge_myo"].as_posix())
            self.assertIn("lge_myo_aux_original", layout)

    def test_complete_bundle_quarantine_preserves_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            layout, _ = MODULE.choose_layout(root, "checked")
            complete, _, before = MODULE.pairing_audit(layout, check_shapes=True)
            self.assertEqual(before["complete_pairs"], 2)
            reasons = {"P/1.nii.gz": {"boundary_issue_outside_trusted_sax"}}
            destination = root / "quarantine/unusable_sax/run"
            destination.mkdir(parents=True)
            rows = MODULE.quarantine_bundles(root, layout, complete, reasons, destination)
            self.assertEqual(rows[0]["bundle_files"], 7)
            _, _, after = MODULE.pairing_audit(layout, check_shapes=True)
            self.assertTrue(after["exact_key_sets"])
            self.assertEqual(after["complete_pairs"], 1)
            for directory in layout.values():
                self.assertFalse((directory / "P/1.nii.gz").exists())
                self.assertTrue((directory / "P/2.nii.gz").exists())


if __name__ == "__main__":
    unittest.main()
