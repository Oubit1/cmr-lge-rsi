from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "scripts/promote_checked.py"
SPEC = importlib.util.spec_from_file_location("promote_checked_test", MODULE)
promote = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(promote)


class PromotionTests(unittest.TestCase):
    def test_all_passed_frames_are_applied_with_one_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            check = dataset / "labels/check"
            patient, volume = "CASE", "1.nii.gz"
            source = check / "待处理/cmr" / patient / "1.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"png")
            volume_path = dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
            volume_path.parent.mkdir(parents=True)
            original = np.zeros((8, 8, 2), dtype=np.uint8)
            original[2:5, 2:5, :] = 1
            nib.save(nib.Nifti1Image(original, np.eye(4)), str(volume_path))

            triage = root / "runtime/boundary_triage/slice_triage.csv"
            triage.parent.mkdir(parents=True)
            triage.write_text("relative_key,bucket\nCASE/1.nii.gz,SALVAGE\n")
            manifest = check / "manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nCASE,1.nii.gz\nCASE,1.nii.gz\n")
            candidates = root / "runtime/batch_candidates/cmr_boundary_manifest"
            candidates.mkdir(parents=True)
            fixed0 = original[..., 0].copy()
            fixed0[2, 2] = 0
            fixed1 = original[..., 1].copy()
            fixed1[4, 4] = 0
            np.savez_compressed(candidates / "000000_CASE_k0.npz",
                                original=original[..., 0], fixed=fixed0)
            np.savez_compressed(candidates / "000001_CASE_k1.npz",
                                original=original[..., 1], fixed=fixed1)
            gate = {"records": [
                {"pool": "cmr_boundary_manifest", "index": i, "passed": True,
                 "operation": ["remove_small_components"]} for i in range(2)
            ]}
            (candidates.parent / "candidate_validation.json").write_text(json.dumps(gate))

            previous = promote.ROOT, promote.DATASET, promote.CHECK, promote.LOG, promote.BACKUP_ROOT
            try:
                promote.ROOT = root
                promote.DATASET = dataset
                promote.CHECK = check
                promote.LOG = root / "runtime/promotion/promotion_log.jsonl"
                promote.LOG.parent.mkdir(parents=True)
                promote.BACKUP_ROOT = dataset / "label_iteration/promotion_backups"
                promote.run(SimpleNamespace(execute=True))
                actual = np.asanyarray(nib.load(str(volume_path)).dataobj)
                np.testing.assert_array_equal(actual[..., 0] > 0, fixed0 > 0)
                np.testing.assert_array_equal(actual[..., 1] > 0, fixed1 > 0)
                self.assertFalse(source.exists())
                self.assertTrue((check / "checked/cmr/CASE/1.png").exists())
                log = json.loads(promote.LOG.read_text().splitlines()[0])
                self.assertEqual(log["frames"], [0, 1])
                self.assertEqual(log["changed_pixels"], 2)
                self.assertTrue(Path(log["backup"]).exists())
            finally:
                promote.ROOT, promote.DATASET, promote.CHECK, promote.LOG, promote.BACKUP_ROOT = previous


if __name__ == "__main__":
    unittest.main()
