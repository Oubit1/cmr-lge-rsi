from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app.quality_scoring import PairSpec, build_profile, extract_pair


def load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


holdout = load("freeze_check_holdout")
spatial = load("shadow_spatial_boundary")
watch = load("check_feedback_loop")
pending = load("evaluate_pending_check")


class CheckIterationTests(unittest.TestCase):
    def test_candidate_quality_uses_existing_profile_without_writing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path, label_path = root / "image.nii.gz", root / "label.nii.gz"
            yy, xx = np.ogrid[:32, :32]
            ring = (np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2) <= 10) & (
                np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2) >= 6)
            image = np.repeat((ring * 80).astype(np.float32)[..., None], 3, axis=-1)
            label = np.repeat(ring.astype(np.uint8)[..., None], 3, axis=-1)
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
            nib.save(nib.Nifti1Image(label, np.eye(4)), str(label_path))
            original_bytes = label_path.read_bytes()
            spec = PairSpec(dataset="test", tier="gold", target="cmr_myo",
                            image=image_path, label=label_path, myo=None,
                            relative_id="test", case_id="test", slice_position=0,
                            reviewed=True)
            profile = build_profile([extract_pair(spec)])
            candidate = label.copy()
            candidate[0, 0, 0] = 1
            result = pending.compare_quality(image_path, label_path, candidate, root, profile)
            self.assertEqual(result["quality_status"], "SCORED")
            self.assertIn("quality_delta", result)
            self.assertEqual(label_path.read_bytes(), original_bytes)
            self.assertFalse(list(root.glob("tmp*")))

    def test_pending_evaluation_reports_quality_without_promoting_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, candidate_root = root / "dataset", root / "candidate"
            candidate_root.mkdir()
            yy, xx = np.ogrid[:32, :32]
            radius = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
            ring = (radius <= 10) & (radius >= 6)
            image = np.repeat((ring * 80).astype(np.float32)[..., None], 3, axis=-1)
            label = np.repeat(ring.astype(np.uint8)[..., None], 3, axis=-1)
            image_path = dataset / "images/cmr/CASE/1.nii.gz"
            label_path = dataset / "labels/cmr_checked/cmr_Myo_mask/CASE/1.nii.gz"
            for path, array in ((image_path, image), (label_path, label)):
                path.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
            original_bytes = label_path.read_bytes()
            png = dataset / "labels/check/待筛/cmr/CASE/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"png")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nCASE,1.nii.gz\n")
            triage = root / "triage.csv"
            triage.write_text("relative_key,ref_area\nCASE/1.nii.gz,100\n")
            candidate_path = candidate_root / "000000_CASE_k0.npz"
            fixed = ring.copy()
            fixed[0, 0] = True
            np.savez_compressed(candidate_path, original=ring.astype(np.uint8),
                                fixed=fixed.astype(np.uint8))
            (candidate_root / "candidate_validation.json").write_text(json.dumps({
                "records": [{"passed": True, "pool": "cmr_boundary_manifest",
                             "index": 0, "candidate": str(candidate_path)}],
            }))
            spec = PairSpec(dataset="test", tier="gold", target="cmr_myo",
                            image=image_path, label=label_path, myo=None,
                            relative_id="test", case_id="test", slice_position=0,
                            reviewed=True)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(build_profile([extract_pair(spec)])))
            report = pending.evaluate(dataset, candidate_root, triage,
                                      quality_profile=profile_path)
            self.assertEqual(report["quality"]["scored_volumes"], 1)
            self.assertEqual(report["records"][0]["quality_status"], "SCORED")
            self.assertEqual(label_path.read_bytes(), original_bytes)
            self.assertFalse(list(candidate_root.glob("tmp*")))

    def test_pending_evaluation_scores_existing_label_without_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, candidate_root = root / "dataset", root / "candidate"
            candidate_root.mkdir()
            yy, xx = np.ogrid[:32, :32]
            radius = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
            ring = (radius <= 10) & (radius >= 6)
            image = np.repeat((ring * 80).astype(np.float32)[..., None], 3, axis=-1)
            label = np.repeat(ring.astype(np.uint8)[..., None], 3, axis=-1)
            image_path = dataset / "images/cmr/CASE/1.nii.gz"
            label_path = dataset / "labels/cmr_checked/cmr_Myo_mask/CASE/1.nii.gz"
            for path, array in ((image_path, image), (label_path, label)):
                path.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
            original_bytes = label_path.read_bytes()
            png = dataset / "labels/check/待筛/cmr/CASE/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"png")
            manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("patient,volume\nCASE,1.nii.gz\n")
            triage = root / "triage.csv"
            triage.write_text("relative_key,ref_area\nCASE/1.nii.gz,100\n")
            (candidate_root / "candidate_validation.json").write_text(
                json.dumps({"records": []}))
            spec = PairSpec(dataset="test", tier="gold", target="cmr_myo",
                            image=image_path, label=label_path, myo=None,
                            relative_id="test", case_id="test", slice_position=0,
                            reviewed=True)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(build_profile([extract_pair(spec)])))
            report = pending.evaluate(dataset, candidate_root, triage,
                                      quality_profile=profile_path)
            item = report["records"][0]
            self.assertEqual(report["quality"]["scored_volumes"], 1)
            self.assertEqual(item["quality_status"], "SCORED")
            self.assertEqual(item["quality_before"], item["quality_after"])
            self.assertEqual(item["quality_delta"], 0.0)
            self.assertEqual(label_path.read_bytes(), original_bytes)

    def test_feedback_fingerprint_tracks_review_and_label_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            png = dataset / "labels/check/待筛/cmr/CASE/1.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"png")
            label = dataset / "labels/cmr_checked/cmr_Myo_mask/CASE/1.nii.gz"
            label.parent.mkdir(parents=True)
            label.write_bytes(b"old")
            first = watch.fingerprint(dataset)
            label.write_bytes(b"new-content")
            self.assertNotEqual(first, watch.fingerprint(dataset))
            checked = dataset / "labels/check/checked/cmr/CASE/1.png"
            checked.parent.mkdir(parents=True)
            png.replace(checked)
            self.assertNotEqual(first, watch.fingerprint(dataset))

    def test_feedback_fingerprint_changes_only_when_exemplar_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(watch, "ROOT", Path(tmp)):
            root = Path(tmp)
            dataset = root / "dataset"
            manual = root / "runtime/manual_annotations"
            manual.mkdir(parents=True)
            before = watch.fingerprint(dataset)
            (manual / "analyzed_training_corpus.jsonl").write_text(json.dumps({
                "edit_id": "example-1", "modality": "cmr", "target": "myo",
                "suggested_pattern": "local_expansion",
            }) + "\n")
            provenance = manual / "edits/example-1/provenance.json"
            provenance.parent.mkdir(parents=True)
            provenance.write_text(json.dumps({"edit_id": "example-1", "status": "STAGED"}))
            self.assertEqual(before, watch.fingerprint(dataset))
            provenance.write_text(json.dumps({
                "edit_id": "example-1", "status": "APPLIED", "applied_by": "reviewer",
                "applied_hash": "hash",
            }))
            self.assertNotEqual(before, watch.fingerprint(dataset))

    def test_spatial_core_only_routes_outside_unique_run(self):
        paths = [Path(str(i)) for i in range(7)]
        values = {"0": 0., "1": .8, "2": .8, "3": 0., "4": .8,
                  "5": .8, "6": 0.}
        with patch.object(spatial, "ring_fraction", side_effect=lambda p: values[p.name]):
            self.assertEqual(spatial.trusted_core(paths), ((1, 5), ""))
        values["6"] = .8
        with patch.object(spatial, "ring_fraction", side_effect=lambda p: values[p.name]):
            self.assertEqual(spatial.trusted_core(paths), ((1, 6), ""))

    def test_holdout_is_stratified_deterministic_and_disjoint(self):
        rows = []
        for i in range(123):
            rows.append({"key": f"a/{i}", "decision": "REVIEW", "after_area": 60,
                         "reference_area": 100, "after_cavity": .2})
        for i in range(40):
            rows.append({"key": f"b/{i}", "decision": "REVIEW", "after_area": 20,
                         "reference_area": 100, "after_cavity": .2})
        for i in range(30):
            rows.append({"key": f"c/{i}", "decision": "REVIEW", "after_area": 20,
                         "reference_area": 100, "after_cavity": .8})
        first = holdout.freeze({"records": rows})
        second = holdout.freeze({"records": list(reversed(rows))})
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"holdout": 58, "development": 135})
        self.assertFalse({r["key"] for r in first["holdout"]} &
                         {r["key"] for r in first["development"]})
        self.assertEqual(
            {group: sum(r["stratum"] == group for r in first["holdout"])
             for group in holdout.SIZES}, holdout.SIZES)


if __name__ == "__main__":
    unittest.main()
