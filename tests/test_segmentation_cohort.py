from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from app import segmentation_cohort as cohort
from app import quality_scoring
from app.paired_dataset import _discard
from app.segmentation_evaluation import compare_predictions


class CohortPackagingTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="synthetic-cohort-")).resolve()
        self.addCleanup(lambda: _discard(self.base))
        self.source = self.base / "source"
        self.dest = self.base / "frozen"
        self.config = self.base / "sources.yaml"
        self.targets = {}
        for target in sorted(cohort.TARGETS):
            image_dir = self.source / ("cmr" if target == "cmr_myo" else "lge")
            label_dir = self.source / target
            self.targets[target] = {"image_dir": str(image_dir), "label_dir": str(label_dir)}
            for i, name in enumerate(("SYNTHETIC_A", "SYNTHETIC_B")):
                image = np.arange(24, dtype=np.float32).reshape(4, 6) + i * 100
                if target == "cmr_myo":
                    image = np.stack((image, image + 1, image + 2), axis=-1)
                image_path = image_dir / name / "1.nii.gz"
                if not image_path.exists():
                    self.nifti(image_path, image)
                mask = np.zeros_like(image, dtype=np.uint8)
                if target != "lge_mi":
                    mask[1:3, 1:3] = 1
                self.nifti(label_dir / name / "1.nii.gz", mask)
        self.write_config()
        self.plan = {"schema": "segmentation-cohort-plan-v1", "owner_role": "domain-owner",
                     "config_sha256": cohort._sha_file(self.config), "selections": []}
        specs, _ = quality_scoring.load_specs(self.config)
        for spec in specs:
            name = spec.relative_id.split("/")[0]
            self.plan["selections"].append({"dataset": spec.dataset, "target": spec.target,
                "relative_id": spec.relative_id, "case_id": spec.case_id, "patient_key": name,
                "split": "train" if name == "SYNTHETIC_A" else "validation"})

    def nifti(self, path, data, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)

    def write_config(self, *, tier="gold", duplicate=False):
        data = {"name": "synthetic_site", "tier": tier, "targets": self.targets}
        self.config.write_text(yaml.safe_dump({"datasets": [data, data] if duplicate else [data]}), encoding="utf-8")
        if hasattr(self, "plan"):
            self.plan["config_sha256"] = cohort._sha_file(self.config)

    def freeze(self, **kwargs):
        return cohort.freeze_cohort(self.config, self.plan, self.dest, **kwargs)

    def snapshot(self):
        return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.rglob("*") if p.is_file()}

    def assert_no_export(self):
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.base.glob(".segmentation-cohort-*")), [])

    def rejected(self, pattern):
        with self.assertRaisesRegex(cohort.CohortError, pattern) as raised:
            self.freeze()
        self.assertNotIn("SYNTHETIC", str(raised.exception))
        self.assert_no_export()

    def test_exact_cycle_manifests_private_provenance_hashes_and_source_preservation(self):
        before, plan = self.snapshot(), copy.deepcopy(self.plan)
        result = self.freeze()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.plan, plan)
        self.assertEqual(result["status"], "FROZEN")
        self.assertNotIn("SYNTHETIC", json.dumps(result))
        self.assertEqual(result["baseline_exposure"], "UNKNOWN")
        self.assertEqual(result["final_test_independence"], "NOT_ESTABLISHED")
        provenance = json.loads((self.dest / "manifest.json").read_text())
        self.assertEqual(cohort._sha_file(self.dest / "manifest.json"), result["manifest"]["sha256"])
        self.assertEqual(provenance["config_sha256"], self.plan["config_sha256"])
        self.assertEqual(cohort._hash(provenance["splits"]), provenance["split_sha256"])
        private = json.loads((self.dest / "private/manifest.json").read_text())
        self.assertEqual(provenance["source_sha256"], cohort._hash(private["sources"]))
        self.assertEqual(provenance["private_manifest_sha256"], cohort._sha_file(self.dest / "private/manifest.json"))
        self.assertEqual(len(private["patients"]), 2)
        patient_ids = {}
        for target, splits in result["cohorts"].items():
            for split, binding in splits.items():
                path = Path(binding["path"])
                self.assertEqual(cohort._sha_file(path), binding["sha256"])
                data = json.loads(path.read_text())
                self.assertEqual(set(data), {"schema", "target", "cases"})
                self.assertEqual(data["schema"], "segmentation-cycle-cohort-v1")
                self.assertEqual(data["target"], target)
                row = data["cases"][0]
                self.assertEqual(set(row), {"case_id", "patient_id", "image", "reference"})
                self.assertEqual(patient_ids.setdefault(split, row["patient_id"]), row["patient_id"])
                for role in ("image", "reference"):
                    asset = Path(row[role]["path"])
                    self.assertTrue(asset.is_relative_to(self.dest / ("images" if role == "image" else "references")))
                    self.assertEqual(cohort._sha_file(asset), row[role]["sha256"])
                self.assertNotIn("SYNTHETIC", path.read_text())
        self.assertNotEqual(patient_ids["train"], patient_ids["validation"])
        self.assertEqual(patient_ids["train"], hashlib.sha256(b"SYNTHETIC_A").hexdigest())

    def test_empty_mi_retained_and_direct_prediction_comparison_compatible(self):
        result = self.freeze()
        data = json.loads(Path(result["cohorts"]["lge_mi"]["validation"]["path"]).read_text())
        row = data["cases"][0]
        self.assertFalse(np.asarray(nib.load(row["reference"]["path"]).dataobj).any())
        report = compare_predictions([dict(case_id=row["case_id"], patient_id=row["patient_id"],
            reference=row["reference"], parent=row["reference"], candidate=row["reference"])], target="lge_mi")
        self.assertEqual(report["cohort"]["empty_reference_cases"], 1)
        self.assertEqual(report["metrics"]["empty_reference_fp_rate"]["candidate"], 0)

    def test_explicit_subset_only_and_stable_patient_across_reexports(self):
        self.plan["selections"] = [r for r in self.plan["selections"] if r["target"] == "lge_mi" and r["split"] == "validation"]
        first = self.freeze()
        self.dest = self.base / "second"
        second = self.freeze()
        def member(result):
            return json.loads(Path(result["cohorts"]["lge_mi"]["validation"]["path"]).read_text())["cases"][0]
        self.assertEqual(member(first)["patient_id"], member(second)["patient_id"])
        self.assertEqual(member(first)["case_id"], member(second)["case_id"])
        self.assertEqual(set(first["cohorts"]), {"lge_mi"})
        self.assertEqual(len(list(self.dest.rglob("*.nii.gz"))), 2)

    def test_test_split_name_never_asserts_final_test_independence(self):
        for row in self.plan["selections"]:
            if row["split"] == "validation": row["split"] = "test"
        result = self.freeze()
        self.assertIn("test", result["cohorts"]["lge_mi"])
        self.assertEqual(result["final_test_independence"], "NOT_ESTABLISHED")
        self.assertNotIn("training_evaluation_patient_disjoint", json.dumps(result))

    def test_missing_identity_and_missing_asset_fail_without_partial_export(self):
        original = copy.deepcopy(self.plan)
        self.plan["selections"][0]["relative_id"] = "absent/1.nii.gz"
        self.rejected("missing_or_duplicate_source_identity")
        self.plan = original
        (Path(self.targets["lge_mi"]["image_dir"]) / "SYNTHETIC_A/1.nii.gz").unlink()
        self.rejected("cohort_input_or_io_error")

    def test_duplicate_plan_and_config_identity_fail(self):
        self.plan["selections"].append(copy.deepcopy(self.plan["selections"][0]))
        self.rejected("duplicate_selection")
        self.plan["selections"].pop()
        self.write_config(duplicate=True)
        self.rejected("missing_or_duplicate_source_identity")

    def test_non_gold_and_mismatched_patient_identity_fail(self):
        self.write_config(tier="automatic_qc")
        self.rejected("source_not_configured_gold")
        self.write_config()
        self.plan["selections"][0]["case_id"] = "wrong-case"
        self.rejected("source_patient_identity_mismatch")

    def test_patient_key_and_split_conflicts_fail(self):
        original = copy.deepcopy(self.plan)
        self.plan["selections"][0]["split"] = "test"
        self.rejected("assignment_conflict|patient_split_overlap")
        self.plan = original
        self.plan["selections"][0]["patient_key"] = "different-private-key"
        self.rejected("assignment_conflict")

    def test_shared_patient_key_cannot_receive_conflicting_split(self):
        self.plan["selections"] = self.plan["selections"][:2]
        self.plan["selections"][1]["patient_key"] = self.plan["selections"][0]["patient_key"]
        self.rejected("patient_split_overlap")

    def test_identical_image_under_different_patient_cannot_leak(self):
        images = Path(self.targets["lge_mi"]["image_dir"])
        (images / "SYNTHETIC_B/1.nii.gz").write_bytes((images / "SYNTHETIC_A/1.nii.gz").read_bytes())
        self.rejected("duplicate_image_for_target|image_patient_alias")

    def test_geometry_shape_affine_and_invalid_mask_fail(self):
        ref = Path(self.targets["lge_mi"]["label_dir"]) / "SYNTHETIC_A/1.nii.gz"
        for values, affine, reason in (
            (np.zeros((3, 6), np.uint8), np.eye(4), "geometry_mismatch"),
            (np.zeros((4, 6), np.uint8), np.diag([2, 1, 1, 1]), "geometry_mismatch"),
            (np.full((4, 6), 2, np.uint8), np.eye(4), "reference_not_binary"),
            (np.full((4, 6), np.nan, np.float32), np.eye(4), "invalid_image_values"),
        ):
            with self.subTest(reason=reason):
                self.nifti(ref, values, affine)
                self.rejected(reason)

    def test_trailing_singleton_axes_and_255_mask_preserved(self):
        ref = Path(self.targets["lge_mi"]["label_dir"]) / "SYNTHETIC_A/1.nii.gz"
        self.nifti(ref, np.full((4, 6, 1), 255, np.uint8))
        original = ref.read_bytes()
        result = self.freeze()
        data = json.loads(Path(result["cohorts"]["lge_mi"]["train"]["path"]).read_text())
        self.assertEqual(Path(data["cases"][0]["reference"]["path"]).read_bytes(), original)

    def test_config_hash_and_cache_binding_required(self):
        self.plan["config_sha256"] = "0" * 64
        self.rejected("config_hash_mismatch")
        self.plan["config_sha256"] = cohort._sha_file(self.config)
        self.plan["cache_sha256"] = "0" * 64
        self.rejected("cache_binding_required")

    def make_cache(self):
        from app.rsi_quality_benchmark import patient_split
        rows = [dict(dataset=r["dataset"], target=r["target"], case_id=r["case_id"],
                     relative_id=r["relative_id"], tier="gold", valid="True") for r in self.plan["selections"]]
        train, validation = patient_split(rows)
        splits = {r["case_id"]: split for split, items in (("train", train), ("validation", validation)) for r in items}
        for row in self.plan["selections"]: row["split"] = splits[row["case_id"]]
        path = self.base / "cache.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        self.plan["cache_sha256"] = cohort._sha_file(path)
        return path

    def test_cached_gold_exact_join_reuses_known_split_without_selecting_extra_cases(self):
        cache = self.make_cache()
        self.plan["selections"] = self.plan["selections"][:1]
        result = self.freeze(cache_path=cache)
        provenance = json.loads((self.dest / "manifest.json").read_text())
        self.assertEqual(provenance["cache_sha256"], cohort._sha_file(cache))
        self.assertEqual(sum(v["cases"] for splits in result["counts"].values() for v in splits.values()), 1)

    def test_cache_duplicate_missing_and_split_conflict_fail(self):
        cache = self.make_cache()
        original = cache.read_text()
        for contents in (original + original.splitlines()[1] + "\n", "\n".join(original.splitlines()[:1] + original.splitlines()[2:]) + "\n"):
            cache.write_text(contents)
            self.plan["cache_sha256"] = cohort._sha_file(cache)
            with self.assertRaisesRegex(cohort.CohortError, "cache_join"):
                self.freeze(cache_path=cache)
            self.assert_no_export()
        cache.write_text(original)
        self.plan["cache_sha256"] = cohort._sha_file(cache)
        self.plan["selections"][0]["split"] = "test"
        with self.assertRaisesRegex(cohort.CohortError, "cache_join"):
            self.freeze(cache_path=cache)
        self.assert_no_export()

    def test_source_and_config_changes_after_earlier_copy_fail(self):
        real_copy = cohort._copy
        original = self.snapshot()
        for change_config in (False, True):
            with self.subTest(config=change_config):
                calls, first = [], []
                def mutate(source, destination, expected):
                    real_copy(source, destination, expected)
                    calls.append(source)
                    if not first: first.append(source)
                    if len(calls) == len(self.plan["selections"]) * 2:
                        with (self.config if change_config else first[0]).open("ab") as f: f.write(b" ")
                with patch.object(cohort, "_copy", side_effect=mutate):
                    self.rejected("source_or_copy_changed|config_or_cache_changed")
                for path, (content, _) in original.items():
                    Path(path).write_bytes(content)
                self.write_config()

    def test_staged_tampering_is_detected(self):
        real_write = cohort._write
        def mutate(path, payload):
            digest = real_write(path, payload)
            if path.name == "manifest.sha256":
                private = path.parent / "private/manifest.json"
                private.write_bytes(b"{}")
            return digest
        with patch.object(cohort, "_write", side_effect=mutate):
            self.rejected("staged_manifest_changed")

    def test_existing_destination_and_source_overlap_never_overwritten(self):
        self.dest.mkdir()
        marker = self.dest / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(cohort.CohortError, "destination_exists"):
            self.freeze()
        self.assertEqual(marker.read_text(), "preserve")
        self.dest = self.source / "lge" / "new-export"
        self.rejected("destination_overlaps_source")

    def test_publish_race_preserves_competing_directory(self):
        publish = cohort._publish_noreplace
        def race(stage, destination):
            destination.mkdir()
            (destination / "winner.txt").write_text("other publisher")
            publish(stage, destination)
        with patch.object(cohort, "_publish_noreplace", side_effect=race):
            with self.assertRaises(cohort.CohortError): self.freeze()
        self.assertEqual((self.dest / "winner.txt").read_text(), "other publisher")
        self.assertEqual(list(self.base.glob(".segmentation-cohort-*")), [])

    def test_failure_mid_copy_removes_all_partial_outputs(self):
        with patch.object(cohort, "_copy", side_effect=OSError("private-source-name")):
            self.rejected("cohort_input_or_io_error:OSError")

    def test_path_links_rejected_when_supported(self):
        image = Path(self.targets["lge_mi"]["image_dir"]) / "SYNTHETIC_A/1.nii.gz"
        backup = image.with_name("original.bin")
        image.rename(backup)
        try:
            image.symlink_to(backup)
        except OSError:
            backup.rename(image)
            self.skipTest("symlink creation unavailable")
        self.rejected("cohort_input_or_io_error")

    def test_original_path_guard_applies_to_images_and_masks(self):
        root = self.source / "registration_copy"
        self.source.joinpath("lge").rename(root)
        for target in ("lge_myo", "lge_mi"): self.targets[target]["image_dir"] = str(root)
        self.write_config()
        self.rejected("cohort_input_or_io_error")

    def test_relative_prefix_resolution_reuses_existing_loader(self):
        old = self.source / "lge"
        old.rename(self.source / "stack")
        old.mkdir()
        (self.source / "stack").rename(old / "stack")
        for target in ("lge_myo", "lge_mi"):
            self.targets[target]["relative_prefix"] = "stack"
        self.write_config()
        for row in self.plan["selections"]:
            if row["target"].startswith("lge"):
                row["relative_id"] = "stack/" + row["relative_id"]
        self.freeze()

    def test_python39_grammar(self):
        ast.parse(Path(cohort.__file__).read_text(), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
