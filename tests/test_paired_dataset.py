from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app import paired_dataset as paired


class PairedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        self.dataset.mkdir()
        self.destination = self.base / "release"
        self.key = "SYNTHETIC_A/level.nii.gz"
        self.paths = self.make_bundle(self.key)
        self.evidence = self.make_evidence(self.key)
        self.addCleanup(self.make_writable)

    def make_writable(self):
        # Releases are intentionally read-only, including on Windows.
        for directory, dirs, files in os.walk(self.base):
            Path(directory).chmod(0o700)
            for name in dirs:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o700)
            for name in files:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o600)

    def make_bundle(self, key):
        # Independent layout expectations: do not derive fixtures from LAYOUT.
        directories = {
            "cmr_image": "images/cmr",
            "lge_image": "images/lge",
            "cmr_myo": "labels/cmr_checked/cmr_Myo_mask",
            "lge_myo": "labels/lge_checked/lge_Myo_labels",
            "lge_mi": "labels/lge_checked/lge_MI_labels",
        }
        paths = {}
        for role, relative in directories.items():
            path = self.dataset / relative / key
            path.parent.mkdir(parents=True, exist_ok=True)
            cmr = role.startswith("cmr")
            shape = (4, 5, 2, 3) if cmr else (6, 5, 2)
            affine = np.eye(4) if cmr else np.diag([2., 2., 3., 1.])
            data = np.ones(shape, dtype=np.float32)
            if role.endswith("image"):
                data *= 17
            nib.save(nib.Nifti1Image(data, affine), str(path))
            paths[role] = path
        return paths

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_evidence(self, key):
        hashes = {role: self.sha(self.dataset / relative / key)
                  for role, relative in paired.LAYOUT.items()}
        result = []
        for target, kind in zip(paired.TARGETS, ("human", "automatic_qc", "doctor_reference")):
            bindings = {"image": hashes[target[:3] + "_image"], "label": hashes[target]}
            if target == "lge_myo":
                bindings["mi"] = hashes["lge_mi"]
            elif target == "lge_mi":
                bindings["myo"] = hashes["lge_myo"]
            result.append({"key": key, "target": target, "status": "ACCEPTED",
                           "scope": "whole_volume", "hashes": bindings,
                           "provenance": {"kind": kind, "record_id": "synthetic-" + target,
                                          "reviewer": "synthetic reviewer"}})
        return result

    def rewrite(self, role, data=None, affine=None):
        old = nib.load(str(self.paths[role]))
        data = np.asanyarray(old.dataobj).copy() if data is None else data
        affine = old.affine.copy() if affine is None else affine
        nib.save(nib.Nifti1Image(data, affine), str(self.paths[role]))

    def audit(self, evidence=None, **kwargs):
        return paired.audit_pairs(self.dataset, self.evidence if evidence is None else evidence, **kwargs)

    def reasons(self, audit, target):
        return audit["rows"][0]["blockers"][target]

    def assert_unpublished(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.base.glob(".paired-release-*")), [])

    def test_exact_bundle_with_distinct_modalities_and_all_provenance_kinds(self):
        before = {role: self.sha(path) for role, path in self.paths.items()}
        report = self.audit()
        self.assertEqual(report["summary"]["complete_files"], 1)
        self.assertEqual(report["summary"]["releasable_pairs"], 1)
        self.assertIn("not proof of anatomical registration", report["pairing"])
        self.assertEqual(report["rows"][0]["hashes"], before)
        self.assertEqual(report["rows"][0]["provenance"]["lge_mi"][0]["kind"], "doctor_reference")
        self.assertEqual({role: self.sha(path) for role, path in self.paths.items()}, before)
        self.assertNotIn("SYNTHETIC_A", json.dumps(report["summary"]))
        self.assertNotIn("synthetic reviewer", json.dumps(report["summary"]))
        json.dumps(report, allow_nan=False)

    def test_checked_folders_are_not_evidence(self):
        report = self.audit([])
        self.assertEqual(report["summary"]["releasable_pairs"], 0)

    def test_trailing_singleton_geometry_is_equivalent(self):
        self.rewrite("lge_image", data=np.ones((6,5,1),dtype=np.float32))
        self.rewrite("lge_myo", data=np.ones((6,5),dtype=np.uint8))
        self.rewrite("lge_mi", data=np.ones((6,5,1,1),dtype=np.uint8))
        report = self.audit(self.make_evidence(self.key))
        self.assertEqual(report["summary"]["releasable_pairs"], 1)

    def test_missing_evidence_is_explicit(self):
        report = self.audit([])
        self.assertEqual(report["summary"]["complete_files"], 1)
        self.assertEqual(report["summary"]["releasable_pairs"], 0)
        for target in paired.TARGETS:
            self.assertEqual(report["summary"]["blockers_by_target"][target], {"missing_evidence": 1})

    def test_missing_each_of_five_files_blocks_bundle(self):
        for role, path in self.paths.items():
            with self.subTest(role=role):
                original = path.read_bytes()
                path.unlink()
                report = self.audit()
                self.assertEqual(report["summary"]["complete_files"], 0)
                self.assertEqual(report["summary"]["releasable_pairs"], 0)
                target = role if role in paired.TARGETS else role[:3] + "_myo"
                self.assertIn("missing_file", self.reasons(report, target))
                path.write_bytes(original)

    def test_exact_key_includes_patient_directory_and_extension(self):
        original = self.paths["lge_mi"]
        other = original.parent.parent / "SYNTHETIC_B" / original.name
        other.parent.mkdir()
        original.rename(other)
        report = self.audit()
        self.assertEqual(report["summary"]["keys"], 2)
        self.assertEqual(report["summary"]["complete_files"], 0)
        other.rename(original.with_suffix(""))
        report = self.audit()
        self.assertEqual(report["summary"]["keys"], 2)
        self.assertEqual(report["summary"]["releasable_pairs"], 0)

    def test_image_or_label_changes_stale_evidence(self):
        for role in paired.LAYOUT:
            with self.subTest(role=role):
                path = self.paths[role]
                original = path.read_bytes()
                old = nib.load(str(path))
                data = np.asanyarray(old.dataobj).copy()
                data.flat[0] = 19 if role.endswith("image") else 0
                self.rewrite(role, data)
                report = self.audit()
                self.assertEqual(report["summary"]["complete_files"], 1)
                self.assertEqual(report["summary"]["releasable_pairs"], 0)
                target = role if role in paired.TARGETS else role[:3] + "_myo"
                self.assertIn("stale_evidence_hash", self.reasons(report, target))
                path.write_bytes(original)

    def test_lge_evidence_requires_current_counterpart_in_both_directions(self):
        for changed, dependent in (("lge_mi", "lge_myo"), ("lge_myo", "lge_mi")):
            with self.subTest(changed=changed):
                original = self.paths[changed].read_bytes()
                self.rewrite(changed, np.zeros((6, 5, 2), dtype=np.float32))
                current = self.make_evidence(self.key)
                # The changed target is reviewed again; the other review is stale.
                current = [old if old["target"] == dependent else new
                           for old, new in zip(self.evidence, current)]
                report = self.audit(current)
                self.assertIn("stale_dependency_hash", self.reasons(report, dependent))
                self.assertEqual(self.reasons(report, changed), [])
                self.paths[changed].write_bytes(original)

    def test_hash_schema_and_counterpart_are_mandatory(self):
        for index, field in ((0, "image"), (0, "label"), (1, "mi"), (2, "myo")):
            for value in (None, "", "not-a-sha", "f" * 64):
                with self.subTest(index=index, field=field, value=value):
                    records = copy.deepcopy(self.evidence)
                    if value is None:
                        del records[index]["hashes"][field]
                    else:
                        records[index]["hashes"][field] = value
                    self.assertEqual(self.audit(records)["summary"]["releasable_pairs"], 0)

    def test_evidence_status_scope_and_provenance_fail_closed(self):
        changes = [("status", "PROMOTED"), ("status", "REJECTED"),
                   ("scope", "frame"), ("scope", "partial_frame"), ("scope", None),
                   ("provenance", {}), ("provenance", {"kind": "folder", "record_id": "x"}),
                   ("provenance", {"kind": "human", "record_id": " "})]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                records = copy.deepcopy(self.evidence)
                records[0][field] = value
                records[0]["frames"] = [0]
                self.assertEqual(self.audit(records)["summary"]["releasable_pairs"], 0)

    def test_old_partial_and_stale_records_do_not_override_current_evidence(self):
        old = copy.deepcopy(self.evidence)
        old[0]["scope"] = "frame"
        old[1]["hashes"]["mi"] = "0" * 64
        report = self.audit(old + self.evidence)
        self.assertEqual(report["summary"]["releasable_pairs"], 1)
        self.assertEqual(len(report["rows"][0]["accepted_evidence"]["cmr_myo"]), 1)
        self.evidence[0]["provenance"]["record_id"] = "changed by caller"
        self.assertNotEqual(report["rows"][0]["provenance"]["cmr_myo"][0]["record_id"], "changed by caller")

    def test_spatial_and_time_dimensions_must_match(self):
        for shape in ((4, 5, 2), (4, 5, 2, 1), (4, 5, 2, 3, 2), (5, 4, 2, 3)):
            with self.subTest(shape=shape):
                self.rewrite("cmr_myo", np.ones(shape, dtype=np.float32))
                report = self.audit(self.make_evidence(self.key))
                self.assertIn("shape_mismatch", self.reasons(report, "cmr_myo"))
                self.assertEqual(report["summary"]["complete_files"], 1)

    def test_affine_must_match_within_modality(self):
        affine = np.diag([2., 2., 3., 1.])
        affine[0, 3] = 0.1
        self.rewrite("lge_myo", affine=affine)
        report = self.audit(self.make_evidence(self.key))
        self.assertIn("affine_mismatch", self.reasons(report, "lge_myo"))

    def test_finite_images_and_masks(self):
        for role in paired.LAYOUT:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(role=role, value=value):
                    original = self.paths[role].read_bytes()
                    data = np.asanyarray(nib.load(str(self.paths[role])).dataobj).copy()
                    data.flat[0] = value
                    self.rewrite(role, data)
                    report = self.audit(self.make_evidence(self.key))
                    self.assertEqual(report["summary"]["releasable_pairs"], 0)
                    self.assertIn("nonfinite_or_empty_data", report["rows"][0]["file_errors"][role])
                    self.paths[role].write_bytes(original)

    def test_binary_encodings_accept_zero_one_or_zero_255_not_mixed(self):
        for values, valid in (([0, 1], True), ([0, 255], True), ([0], True),
                              ([0, 1, 255], False), ([0, 2], False),
                              ([0, -1], False), ([0, 0.5], False)):
            with self.subTest(values=values):
                data = np.resize(np.array(values, dtype=np.float32), (6, 5, 2))
                self.rewrite("lge_mi", data)
                report = self.audit(self.make_evidence(self.key))
                self.assertEqual(report["summary"]["releasable_pairs"], int(valid))

    def test_corrupt_nifti_is_blocked_with_complete_file_count(self):
        self.paths["cmr_myo"].write_bytes(b"not a NIfTI")
        report = self.audit(self.make_evidence(self.key))
        self.assertEqual(report["summary"]["complete_files"], 1)
        self.assertIn("unreadable_or_unsafe_file", self.reasons(report, "cmr_myo"))

    def test_nonfinite_affine_is_blocked_and_audit_remains_finite_json(self):
        path = self.paths["lge_mi"]
        raw = bytearray(gzip.decompress(path.read_bytes()))
        # NIfTI-1 srow_x[0], independent of the data array itself.
        struct.pack_into("<f", raw, 280, float("nan"))
        path.write_bytes(gzip.compress(bytes(raw)))
        report = self.audit(self.make_evidence(self.key))
        self.assertIn("nonfinite_affine", self.reasons(report, "lge_mi"))
        self.assertEqual(report["summary"]["releasable_pairs"], 0)
        json.dumps(report, allow_nan=False)

    def test_source_mutation_during_audit_is_blocked(self):
        sha = paired._sha_file
        reads = 0

        def unstable(path):
            nonlocal reads
            if path == self.paths["cmr_image"]:
                reads += 1
                if reads == 2:
                    self.rewrite("cmr_image", np.zeros((4, 5, 2, 3), dtype=np.float32))
            return sha(path)

        with patch.object(paired, "_sha_file", side_effect=unstable):
            report = self.audit()
        self.assertIn("source_changed", self.reasons(report, "cmr_myo"))
        self.assertEqual(report["summary"]["releasable_pairs"], 0)

    def test_invalid_record_identity_and_nonjson_evidence_fail_closed(self):
        for extra in (None, {}, {"key": self.key, "target": "unexpected"}):
            with self.subTest(extra=extra):
                self.assertEqual(self.audit(self.evidence + [extra])["summary"]["releasable_pairs"], 0)
        with self.assertRaises(ValueError):
            self.audit([{"bad": float("nan")}])
        with self.assertRaises(ValueError):
            self.audit(excluded_patients="SYNTHETIC_A")

    def test_exclusions_apply_to_whole_patient_with_exact_identity(self):
        self.make_bundle("SYNTHETIC_AB/second.nii")
        evidence = self.evidence + self.make_evidence("SYNTHETIC_AB/second.nii")
        report = self.audit(evidence, excluded_patients={"SYNTHETIC_A"})
        self.assertEqual(report["summary"]["complete_files"], 2)
        self.assertEqual(report["summary"]["releasable_pairs"], 1)
        self.assertEqual(report["summary"]["excluded_pairs"], 1)
        self.assertNotIn("SYNTHETIC_A", json.dumps(report["summary"]))

    def test_invalid_and_escaping_evidence_keys_are_never_read(self):
        for key in ("../outside.nii", "/tmp/outside.nii", "A/../outside.nii",
                    "A\\outside.nii", "C:/outside.nii", "A//outside.nii",
                    "A/./outside.nii", "volume.nii", "A./outside.nii"):
            with self.subTest(key=key):
                records = copy.deepcopy(self.evidence)
                records[0]["key"] = key
                report = self.audit(records)
                self.assertIn("invalid_evidence_identity", report["summary"]["input_issues"])
                self.assertEqual(report["summary"]["releasable_pairs"], 0)

    def test_symlink_source_file_is_rejected_even_inside_dataset(self):
        source = self.paths["cmr_myo"]
        saved = self.dataset / "saved.nii.gz"
        shutil.copyfile(source, saved)
        source.unlink()
        try:
            source.symlink_to(saved)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        report = self.audit()
        self.assertEqual(report["summary"]["releasable_pairs"], 0)
        self.assertIn("unsafe_path", report["summary"]["input_issues"])

    def test_directory_symlink_escape_and_linked_dataset_root_are_rejected(self):
        directory = self.dataset / "images/cmr"
        saved = self.base / "saved"
        directory.rename(saved)
        try:
            directory.symlink_to(saved, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlinks unavailable: " + str(exc))
        self.assertEqual(self.audit()["summary"]["releasable_pairs"], 0)
        alias = self.base / "alias"
        alias.symlink_to(self.dataset, target_is_directory=True)
        with self.assertRaises(ValueError):
            paired.audit_pairs(alias, self.evidence)

    def test_reparse_points_are_rejected(self):
        original = Path.lstat
        target = self.dataset / "images"

        def reparse(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == target:
                class Info:
                    st_mode = result.st_mode
                    st_file_attributes = 0x400
                return Info()
            return result

        with patch.object(Path, "lstat", reparse):
            report = self.audit()
        self.assertEqual(report["summary"]["releasable_pairs"], 0)
        self.assertIn("unsafe_path", report["summary"]["input_issues"])

    def test_freeze_copies_bytes_pseudonymizes_and_binds_private_provenance(self):
        report = self.audit()
        result = paired.freeze_release(self.dataset, report, self.destination)
        manifest_path = self.destination / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        self.assertEqual(self.sha(manifest_path), result["manifest_sha256"])
        self.assertTrue((self.destination / "manifest.sha256").read_text().startswith(result["manifest_sha256"]))
        self.assertNotIn("SYNTHETIC_A", manifest_path.read_text())
        self.assertNotIn("synthetic reviewer", manifest_path.read_text())
        private = self.destination / "private/manifest.json"
        self.assertEqual(self.sha(private), manifest["private_manifest_sha256"])
        self.assertEqual(json.loads(private.read_bytes())["bundles"][0]["source_key"], self.key)
        self.assertEqual(json.loads(private.read_bytes())["bundles"][0]["provenance"], report["rows"][0]["provenance"])
        for role, info in manifest["bundles"][0]["files"].items():
            target = self.destination / info["path"]
            self.assertEqual(target.read_bytes(), self.paths[role].read_bytes())
            self.assertEqual(self.sha(target), info["sha256"])
            self.assertNotEqual(target.stat().st_ino, self.paths[role].stat().st_ino)
            self.assertFalse(target.stat().st_mode & stat.S_IWRITE)
            self.assertNotIn("level", info["path"])
        frozen_hash = manifest["bundles"][0]["files"]["lge_mi"]["sha256"]
        self.rewrite("lge_mi", np.zeros((6, 5, 2), dtype=np.float32))
        frozen_path = self.destination / manifest["bundles"][0]["files"]["lge_mi"]["path"]
        self.assertEqual(self.sha(frozen_path), frozen_hash)

    def test_patient_pseudonym_stable_within_release_and_bundles_distinct(self):
        keys = ("SYNTHETIC_A/second.nii", "SYNTHETIC_B/third.nii")
        records = list(self.evidence)
        for key in keys:
            self.make_bundle(key)
            records += self.make_evidence(key)
        result = paired.freeze_release(self.dataset, self.audit(records), self.destination)
        rows = result["manifest"]["bundles"]
        self.assertEqual(rows[0]["patient_id"], rows[1]["patient_id"])
        self.assertNotEqual(rows[1]["patient_id"], rows[2]["patient_id"])
        self.assertEqual(len({row["bundle_id"] for row in rows}), 3)

    def test_export_is_owner_only_before_publication_even_with_public_parent(self):
        if os.name != "nt":
            self.base.chmod(0o755)
        source_state = {role: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
                        for role, path in self.paths.items()}
        chmod, publish = Path.chmod, paired._publish_noreplace
        applied_modes = {}

        def track_mode(path, mode, *args, **kwargs):
            applied_modes[path] = mode
            return chmod(path, mode, *args, **kwargs)

        def check_before_publish(staging, destination):
            self.assertFalse(destination.exists())
            for path in (staging, *staging.rglob("*")):
                expected = 0o500 if path.is_dir() else 0o400
                # Windows cannot enforce POSIX bits, but must still request
                # the restrictive modes rather than making private data public.
                self.assertEqual(applied_modes[path], expected)
                if os.name != "nt":
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected)
            publish(staging, destination)

        with patch.object(Path, "chmod", track_mode), \
                patch.object(paired, "_publish_noreplace", side_effect=check_before_publish):
            paired.freeze_release(self.dataset, self.audit(), self.destination)
        for path in (self.destination, *self.destination.rglob("*")):
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o500 if path.is_dir() else 0o400)
            elif path.is_file():
                self.assertFalse(path.stat().st_mode & stat.S_IWRITE)
        self.assertEqual({role: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
                          for role, path in self.paths.items()}, source_state)

    def test_freeze_only_releasable_subset(self):
        self.make_bundle("SYNTHETIC_B/second.nii")
        report = self.audit()
        self.assertEqual(report["summary"]["complete_files"], 2)
        result = paired.freeze_release(self.dataset, report, self.destination)
        self.assertEqual(result["releasable_pairs"], 1)

    def test_freeze_rejects_empty_or_all_excluded_audit(self):
        for report in (self.audit([]), self.audit(excluded_patients={"SYNTHETIC_A"})):
            with self.assertRaises(ValueError):
                paired.freeze_release(self.dataset, report, self.destination)
            self.assert_unpublished()

    def test_freeze_rejects_stale_audit_and_mutated_audit(self):
        report = self.audit()
        tampered = copy.deepcopy(report)
        tampered["rows"][0]["paths"]["cmr_image"] = str(self.base / "outside.nii")
        with self.assertRaisesRegex(ValueError, "audit_changed"):
            paired.freeze_release(self.dataset, tampered, self.destination)
        self.rewrite("lge_mi", np.zeros((6, 5, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "audit_stale"):
            paired.freeze_release(self.dataset, report, self.destination)
        self.assert_unpublished()

    def test_forged_releasable_flags_cannot_bypass_fresh_audit(self):
        report = self.audit([])
        report["rows"][0]["releasable"] = True
        report.pop("audit_sha256")
        report["audit_sha256"] = paired._digest(report)
        with self.assertRaisesRegex(ValueError, "audit_stale"):
            paired.freeze_release(self.dataset, report, self.destination)
        self.assert_unpublished()

    def test_freeze_exclusion_is_preserved_in_subset(self):
        key = "SYNTHETIC_B/level.nii.gz"
        self.make_bundle(key)
        report = self.audit(self.evidence + self.make_evidence(key),
                            excluded_patients={"SYNTHETIC_A"})
        result = paired.freeze_release(self.dataset, report, self.destination)
        self.assertEqual(result["releasable_pairs"], 1)
        private = json.loads((self.destination / "private/manifest.json").read_bytes())
        self.assertEqual([row["source_key"] for row in private["bundles"]], [key])

    def test_freeze_rejects_existing_destination_and_destination_in_dataset(self):
        report = self.audit()
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            paired.freeze_release(self.dataset, report, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])
        for destination in (self.dataset, self.dataset / "export"):
            with self.assertRaises(ValueError):
                paired.freeze_release(self.dataset, report, destination)

    def test_publication_race_never_overwrites_even_empty_destination(self):
        publish = paired._publish_noreplace

        def race(staging, destination):
            self.assertTrue((staging / "manifest.json").is_file())
            destination.mkdir()
            publish(staging, destination)

        with patch.object(paired, "_publish_noreplace", side_effect=race):
            with self.assertRaises(FileExistsError):
                paired.freeze_release(self.dataset, self.audit(), self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(list(self.base.glob(".paired-release-*")), [])

    def test_source_change_after_earlier_copy_is_caught(self):
        original = shutil.copyfile
        count = 0

        def changing(source, target):
            nonlocal count
            original(source, target)
            count += 1
            if count == 5:
                self.rewrite("cmr_image", np.zeros((4, 5, 2, 3), dtype=np.float32))

        with patch.object(paired.shutil, "copyfile", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "source_or_copy_changed"):
                paired.freeze_release(self.dataset, self.audit(), self.destination)
        self.assert_unpublished()

    def test_copy_mutation_and_copy_failure_never_publish_partial_release(self):
        original = shutil.copyfile
        for fail in (False, True):
            def broken(source, target):
                original(source, target)
                if fail:
                    raise OSError("synthetic copy failure")
                target.write_bytes(b"damaged copy")

            with self.subTest(fail=fail), patch.object(paired.shutil, "copyfile", side_effect=broken):
                with self.assertRaisesRegex(OSError if fail else ValueError,
                                            "synthetic copy failure" if fail else "source_or_copy_changed"):
                    paired.freeze_release(self.dataset, self.audit(), self.destination)
            self.assert_unpublished()

    def test_manifest_mutation_before_publication_is_detected(self):
        seal = paired._seal
        for relative in ("manifest.json", "private/manifest.json", "manifest.sha256"):
            def corrupt(staging):
                (staging / relative).write_text("{}")
                seal(staging)

            with self.subTest(relative=relative), patch.object(paired, "_seal", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "manifest_changed"):
                    paired.freeze_release(self.dataset, self.audit(), self.destination)
            self.assert_unpublished()

    def test_python39_syntax_and_postponed_annotations(self):
        for path in (Path(paired.__file__), Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
        self.assertIsInstance(paired.audit_pairs.__annotations__["excluded_patients"], str)


if __name__ == "__main__":
    unittest.main()
