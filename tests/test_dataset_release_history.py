"""Release history survives live audits; every image and receipt is synthetic."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from app.segmentation_status import status
from scripts import dataset_release


class DatasetReleaseHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-release-history-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.addCleanup(self.writable)
        self.root = self.base / "project"
        self.dataset = self.base / "PRIVATE_SOURCE"
        self.destination = self.base / "PRIVATE_EXPORT"
        self.private = self.root / "runtime/dataset_release"
        self.key = "SYNTHETIC_ONLY/volume.nii.gz"
        self.paths = {}
        layout = {"cmr_image": "images/cmr", "lge_image": "images/lge",
                  "cmr_myo": "labels/cmr_checked/cmr_Myo_mask",
                  "lge_myo": "labels/lge_checked/lge_Myo_labels",
                  "lge_mi": "labels/lge_checked/lge_MI_labels"}
        for role, folder in layout.items():
            path = self.dataset / folder / self.key
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = (4, 5, 2, 3) if role.startswith("cmr") else (4, 5, 2)
            nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), path)
            self.paths[role] = path
        self.evidence = self.acceptance()
        self.collect = self.patch(dataset_release, "collect_evidence",
                                  side_effect=lambda *_: (self.evidence, {"synthetic": len(self.evidence)}))
        self.patch(dataset_release, "frozen_patients", return_value=set())
        self.network = self.patch(__import__("socket").socket, "connect", side_effect=AssertionError("network forbidden"))

    def patch(self, obj, name, **kwargs):
        patcher = mock.patch.object(obj, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def writable(self):
        for directory, folders, files in os.walk(self.base):
            Path(directory).chmod(0o700)
            for name in folders + files:
                path = Path(directory) / name
                path.chmod(0o700 if path.is_dir() else 0o600)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def snapshot(folder):
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def acceptance(self):
        hashes = {role: self.sha(path) for role, path in self.paths.items()}
        evidence = []
        for target in ("cmr_myo", "lge_myo", "lge_mi"):
            bindings = {"image": hashes[target[:3] + "_image"], "label": hashes[target]}
            if target.startswith("lge"):
                other = "mi" if target == "lge_myo" else "myo"
                bindings[other] = hashes["lge_" + other]
            evidence.append({"key": self.key, "target": target, "status": "ACCEPTED",
                             "scope": "whole_volume", "hashes": bindings,
                             "provenance": {"kind": "human", "record_id": "synthetic-" + target}})
        return evidence

    def change_source(self):
        path = self.paths["cmr_image"]
        source = nib.load(path)
        nib.save(nib.Nifti1Image(np.asanyarray(source.dataobj) + 1, source.affine), path)

    def read(self, filename):
        return json.loads((self.private / filename).read_text(encoding="utf-8"))

    def run_release(self, destination=None):
        return dataset_release.run(self.root, self.dataset, destination)

    def assert_path_free(self, summary):
        self.assertEqual(set(summary["release"]), {"release_id", "created_at", "manifest_sha256", "pair_count"})
        serialized = json.dumps(summary)
        for private_value in (str(self.dataset), str(self.destination), self.key, "PRIVATE_SOURCE", "PRIVATE_EXPORT"):
            self.assertNotIn(private_value, serialized)

    def test_freeze_then_changed_audit_preserves_export_without_claiming_a_match(self):
        sources = self.snapshot(self.dataset)
        released = self.run_release(self.destination)
        receipt = released["release"]
        self.assertEqual(released["status"], "RELEASED")
        self.assertTrue(released["release_created_by_current_audit"])
        self.assertEqual(receipt["pair_count"], 1)
        self.assertEqual(receipt["manifest_sha256"], self.sha(self.destination / "manifest.json"))
        self.assertEqual(receipt["release_id"], receipt["manifest_sha256"])
        self.assertIsNotNone(datetime.fromisoformat(receipt["created_at"]).tzinfo)
        self.assertEqual(self.read("latest_release.json"), receipt)
        self.assertEqual(self.snapshot(self.dataset), sources)
        frozen = self.snapshot(self.destination)

        self.change_source()
        audited = self.run_release()
        self.assertEqual(audited["status"], "AUDITED")
        self.assertFalse(audited["release_created_by_current_audit"])
        self.assertEqual(audited["summary"]["releasable_pairs"], 0)
        self.assertEqual(audited["release"], receipt)
        self.assertEqual(self.read("summary.json"), audited)
        self.assertEqual(self.read("latest_release.json"), receipt)
        self.assertEqual(self.snapshot(self.destination), frozen)
        # The existing UI projection still sees the last export after refresh.
        projected = status(self.root)
        self.assertEqual(projected["release"], receipt)
        self.assertEqual(projected["dataset_status"], "AUDITED")
        self.assert_path_free(audited)
        self.network.assert_not_called()

    def test_failed_freeze_keeps_prior_receipt_and_reports_current_audit_separately(self):
        released = self.run_release(self.destination)
        receipt_bytes = (self.private / "latest_release.json").read_bytes()
        frozen = self.snapshot(self.destination)
        self.change_source()
        with self.assertRaisesRegex(ValueError, "no_releasable_pairs"):
            self.run_release(self.base / "FAILED_EXPORT")
        current = self.read("summary.json")
        self.assertEqual(current["release"], released["release"])
        self.assertEqual(current["summary"]["releasable_pairs"], 0)
        self.assertEqual(current["status"], "AUDITED")
        self.assertFalse(current["release_created_by_current_audit"])
        self.assertEqual((self.private / "latest_release.json").read_bytes(), receipt_bytes)
        self.assertEqual(self.snapshot(self.destination), frozen)
        self.assertFalse((self.base / "FAILED_EXPORT").exists())
        self.assertEqual(self.run_release()["release"], released["release"])

    def test_failed_first_freeze_does_not_create_a_success_receipt(self):
        self.change_source()
        with self.assertRaisesRegex(ValueError, "no_releasable_pairs"):
            self.run_release(self.destination)
        self.assertIsNone(self.read("summary.json")["release"])
        self.assertFalse((self.private / "latest_release.json").exists())
        self.assertFalse(self.destination.exists())

    def test_failed_audit_leaves_both_previous_summaries_unchanged(self):
        self.run_release(self.destination)
        before = self.snapshot(self.private)
        with mock.patch.object(dataset_release, "audit_pairs", side_effect=OSError("synthetic audit failure")):
            with self.assertRaisesRegex(OSError, "synthetic audit failure"):
                self.run_release()
        self.assertEqual(self.snapshot(self.private), before)

    def test_new_success_replaces_latest_export_and_survives_stale_summary(self):
        first = self.run_release(self.destination)
        self.change_source()
        self.evidence = self.acceptance()
        second = self.run_release(self.base / "SECOND_EXPORT")
        self.assertNotEqual(first["release"]["release_id"], second["release"]["release_id"])
        self.assertEqual(second["release"]["manifest_sha256"], self.sha(self.base / "SECOND_EXPORT/manifest.json"))
        # A delayed/stale audit summary cannot supersede the dedicated receipt.
        dataset_release.save_journal(self.private / "summary.json", first)
        self.assertEqual(self.run_release()["release"], second["release"])
        self.assertEqual(self.read("latest_release.json"), second["release"])

    def test_successful_freeze_receipt_survives_final_summary_write_failure(self):
        first = self.run_release(self.destination)
        real_save = dataset_release.save_journal

        def fail_final_summary(path, value):
            if path.name == "summary.json" and value["status"] == "RELEASED":
                raise OSError("synthetic summary failure")
            real_save(path, value)

        second_destination = self.base / "SECOND_EXPORT"
        with mock.patch.object(dataset_release, "save_journal", side_effect=fail_final_summary):
            with self.assertRaisesRegex(OSError, "synthetic summary failure"):
                self.run_release(second_destination)
        receipt = self.read("latest_release.json")
        self.assertNotEqual(receipt, first["release"])
        self.assertEqual(receipt["manifest_sha256"], self.sha(second_destination / "manifest.json"))
        recovered = self.run_release()
        self.assertEqual(recovered["release"], receipt)
        self.assertEqual(recovered["status"], "AUDITED")
        self.assertFalse(recovered["release_created_by_current_audit"])

    def test_legacy_summary_migration_is_path_free_and_does_not_invent_creation_time(self):
        for hash_key in ("manifest_sha256", "manifest_sha"):
            with self.subTest(hash_key=hash_key):
                self.private.mkdir(parents=True, exist_ok=True)
                (self.private / "latest_release.json").unlink(missing_ok=True)
                dataset_release.save_journal(self.private / "summary.json", {
                    "status": "RELEASED", "generated_at": "2020-01-01T00:00:00+00:00",
                    "release": {hash_key: "a" * 64, "pair_count": 7,
                                "destination": str(self.destination), "manifest": {"source": str(self.dataset)}},
                })
                audited = self.run_release()
                receipt = audited["release"]
                self.assertEqual(receipt, {"release_id": "a" * 64, "created_at": None,
                                           "manifest_sha256": "a" * 64, "pair_count": 7})
                self.assertEqual(audited["summary"]["releasable_pairs"], 1)
                self.assertEqual(audited["status"], "AUDITED")
                self.assertFalse(audited["release_created_by_current_audit"])
                self.assertEqual(self.read("latest_release.json"), receipt)
                self.assertEqual(self.run_release()["release"], receipt)
                self.assert_path_free(audited)

    def test_malformed_legacy_summaries_never_fabricate_release_history(self):
        self.private.mkdir(parents=True)
        values = ["{", "[]", "null", json.dumps({"release": ["bad"]})]
        for receipt in ({"manifest_sha256": "invalid", "pair_count": 2},
                        {"manifest_sha256": "a" * 64, "pair_count": True},
                        {"manifest_sha256": "a" * 64, "pair_count": -1},
                        {"manifest_sha256": "a" * 64, "pair_count": "2"}):
            values.append(json.dumps({"release": receipt}))
        for value in values:
            with self.subTest(value=value):
                (self.private / "summary.json").write_text(value, encoding="utf-8")
                audited = self.run_release()
                self.assertIsNone(audited["release"])
                self.assertFalse(audited["release_created_by_current_audit"])
                self.assertFalse((self.private / "latest_release.json").exists())


if __name__ == "__main__":
    unittest.main()
