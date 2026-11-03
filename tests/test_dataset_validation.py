from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app import dataset_validation as validator
from app import paired_dataset as paired
from app.check_auto_apply import MIN_QUALITY_SCORE
from app.rsi_kernel import canonical_hash
from app.segmentation_evidence import collect_evidence


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-validation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / "data"
        self.dataset.mkdir()
        self.key = "SYNTHETIC_A/level.nii.gz"
        self.paths = self.bundle(self.key)
        self.version = "calibration-A"
        self.scores = {target: MIN_QUALITY_SCORE for target in paired.TARGETS}
        self.calibration = self.mock("app.lge_execution.calibration", side_effect=self.calibrate)
        self.grade = self.mock("app.quality_scoring.score_record",
                               side_effect=lambda features, profile: {"score": self.scores[features["target"]]})
        self.network = self.mock("socket.socket.connect", side_effect=AssertionError("network forbidden"))

    def mock(self, name, **kwargs):
        item = patch(name, **kwargs)
        self.addCleanup(item.stop)
        return item.start()

    def calibrate(self, root, stage):
        self.assertEqual(root, self.root)
        self.assertIn(stage, ("cmr_paired", "lge_myo", "lge_mi"))
        return {"synthetic_profile": stage}, self.version + ":" + stage

    def bundle(self, key, *, cmr_shape=(12, 13, 1, 3)):
        # Independent layout expectations, including native singleton axes.
        layout = {"cmr_image": "images/cmr", "lge_image": "images/lge",
                  "cmr_myo": "labels/cmr_checked/cmr_Myo_mask",
                  "lge_myo": "labels/lge_checked/lge_Myo_labels",
                  "lge_mi": "labels/lge_checked/lge_MI_labels"}
        paths = {}
        for role, directory in layout.items():
            path = self.dataset / directory / key
            path.parent.mkdir(parents=True, exist_ok=True)
            shape = cmr_shape if role.startswith("cmr") else (12, 13, 1)
            data = np.ones(shape, dtype=np.float32)
            if role.endswith("image"):
                data *= 17
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            paths[role] = path
        return paths

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def audit(self, evidence=None, **kwargs):
        return paired.audit_pairs(self.dataset, evidence or [], **kwargs)

    def run_batch(self, audit=None, **kwargs):
        return validator.validate_batch(self.root, self.dataset, self.audit() if audit is None else audit, **kwargs)

    def journal(self, name="receipts"):
        path = self.root / "runtime/dataset_acceptance" / (name + ".jsonl")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def acceptances(self):
        return [r for r in self.journal() if r["status"] == "ACCEPTED"]

    def snapshot(self):
        return {p.relative_to(self.dataset).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.dataset.rglob("*") if p.is_file()}

    def bucket(self, folder, modality, name):
        path = self.dataset / "labels/check" / folder / modality / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic review PNG")
        return path

    def freeze_cmr(self, key=None):
        from app.check_validation import _digest
        rows = [{"key": key or "SYNTHETIC_A/another-level.nii.gz", "stratum": "synthetic"}]
        self.write(self.root / "runtime/check_iterations/holdout_v1.json",
                   {"holdout": rows, "holdout_sha256": _digest(rows)})

    def mutate_image(self):
        image = nib.load(self.paths["cmr_image"])
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj) + 1, image.affine), self.paths["cmr_image"])

    def test_acceptance_roundtrip_threshold_native_singletons_no_source_writes(self):
        self.bucket("checked", "cmr", "SYNTHETIC_A/level.png")
        before = self.snapshot()
        result = self.run_batch()
        self.assertEqual(result["counts"], {"accepted_pairs": 1, "accepted_targets": 3})
        self.assertEqual(result["production_labels_written"], 0)
        self.assertEqual(result["provider_calls"], 0)
        self.assertEqual(self.snapshot(), before)
        self.network.assert_not_called()
        self.assertEqual(self.grade.call_count, 3)
        self.assertEqual(self.grade.call_args_list[0].args[0]["frames"], 3)
        self.assertEqual(self.grade.call_args_list[1].args[0]["frames"], 1)
        for call in self.grade.call_args_list:
            target = call.args[0]["target"]
            self.assertEqual(call.args[1], {"synthetic_profile": "cmr_paired" if target == "cmr_myo" else target})
        evidence, meta = collect_evidence(self.root, self.dataset)
        self.assertEqual(meta["normalized_acceptances"], 3)
        accepted = self.audit(evidence)
        self.assertEqual(accepted["summary"]["releasable_pairs"], 1)
        for record in evidence:
            self.assertEqual(record["scope"], "whole_volume")
            self.assertEqual(record["provenance"]["threshold"], MIN_QUALITY_SCORE)
            self.assertEqual(record["provenance"]["kind"], "automatic_qc")
            self.assertFalse(record["provenance"]["label_write"])
        json.dumps(result, allow_nan=False)
        again = self.run_batch(accepted)
        self.assertEqual(again["attempted"], 0)
        self.assertEqual(len(self.acceptances()), 3)

    def test_sparse_phase_never_certifies_unannotated_cine(self):
        self.paths = self.bundle(self.key, cmr_shape=(12, 13, 1, 8))
        mask = np.zeros((12, 13, 1, 8), dtype=np.float32)
        mask[3:8, 3:8, 0, :2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), self.paths["cmr_myo"])
        self.scores = dict.fromkeys(paired.TARGETS, 100)
        before = self.snapshot()
        result = self.run_batch()
        self.assertEqual(result["counts"], {"partial_volume_exclusion": 1})
        self.assertEqual(self.acceptances(), [])
        self.assertEqual(self.journal("attempts")[-1]["status"], "INELIGIBLE")
        self.grade.assert_not_called()
        self.assertEqual(self.run_batch()["attempted"], 0)
        self.assertEqual(self.snapshot(), before)

    def test_cached_release_report_envelope_preserves_core_evidence_and_provenance(self):
        from scripts.dataset_release import run
        # Start with actual accepted evidence, not a simplified alternate schema.
        self.run_batch()
        run(self.root, self.dataset)
        cached = json.loads((self.root / "runtime/dataset_release/audit.json").read_text(encoding="utf-8"))
        self.assertEqual(set(cached["rows"][0]["accepted_evidence"]), set(paired.TARGETS))
        self.assertEqual(len(cached["rows"][0]["provenance"]["lge_mi"]), 1)
        before = copy.deepcopy(cached)
        # Calibration change forces a real new attempt using the cached report.
        self.version = "calibration-B"
        self.assertEqual(self.run_batch(cached)["counts"]["accepted_pairs"], 1)
        self.assertEqual(cached, before)
        cached["unrecognized_annotation"] = True
        with self.assertRaisesRegex(ValueError, "audit_changed"):
            self.run_batch(cached)

    def test_missing_runtime_parent_is_created_before_locking(self):
        self.assertFalse((self.root / "runtime").exists())
        self.assertEqual(self.run_batch()["attempted"], 1)
        self.assertTrue((self.root / "runtime/dataset_acceptance/.registry.lock").is_file())

    def test_batch_reinspects_only_selected_bundle_not_whole_cached_dataset(self):
        self.bundle("SYNTHETIC_B/other.nii.gz")
        cached = self.audit()
        with patch.object(paired, "audit_pairs", side_effect=AssertionError("full rescan forbidden")), \
             patch.object(paired, "_inspect", wraps=paired._inspect) as inspect:
            self.assertEqual(self.run_batch(cached, limit=1)["attempted"], 1)
        self.assertEqual(inspect.call_count, 5)
        self.assertTrue(all(self.key in call.args[1].as_posix() for call in inspect.call_args_list))

    def test_fully_annotated_cine_and_2d_singleton_are_supported(self):
        self.paths = self.bundle(self.key, cmr_shape=(12, 13, 1, 8))
        self.bundle("SYNTHETIC_B/singleton.nii", cmr_shape=(12, 13, 1, 1))
        result = self.run_batch()
        self.assertEqual(result["counts"]["accepted_pairs"], 2)
        self.assertEqual(self.audit(self.acceptances())["summary"]["releasable_pairs"], 2)

    def test_pending_work_trash_and_cmr_frame_names_are_excluded_without_moves(self):
        names = ("SYNTHETIC_A/level.png", "SYNTHETIC_A/level_s02.png",
                 "SYNTHETIC_A/SYNTHETIC_A_level_s02.png", "SYNTHETIC_A_level_s02.png",
                 "SYNTHETIC_A/level.nii.gz.png", "SYNTHETIC_A/level_f02.png")
        for folder in ("待筛", "待处理", "垃圾桶"):
            for name in names:
                with self.subTest(folder=folder, name=name):
                    path = self.bucket(folder, "cmr", name)
                    before = self.snapshot()
                    result = self.run_batch()
                    self.assertEqual(result["counts"], {"existing_review_or_exclusion": 1})
                    self.assertEqual(self.snapshot(), before)
                    path.unlink()
        for folder in ("待筛", "待处理", "垃圾桶"):
            path = self.bucket(folder, "lge", "SYNTHETIC_A/level.png")
            self.assertEqual(self.run_batch()["attempted"], 0)
            path.unlink()
        self.grade.assert_not_called()
        self.assertEqual(self.journal("attempts"), [])

    def test_review_names_preserve_decimal_stems_nested_paths_and_nii_extension(self):
        self.bundle("SYNTHETIC_B/nested/78.6223.nii")
        self.bucket("垃圾桶", "cmr", "SYNTHETIC_B/nested/78.6223_s01.png")
        result = self.run_batch()
        self.assertEqual(result["counts"]["existing_review_or_exclusion"], 1)
        self.assertEqual(result["counts"]["accepted_pairs"], 1)

    def test_cmr_holdout_excludes_all_patient_volumes_even_if_caller_omits_it(self):
        self.freeze_cmr()
        self.bundle("SYNTHETIC_A/third-level.nii.gz")
        result = self.run_batch()
        self.assertEqual(result["counts"], {"structural_or_frozen_exclusion": 2})
        self.grade.assert_not_called()

    def test_audit_explicit_patient_exclusion_is_preserved(self):
        result = self.run_batch(self.audit(excluded_patients={"SYNTHETIC_A"}))
        self.assertEqual(result["counts"], {"structural_or_frozen_exclusion": 1})
        self.grade.assert_not_called()

    def test_cmr_holdout_uses_existing_unicode_digest_contract(self):
        self.bundle("合成患者/level.nii.gz")
        self.freeze_cmr("合成患者/other.nii.gz")
        result = self.run_batch()
        self.assertEqual(result["counts"]["structural_or_frozen_exclusion"], 1)
        self.assertEqual(result["counts"]["accepted_pairs"], 1)

    def test_lge_holdout_mapping_and_integrity_fail_closed(self):
        from experiments.runtime_evaluation import _case_id
        import hashlib
        salt = "synthetic-salt"
        value = {"dataset": str(self.dataset), "identity_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
                 "holdout_patients": [_case_id(salt, str(self.dataset) + ":patient:SYNTHETIC_A")]}
        value["sha256"] = canonical_hash(value)
        path = self.root / "runtime/lge_execution/holdout.json"
        self.write(path, value)
        with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": salt}):
            self.assertEqual(self.run_batch()["attempted"], 0)
        with patch.dict(os.environ, {"CMR_LGE_CASE_SALT": "wrong"}):
            with self.assertRaisesRegex(ValueError, "identity_unavailable"):
                self.run_batch()
        value["holdout_patients"] = []
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "frozen_holdout_changed"):
            self.run_batch()
        self.grade.assert_not_called()

    def test_corrupt_cmr_holdout_is_not_silently_ignored(self):
        self.write(self.root / "runtime/check_iterations/holdout_v1.json", {"holdout": [], "holdout_sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "frozen_holdout_changed"):
            self.run_batch()
        self.grade.assert_not_called()

    def test_invalid_audit_schema_cycles_roles_and_forged_blockers_fail_closed(self):
        original = self.audit()
        variants = []
        for field, value in (("schema", "wrong"), ("dataset", str(self.root)), ("issues", [{}])):
            item = copy.deepcopy(original)
            item[field] = value
            variants.append(item)
        for role in ("cmr_image", "lge_myo"):
            item = copy.deepcopy(original)
            item["rows"][0]["paths"][role] = str(self.paths["lge_mi"])
            item["audit_sha256"] = paired._digest({k: v for k, v in item.items() if k != "audit_sha256"})
            variants.append(item)
        item = copy.deepcopy(original)
        item["rows"][0]["blockers"] = dict.fromkeys(paired.TARGETS, [])
        item["audit_sha256"] = paired._digest({k: v for k, v in item.items() if k != "audit_sha256"})
        variants.append(item)
        cycle = {}
        cycle["rows"] = cycle
        variants.extend([cycle, {}, None, {"rows": float("nan")}, []])
        for item in variants:
            with self.subTest(audit_type=type(item).__name__):
                with self.assertRaises(ValueError):
                    validator.validate_batch(self.root, self.dataset, item)
        self.grade.assert_not_called()
        self.assertEqual(self.acceptances(), [])

    def test_stale_audit_hash_requires_fresh_audit(self):
        audit = self.audit()
        self.mutate_image()
        with self.assertRaisesRegex(ValueError, "stale_or_changed"):
            self.run_batch(audit)
        self.grade.assert_not_called()
        self.assertEqual(self.acceptances(), [])
        self.assertEqual(self.run_batch()["counts"]["accepted_pairs"], 1)

    def test_structural_mask_and_geometry_failures_are_not_evidence_debt(self):
        nib.save(nib.Nifti1Image(np.full((12, 13, 1, 3), 2, dtype=np.float32), np.eye(4)), self.paths["cmr_myo"])
        self.assertEqual(self.run_batch()["counts"], {"structural_or_frozen_exclusion": 1})
        self.grade.assert_not_called()

    def test_missing_dependency_file_is_ineligible(self):
        self.paths["lge_myo"].unlink()
        self.assertEqual(self.run_batch()["counts"], {"structural_or_frozen_exclusion": 1})
        self.grade.assert_not_called()

    def test_forged_geometry_blockers_cannot_authorize_current_mismatched_files(self):
        image = nib.load(self.paths["lge_mi"])
        affine = image.affine.copy()
        affine[0, 3] += 2
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), self.paths["lge_mi"])
        audit = self.audit()
        audit["rows"][0]["blockers"]["lge_mi"].remove("affine_mismatch")
        audit["audit_sha256"] = paired._digest({k: v for k, v in audit.items() if k != "audit_sha256"})
        with self.assertRaisesRegex(ValueError, "invalid_bundle_geometry"):
            self.run_batch(audit)
        self.grade.assert_not_called()
        self.assertEqual(self.acceptances(), [])

    def test_durable_started_does_not_mask_original_error_if_error_journal_fails(self):
        append = validator._append

        def failing_append(path, records):
            if records[0]["status"] == "ERROR":
                raise OSError("synthetic journal failure")
            append(path, records)

        with patch.object(validator, "_append", side_effect=failing_append), \
             patch.object(validator, "score_bundle", side_effect=RuntimeError("synthetic scorer bug")):
            with self.assertRaisesRegex(RuntimeError, "synthetic scorer bug"):
                self.run_batch()
        self.assertEqual(self.journal("attempts")[-1]["status"], "STARTED")
        self.assertEqual(self.run_batch()["attempted"], 0)

    def test_source_changed_during_scoring_does_not_get_receipt(self):
        def changed(row, profiles):
            self.mutate_image()
            return self.scores
        with patch.object(validator, "score_bundle", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "source_changed"):
                self.run_batch()
        self.assertEqual(self.acceptances(), [])
        self.assertEqual(self.journal("attempts")[-1]["status"], "ERROR")

    def test_new_review_entry_or_holdout_during_scoring_prevents_acceptance(self):
        def changed(row, profiles):
            self.bucket("待处理", "cmr", "SYNTHETIC_A/level.png")
            return self.scores
        with patch.object(validator, "score_bundle", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "exclusion_changed"):
                self.run_batch()
        self.assertEqual(self.acceptances(), [])

    def test_calibration_drift_during_scoring_prevents_acceptance(self):
        def changed(row, profiles):
            self.version = "calibration-B"
            return self.scores
        with patch.object(validator, "score_bundle", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "calibration_changed"):
                self.run_batch()
        self.assertEqual(self.acceptances(), [])

    def test_scorer_errors_propagate_and_are_durably_suppressed(self):
        for index, error in enumerate((OSError("load"), ValueError("features"), KeyError("score"), RuntimeError("bug"))):
            self.version = "failure-" + str(index)
            with self.subTest(error=type(error).__name__), patch.object(validator, "score_bundle", side_effect=error) as score:
                with self.assertRaises(type(error)):
                    self.run_batch()
                self.assertEqual(self.journal("attempts")[-1]["error_type"], type(error).__name__)
                self.assertEqual(self.run_batch()["counts"], {"unchanged_already_validated": 1})
                self.assertEqual(score.call_count, 1)
        self.assertEqual(self.acceptances(), [])

    def test_crash_after_started_is_suppressed_on_restart(self):
        with patch.object(validator, "score_bundle", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.assertEqual(self.journal("attempts")[-1]["status"], "STARTED")
        self.assertEqual(self.run_batch()["attempted"], 0)
        self.assertEqual(self.acceptances(), [])

    def test_rejected_bytes_and_calibration_cycles_do_not_repeat_faulty_inputs(self):
        self.scores = dict.fromkeys(paired.TARGETS, MIN_QUALITY_SCORE - 1)
        original = self.paths["cmr_image"].read_bytes()
        self.assertEqual(self.run_batch()["counts"]["needs_review_pairs"], 1)
        self.assertEqual(self.run_batch()["attempted"], 0)
        self.mutate_image()
        self.assertEqual(self.run_batch()["attempted"], 1)
        self.paths["cmr_image"].write_bytes(original)
        self.assertEqual(self.run_batch()["attempted"], 0)
        self.version = "calibration-B"
        self.assertEqual(self.run_batch()["attempted"], 1)
        self.version = "calibration-A"
        self.assertEqual(self.run_batch()["attempted"], 0)
        self.assertEqual(self.acceptances(), [])
        self.assertEqual(self.grade.call_count, 9)

    def test_invalid_scores_never_become_receipts(self):
        for index, score in enumerate((True, float("nan"), float("inf"), -1, 101, "100")):
            self.version = "bad-score-" + str(index)
            self.scores["lge_mi"] = score
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "invalid_score"):
                self.run_batch()
        self.assertEqual(self.acceptances(), [])

    def test_mi_requires_accepted_myo_and_binds_both_current_masks(self):
        self.scores["lge_myo"] = MIN_QUALITY_SCORE - 1
        self.scores["lge_mi"] = 100
        result = self.run_batch()
        self.assertEqual(result["counts"], {"needs_review_pairs": 1, "accepted_targets": 1})
        self.assertEqual([r["target"] for r in self.acceptances()], ["cmr_myo"])
        self.version = "calibration-B"
        self.scores["lge_myo"] = MIN_QUALITY_SCORE
        self.run_batch()
        evidence, _ = collect_evidence(self.root, self.dataset)
        self.assertEqual(self.audit(evidence)["summary"]["releasable_pairs"], 1)
        nib.save(nib.Nifti1Image(np.zeros((12, 13, 1), dtype=np.float32), np.eye(4)), self.paths["lge_myo"])
        audited = self.audit(evidence)
        self.assertIn("stale_dependency_hash", audited["rows"][0]["blockers"]["lge_mi"])

    def test_myo_acceptance_is_independent_of_failed_mi_score(self):
        self.scores["lge_mi"] = MIN_QUALITY_SCORE - 1
        self.run_batch()
        self.assertEqual({r["target"] for r in self.acceptances()}, {"cmr_myo", "lge_myo"})
        myo = next(r for r in self.acceptances() if r["target"] == "lge_myo")
        self.assertEqual(myo["hashes"]["mi"], paired._sha_file(self.paths["lge_mi"]))

    def test_new_calibration_rejections_supersede_unchanged_accepted_bytes_and_block_export(self):
        before = self.snapshot()
        for index, failed in enumerate(paired.TARGETS):
            with self.subTest(failed=failed):
                self.version = f"calibration-{index}-A"
                self.scores = dict.fromkeys(paired.TARGETS, MIN_QUALITY_SCORE)
                self.run_batch()
                accepted, _ = collect_evidence(self.root, self.dataset)
                old_audit = self.audit(accepted)
                self.assertEqual(old_audit["summary"]["releasable_pairs"], 1)
                self.version = f"calibration-{index}-B"
                self.scores[failed] = MIN_QUALITY_SCORE - 1
                self.run_batch(old_audit)
                latest, meta = collect_evidence(self.root, self.dataset)
                negative = {failed, "lge_mi"} if failed == "lge_myo" else {failed}
                self.assertEqual({r["target"] for r in latest if r["status"] == "NEEDS_REVIEW"}, negative)
                self.assertEqual(meta["normalized_acceptances"], 3 - len(negative))
                for r in latest:
                    self.assertEqual(r["provenance"]["score"], self.scores[r["target"]])
                    self.assertEqual(r["provenance"]["calibration_id"], self.version + ":" + (
                        "cmr_paired" if r["target"] == "cmr_myo" else r["target"]))
                    self.assertEqual(r["provenance"]["provisional"], r["target"] in negative)
                blocked = self.audit(latest)
                self.assertEqual(blocked["summary"]["releasable_pairs"], 0)
                with self.assertRaisesRegex(ValueError, "no_releasable_pairs"):
                    paired.freeze_release(self.dataset, blocked, self.root / "forbidden-release")
                # Returning to already-attempted calibration A cannot resurrect A.
                self.version = f"calibration-{index}-A"
                self.assertEqual(self.run_batch(blocked)["attempted"], 0)
                self.assertEqual(collect_evidence(self.root, self.dataset)[0], latest)
        self.assertEqual(self.snapshot(), before)

    def test_error_and_crash_leave_explicit_negative_evidence_not_old_acceptance(self):
        for index, error in enumerate((RuntimeError("synthetic scorer failure"), KeyboardInterrupt())):
            self.version = f"before-error-{index}"
            self.run_batch()
            old = self.audit(collect_evidence(self.root, self.dataset)[0])
            self.version = f"after-error-{index}"
            with patch.object(validator, "score_bundle", side_effect=error):
                with self.assertRaises(type(error)):
                    self.run_batch(old)
            latest, meta = collect_evidence(self.root, self.dataset)
            self.assertEqual(meta["normalized_acceptances"], 0)
            self.assertEqual({r["status"] for r in latest}, {"ERROR" if index == 0 else "PENDING"})
            self.assertTrue(all(r["provenance"]["provisional"] for r in latest))
            self.assertTrue(all("score" not in r["provenance"] for r in latest))
            self.assertEqual(self.audit(latest)["summary"]["releasable_pairs"], 0)

    def test_key_filter_only_revalidates_requested_keys_and_preserves_exclusions(self):
        selected = "SYNTHETIC_B/failure.nii.gz"
        blocked = "SYNTHETIC_C/failure.nii.gz"
        frozen = "SYNTHETIC_D/failure.nii.gz"
        for key in (selected, blocked, frozen):
            self.bundle(key)
        self.bucket("待筛", "cmr", "SYNTHETIC_C/failure_s02.png")
        self.freeze_cmr(frozen)
        audit = self.audit()
        with patch.object(paired, "_inspect", wraps=paired._inspect) as inspect:
            result = self.run_batch(audit, key_filter=[selected, blocked, frozen], limit=100)
        self.assertEqual(result["attempted"], 1)
        self.assertEqual(inspect.call_count, 5)
        self.assertEqual({r["key"] for r in self.journal()}, {selected})
        self.assertEqual(result["counts"]["existing_review_or_exclusion"], 1)
        self.assertEqual(result["counts"]["structural_or_frozen_exclusion"], 1)
        self.assertEqual(self.run_batch(audit, key_filter=[])["attempted"], 0)
        for bad in ([self.key, self.key], ["../escape.nii"], {self.key}, ["UNKNOWN/file.nii.gz"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_batch(audit, key_filter=bad)

    def test_key_filter_does_not_bypass_current_source_hash(self):
        audit = self.audit()
        self.mutate_image()
        with self.assertRaisesRegex(ValueError, "stale_or_changed"):
            self.run_batch(audit, key_filter=[self.key], limit=100)
        self.grade.assert_not_called()
        self.assertEqual(self.acceptances(), [])

    def test_limit_counts_attempts_and_next_batch_progresses(self):
        for index in range(3):
            self.bundle(f"SYNTHETIC_B/level{index}.nii.gz")
        self.assertEqual(self.run_batch(limit=2)["attempted"], 2)
        self.assertEqual(self.run_batch(limit=2)["attempted"], 2)
        self.assertEqual(len(self.acceptances()), 12)
        for limit in (0, -1, 101, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_batch(limit=limit)

    def test_damaged_journal_is_not_silently_ignored(self):
        for name in ("attempts", "receipts"):
            path = self.root / "runtime/dataset_acceptance" / (name + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"unfinished":', encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_batch()
            path.unlink()
        self.grade.assert_not_called()

    def test_runtime_must_not_write_inside_dataset(self):
        with self.assertRaisesRegex(ValueError, "runtime_inside_dataset"):
            validator.validate_batch(self.dataset, self.dataset, self.audit())
        self.assertFalse((self.dataset / "runtime").exists())

    def test_process_lock_serializes_validation(self):
        script = '''
import sys
from pathlib import Path
from app.segmentation_registry import _locked, ConcurrentUpdate
try:
    with _locked(Path(sys.argv[1])):
        raise AssertionError("concurrent lock acquired")
except ConcurrentUpdate:
    print("busy")
'''
        def competing(row, profiles):
            result = subprocess.run([sys.executable, "-B", "-c", script,
                                     str(self.root / "runtime/dataset_acceptance")],
                                    capture_output=True, text=True, timeout=20, check=True)
            self.assertEqual(result.stdout.strip(), "busy")
            return self.scores
        with patch.object(validator, "score_bundle", side_effect=competing):
            self.assertEqual(self.run_batch()["counts"]["accepted_pairs"], 1)

    def test_review_directory_cycle_is_rejected(self):
        base = self.dataset / "labels/check/待筛/cmr"
        base.mkdir(parents=True)
        link = base / "cycle"
        if os.name == "nt":
            # Junctions need no Windows symlink privilege; always remove just the link.
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(base)],
                                    capture_output=True)
            if result.returncode:
                self.skipTest("Windows junction creation unavailable: " + repr(result.stderr))
        else:
            link.symlink_to(base, target_is_directory=True)
        try:
            with self.assertRaisesRegex(ValueError, "unsafe_path"):
                self.run_batch()
            self.grade.assert_not_called()
        finally:
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()

    def test_review_reparse_point_fails_before_recursion(self):
        # Exercise the Windows safety branch even when junction creation is denied.
        link = self.dataset / "labels/check/待筛/cmr/cycle"
        link.mkdir(parents=True)
        original = Path.lstat

        def reparse(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == link:
                return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        with patch.object(Path, "lstat", reparse):
            with self.assertRaisesRegex(ValueError, "unsafe_path"):
                self.run_batch()
        self.grade.assert_not_called()


if __name__ == "__main__":
    unittest.main()
