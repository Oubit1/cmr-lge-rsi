from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from app import segmentation_evaluation as evaluation


class SegmentationEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    @staticmethod
    def binding(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write_mask(self, path, data, *, affine=None, image_type=nib.Nifti1Image):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape((2, 2, 1))
        image = image_type(data.astype(np.float64), np.eye(4) if affine is None else affine)
        nib.save(image, str(path))
        return self.binding(path)

    def case(self, case_id="case-1", patient_id="patient-1", *, reference=None,
             parent=None, candidate=None, extension=".nii.gz", image_type=nib.Nifti1Image):
        reference = [1, 1, 0, 0] if reference is None else reference
        parent = reference if parent is None else parent
        candidate = reference if candidate is None else candidate
        row = {"patient_id": patient_id, "case_id": case_id}
        for role, data in (("reference", reference), ("parent", parent), ("candidate", candidate)):
            row[role] = self.write_mask(self.root / (case_id + "-" + role + extension), data,
                                        image_type=image_type)
        return row

    def compare(self, cases, target="lge_mi"):
        return evaluation.compare_predictions(cases, target=target)

    def assert_metric(self, report, name, parent, candidate, count):
        metric = report["metrics"][name]
        self.assertEqual(set(metric), {"parent", "candidate", "parent_count", "candidate_count"})
        for role, value in (("parent", parent), ("candidate", candidate)):
            if value is None:
                self.assertIsNone(metric[role])
            else:
                self.assertAlmostEqual(metric[role], value)
            self.assertEqual(metric[role + "_count"], count)

    def test_actual_dice_and_provenance_without_source_or_manifest_writes(self):
        row = self.case(parent=[1, 0, 1, 0], candidate=[255, 255, 255, 0])
        # Irrelevant owner-supplied metadata cannot replace overlap measurements.
        row.update(qc_score=100, dice=1, metrics={"dice": 1})
        row["candidate"]["qc_score"] = 100
        cases = [row]
        original = copy.deepcopy(cases)
        files = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.iterdir()}
        report = self.compare(cases)
        self.assert_metric(report, "dice", 0.5, 0.8, 1)
        self.assert_metric(report, "positive_reference_dice", 0.5, 0.8, 1)
        self.assert_metric(report, "empty_reference_fp_rate", None, None, 0)
        self.assert_metric(report, "coverage", 1, 1, 1)
        self.assertEqual(set(report["metrics"]), {"dice", "positive_reference_dice",
                                                   "empty_reference_fp_rate", "coverage"})
        self.assertEqual(report["source"], "predictions")
        self.assertIs(report["self_reported"], False)
        self.assertEqual(report["aggregation"], "patient_macro")
        self.assertEqual(report["target"], "lge_mi")
        self.assertEqual(cases, original)
        self.assertEqual(set(self.root.iterdir()), set(files))
        for path, (content, mtime) in files.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        # Reports are strict JSON, including absent-stratum values.
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        manifest = report["manifest"]
        for role in ("reference", "parent", "candidate"):
            self.assertEqual(manifest["cases"][0][role], self.binding(Path(row[role]["path"])))
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        self.assertEqual(report["manifest_sha256"], hashlib.sha256(canonical).hexdigest())
        for role, key in (("reference", "reference_sha256"), ("parent", "parent_predictions_sha256"),
                          ("candidate", "candidate_predictions_sha256")):
            source = {"target": "lge_mi", "cases": [{"patient_id": row["patient_id"],
                       "case_id": row["case_id"], "sha256": row[role]["sha256"]}]}
            payload = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            self.assertEqual(report[key], hashlib.sha256(payload).hexdigest())

    def test_patient_macro_is_not_case_or_voxel_weighted(self):
        cases = [self.case("a1", "A", parent=[0, 0, 0, 0]),
                 self.case("a2", "A", parent=[1, 0, 1, 0]),
                 self.case("a3", "A", parent=[1, 1, 0, 0]),
                 self.case("b1", "B", reference=np.ones((3, 4, 5)),
                           parent=np.ones((3, 4, 5)), candidate=np.zeros((3, 4, 5)))]
        report = self.compare(cases)
        self.assert_metric(report, "dice", 0.75, 0.5, 2)
        self.assert_metric(report, "positive_reference_dice", 0.75, 0.5, 2)
        self.assert_metric(report, "coverage", 1, 1, 4)
        self.assert_metric(report["per_patient"][0], "dice", 0.5, 1, 3)
        self.assert_metric(report["per_patient"][1], "dice", 1, 0, 1)
        self.assertEqual(report["cohort"], {"expected_cases": 4, "evaluated_cases": 4,
                         "patients": 2, "positive_reference_cases": 4, "empty_reference_cases": 0})

    def test_empty_mi_fp_rate_is_patient_macro_any_foreground_per_case(self):
        empty, single, positive = [0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 0, 0]
        cases = [self.case("a-empty-1", "A", reference=empty, parent=empty, candidate=single),
                 self.case("a-empty-2", "A", reference=empty, parent=empty, candidate=empty),
                 self.case("a-empty-3", "A", reference=empty, parent=empty, candidate=empty),
                 self.case("a-positive", "A", reference=positive, parent=empty, candidate=positive),
                 self.case("b-empty", "B", reference=empty, parent=single, candidate=single),
                 self.case("c-positive", "C", reference=positive, parent=positive, candidate=positive)]
        report = self.compare(cases)
        self.assert_metric(report, "empty_reference_fp_rate", 0.5, 2 / 3, 2)
        self.assert_metric(report, "positive_reference_dice", 0.5, 1, 2)
        self.assert_metric(report, "dice", 7 / 12, 7 / 12, 3)
        self.assert_metric(report, "coverage", 1, 1, 6)
        self.assert_metric(report["per_patient"][0], "empty_reference_fp_rate", 0, 1 / 3, 3)
        self.assert_metric(report["per_patient"][1], "positive_reference_dice", None, None, 0)
        self.assert_metric(report["per_patient"][2], "empty_reference_fp_rate", None, None, 0)
        self.assertEqual(report["cohort"]["positive_reference_cases"], 2)
        self.assertEqual(report["cohort"]["empty_reference_cases"], 4)

    def test_all_empty_and_all_positive_strata(self):
        empty = [0, 0, 0, 0]
        report = self.compare([self.case(reference=empty)])
        self.assert_metric(report, "dice", 1, 1, 1)
        self.assert_metric(report, "positive_reference_dice", None, None, 0)
        self.assert_metric(report, "empty_reference_fp_rate", 0, 0, 1)
        report = self.compare([self.case(parent=empty, candidate=empty)])
        self.assert_metric(report, "dice", 0, 0, 1)
        self.assert_metric(report, "positive_reference_dice", 0, 0, 1)
        self.assert_metric(report, "empty_reference_fp_rate", None, None, 0)

    def test_supported_targets_encodings_dimensions_and_nifti_versions(self):
        for target in ("cmr_myo", "lge_myo", "lge_mi"):
            for image_type in (nib.Nifti1Image, nib.Nifti2Image):
                for extension in (".nii", ".nii.gz"):
                    with self.subTest(target=target, image_type=image_type, extension=extension):
                        row = self.case(reference=np.ones((2, 2, 1, 2)), parent=np.full((2, 2, 1, 2), 255),
                                        candidate=np.ones((2, 2, 1, 2)), extension=extension, image_type=image_type)
                        self.assert_metric(self.compare([row], target), "dice", 1, 1, 1)

    def test_incomplete_or_invalid_manifest_fails(self):
        row = self.case()
        for invalid in ([], None, {}, "cases", (row,)):
            with self.subTest(invalid=invalid), self.assertRaises(evaluation.EvaluationError):
                self.compare(invalid)
        for field in ("patient_id", "case_id", "reference", "parent", "candidate"):
            for value in (None, "", " ", 0, [], {}):
                broken = copy.deepcopy(row)
                broken[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(evaluation.EvaluationError):
                    self.compare([broken])
            broken = copy.deepcopy(row)
            broken.pop(field)
            with self.assertRaises(evaluation.EvaluationError):
                self.compare([broken])
        for target in (None, "", "mi", "qc_score", []):
            with self.subTest(target=target), self.assertRaises(evaluation.EvaluationError):
                self.compare([row], target)

    def test_missing_prediction_or_reference_fails_whole_cohort(self):
        for role in ("reference", "parent", "candidate"):
            with self.subTest(role=role):
                good, bad = self.case("good"), self.case("bad")
                Path(bad[role]["path"]).unlink()
                with self.assertRaisesRegex(evaluation.EvaluationError, role + ": missing"):
                    self.compare([good, bad])

    def test_duplicate_case_id_fails_even_for_different_patients(self):
        row = self.case()
        for patient_id in (row["patient_id"], "another-patient"):
            duplicate = copy.deepcopy(row)
            duplicate["patient_id"] = patient_id
            with self.subTest(patient_id=patient_id), self.assertRaisesRegex(
                    evaluation.EvaluationError, "duplicate case_id"):
                self.compare([row, duplicate])

    def test_invalid_bindings_and_preexisting_hash_drift(self):
        for role in ("reference", "parent", "candidate"):
            row = self.case()
            for field, value in (("sha256", None), ("sha256", ""), ("sha256", "a" * 63),
                                 ("sha256", "g" * 64), ("sha256", "0" * 64),
                                 ("path", None), ("path", ""), ("path", "x.txt")):
                broken = copy.deepcopy(row)
                broken[role][field] = value
                with self.subTest(role=role, field=field, value=value), self.assertRaises(evaluation.EvaluationError):
                    self.compare([broken])
            path = Path(row[role]["path"])
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(evaluation.EvaluationError, role + ": SHA-256 mismatch"):
                self.compare([row])

    def test_late_hash_drift_or_deletion_of_early_case_fails_entire_cohort(self):
        original_reader = evaluation._read_mask
        for role in ("reference", "parent", "candidate"):
            for delete in (False, True):
                first, last = self.case("a-first"), self.case("z-last")
                source = Path(first[role]["path"])

                def read_then_change(binding, label):
                    mask = original_reader(binding, label)
                    if label == "z-last/candidate":
                        if delete:
                            source.unlink()
                        else:
                            source.write_bytes(source.read_bytes() + b"changed-after-measurement")
                    return mask

                with self.subTest(role=role, delete=delete), patch.object(
                        evaluation, "_read_mask", side_effect=read_then_change):
                    with self.assertRaisesRegex(evaluation.EvaluationError, "a-first/" + role):
                        self.compare([first, last])

    def test_nan_infinity_nonbinary_and_mixed_encodings_fail_for_every_role(self):
        values = ([float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0],
                  [-float("inf"), 0, 0, 0], [0.5, 0, 0, 0], [-1, 0, 0, 0],
                  [2, 0, 0, 0], [254, 0, 0, 0], [1, 255, 0, 0], [1.00000001, 0, 0, 0])
        good = self.case("good")
        for role in ("reference", "parent", "candidate"):
            for data in values:
                with self.subTest(role=role, data=data):
                    bad = self.case("bad")
                    bad[role] = self.write_mask(Path(bad[role]["path"]), data)
                    with self.assertRaisesRegex(evaluation.EvaluationError, role + ": (nonfinite|nonbinary)"):
                        self.compare([good, bad])

    def test_trailing_singletons_do_not_require_resampling(self):
        reference = np.array([[1, 0], [0, 1]])
        row = self.case(reference=reference, parent=reference[..., None],
                        candidate=reference[..., None, None])
        self.assert_metric(self.compare([row]), "dice", 1, 1, 1)

    def test_shape_and_affine_mismatch_fail_without_resampling(self):
        for role in ("reference", "parent", "candidate"):
            for mismatch in ("shape", "extra_dimension", "translation", "tiny_translation", "spacing", "flip"):
                with self.subTest(role=role, mismatch=mismatch):
                    row = self.case()
                    affine, data = np.eye(4), np.ones((2, 2, 1))
                    if mismatch == "shape":
                        data = np.ones((2, 3, 1))
                    elif mismatch == "extra_dimension":
                        data = np.ones((2, 2, 1, 2))
                    elif mismatch == "translation":
                        affine[0, 3] = 10
                    elif mismatch == "tiny_translation":
                        affine[0, 3] = 1e-7
                    elif mismatch == "spacing":
                        affine[0, 0] = 2
                    else:
                        affine[0, 0] = -1
                    row[role] = self.write_mask(Path(row[role]["path"]), data, affine=affine)
                    expected = "shape mismatch" if mismatch in ("shape", "extra_dimension") else "affine mismatch"
                    with self.assertRaisesRegex(evaluation.EvaluationError, expected):
                        self.compare([row])

    def test_nonfinite_affine_fails_even_if_all_images_match(self):
        row = self.case(extension=".nii")
        for role in ("reference", "parent", "candidate"):
            path = Path(row[role]["path"])
            # Edit the on-disk sform directly; image constructors reject NaN affines.
            with path.open("r+b") as stream:
                header = nib.Nifti1Header.from_fileobj(stream)
                header["srow_x"][0] = np.nan
                stream.seek(0)
                header.write_to(stream)
            row[role] = self.binding(path)
        with self.assertRaisesRegex(evaluation.EvaluationError, "nonfinite or invalid affine"):
            self.compare([row])

    def test_corrupt_truncated_and_non_nifti_sources_fail_with_correct_hashes(self):
        for role in ("reference", "parent", "candidate"):
            for kind in ("garbage", "truncated", "bad_gzip"):
                with self.subTest(role=role, kind=kind):
                    row = self.case(extension=".nii" if kind != "bad_gzip" else ".nii.gz")
                    path = Path(row[role]["path"])
                    payload = path.read_bytes()[:-8] if kind == "truncated" else b"not a NIfTI image"
                    path.write_bytes(payload)
                    row[role] = self.binding(path)
                    with self.assertRaisesRegex(evaluation.EvaluationError, role + ": invalid NIfTI"):
                        self.compare([row])

    def test_hashes_are_order_independent_and_bind_grouping_and_predictions(self):
        cases = [self.case("b", "B"), self.case("a", "A")]
        original = self.compare(cases)
        self.assertEqual(self.compare(list(reversed(cases))), original)
        cases[0]["candidate"] = self.write_mask(Path(cases[0]["candidate"]["path"]), [0, 0, 0, 0])
        changed = self.compare(cases)
        self.assertNotEqual(original["manifest_sha256"], changed["manifest_sha256"])
        self.assertNotEqual(original["candidate_predictions_sha256"], changed["candidate_predictions_sha256"])
        self.assertEqual(original["parent_predictions_sha256"], changed["parent_predictions_sha256"])
        self.assertEqual(original["reference_sha256"], changed["reference_sha256"])
        cases[0]["patient_id"] = "A"
        regrouped = self.compare(cases)
        for key in ("manifest_sha256", "reference_sha256", "parent_predictions_sha256", "candidate_predictions_sha256"):
            self.assertNotEqual(changed[key], regrouped[key])
        self.assertEqual(regrouped["metrics"]["dice"]["parent_count"], 1)

    def test_python39_syntax(self):
        for path in (Path(evaluation.__file__), Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
