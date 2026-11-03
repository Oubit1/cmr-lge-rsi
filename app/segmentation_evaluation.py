"""Read-only, prediction-derived measurements against caller-frozen references.

The trusted controller must supply the complete frozen cohort as ``cases``;
this API cannot discover cases omitted by the caller or authenticate references.
Any invalid case fails the entire call with EvaluationError (no partial report).
Only single-file NIfTI-1/2 (.nii or .nii.gz) masks are accepted. Each mask must
use either {0, 1} or {0, 255}, without mixing encodings or thresholding. Geometry
must match after removing trailing singleton axes only; there is no resampling,
axis permutation or clinical adjudication.

``metrics`` contains ``dice``, ``positive_reference_dice``,
``empty_reference_fp_rate``, and ``coverage``. Dice is 2|P intersect R|/(|P|+|R|),
with empty/empty = 1. Empty-reference FP is a case-level indicator of ANY predicted
foreground, not a voxel fraction. Scores average cases within each patient,
then average contributing patients equally, separately for each reference stratum.
Their parent_count/candidate_count are patient denominators; ``per_patient``
exposes the corresponding case denominators. Absent strata have None and zero
counts. Coverage is evaluated cases / manifest cases, with CASE denominators;
it must be 1 for both predictions. No QC scores, policy thresholds, or activation
decisions are consumed or produced.

All evaluated bytes are hash-verified in-memory snapshots, and every source is
rehashed before returning. Sources and the input manifest are never written.
The report contains private patient/case IDs and paths. Hashes bind content and
grouping, not evaluator identity, cohort approval, or train/test disjointness.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import Path

import nibabel as nib
import numpy as np


_TARGETS = frozenset({"cmr_myo", "lge_myo", "lge_mi"})
_ROLES = ("reference", "parent", "candidate")
_SCORES = ("dice", "positive_reference_dice", "empty_reference_fp_rate")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


class EvaluationError(ValueError):
    """Incomplete cohort, invalid mask/geometry, or changed source bytes."""


def _require(condition, message):
    if not condition:
        raise EvaluationError(message)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip())
             and value == value.strip() and "\x00" not in value,
             label + ": expected nonempty text")
    return value


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(cases, target):
    _require(isinstance(target, str) and target in _TARGETS, "unsupported target")
    _require(isinstance(cases, list) and bool(cases), "cohort must be a nonempty list")
    rows, seen = [], set()
    for case in cases:
        _require(isinstance(case, dict), "case must be an object")
        patient_id = _text(case.get("patient_id"), "patient_id")
        case_id = _text(case.get("case_id"), "case_id")
        _require(case_id not in seen, "duplicate case_id: " + case_id)
        seen.add(case_id)
        row = {"patient_id": patient_id, "case_id": case_id}
        for role in _ROLES:
            label = case_id + "/" + role
            binding = case.get(role)
            _require(isinstance(binding, dict), label + ": missing file binding")
            path, expected = binding.get("path"), binding.get("sha256")
            _require(isinstance(path, (str, Path)), label + ": invalid path")
            _text(str(path), label + " path")
            _require(str(path).endswith((".nii", ".nii.gz")), label + ": expected NIfTI file")
            _require(isinstance(expected, str) and _SHA.fullmatch(expected) is not None,
                     label + ": invalid SHA-256")
            # Keep the supplied path (including links) so the final check also
            # detects a link being retargeted. Do not resolve away that binding.
            row[role] = {"path": os.path.abspath(path), "sha256": expected.lower()}
        rows.append(row)
    return {"schema_version": 1, "target": target,
            "cases": sorted(rows, key=lambda row: row["case_id"])}


def _read_bound(binding, label, *, snapshot):
    digest, chunks = hashlib.sha256(), []
    try:
        with Path(binding["path"]).open("rb") as stream:
            _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                     label + ": expected regular file")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                if snapshot:
                    chunks.append(block)
    except OSError as exc:
        raise EvaluationError(label + ": missing or unreadable file") from exc
    _require(digest.hexdigest() == binding["sha256"], label + ": SHA-256 mismatch")
    return b"".join(chunks) if snapshot else None


def _read_mask(binding, label):
    payload = _read_bound(binding, label, snapshot=True)
    try:
        if binding["path"].endswith(".gz"):
            payload = gzip.decompress(payload)
        image_type = (nib.Nifti2Image if nib.Nifti2Header.may_contain_header(payload)
                      else nib.Nifti1Image)
        header = image_type.header_class.from_fileobj(io.BytesIO(payload))
        _require(header["magic"].item() == header.single_magic,
                 label + ": expected single-file NIfTI")
        image = image_type.from_bytes(payload)
        # Materialize from the verified snapshot, never a lazy disk-backed proxy.
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine)
    except Exception as exc:
        raise EvaluationError(label + ": invalid NIfTI") from exc
    _require(data.size > 0, label + ": empty image array")
    _require(np.issubdtype(data.dtype, np.number) and not np.iscomplexobj(data),
             label + ": mask must contain real numeric values")
    _require(np.isfinite(data).all(), label + ": nonfinite mask")
    _require(affine.shape == (4, 4) and np.isfinite(affine).all(),
             label + ": nonfinite or invalid affine")
    _require(np.all((data == 0) | (data == 1)) or np.all((data == 0) | (data == 255)),
             label + ": nonbinary mask (expected 0/1 or 0/255)")
    while data.ndim > 2 and data.shape[-1] == 1:
        data = data[..., 0]
    return data != 0, affine


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _paired(parent, candidate, count):
    return {"parent": parent, "candidate": candidate,
            "parent_count": count, "candidate_count": count}


def _source_digest(manifest, role):
    return _digest({"target": manifest["target"], "cases": [
        {"patient_id": row["patient_id"], "case_id": row["case_id"],
         "sha256": row[role]["sha256"]} for row in manifest["cases"]]})


def compare_predictions(cases: list[dict], *, target: str) -> dict:
    """Evaluate every paired prediction or raise EvaluationError for the cohort.

    Each case supplies patient_id, globally unique case_id, and reference/parent/
    candidate objects containing path and sha256. Extra metadata (including QC
    scores) is ignored. ``manifest_sha256`` hashes the returned canonical manifest;
    the three source hashes bind target, case IDs, patient IDs, and content hashes
    in case_id order. JSON hashes use sorted keys, compact separators, ASCII escapes.
    """
    manifest = _manifest(cases, target)
    patients = {}
    positive_cases = 0
    for row in manifest["cases"]:
        patient = patients.setdefault(row["patient_id"], {
            name: {"parent": [], "candidate": []} for name in _SCORES})
        reference, reference_affine = _read_mask(row["reference"], row["case_id"] + "/reference")
        reference_count = int(np.count_nonzero(reference))
        positive_cases += int(reference_count > 0)
        for role in ("parent", "candidate"):
            label = row["case_id"] + "/" + role
            prediction, affine = _read_mask(row[role], label)
            _require(prediction.shape == reference.shape, label + ": shape mismatch")
            _require(np.array_equal(affine, reference_affine), label + ": affine mismatch")
            predicted_count = int(np.count_nonzero(prediction))
            denominator = reference_count + predicted_count
            dice = (2 * int(np.count_nonzero(reference & prediction)) / denominator
                    if denominator else 1.0)
            patient["dice"][role].append(dice)
            if reference_count:
                patient["positive_reference_dice"][role].append(dice)
            else:
                patient["empty_reference_fp_rate"][role].append(float(predicted_count > 0))

    per_patient = []
    for patient_id, scores in sorted(patients.items()):
        metrics = {name: _paired(_mean(values["parent"]), _mean(values["candidate"]),
                                 len(values["parent"])) for name, values in scores.items()}
        per_patient.append({"patient_id": patient_id, "metrics": metrics})
    metrics = {}
    for name in _SCORES:
        contributing = [row["metrics"][name] for row in per_patient
                        if row["metrics"][name]["parent_count"]]
        metrics[name] = _paired(_mean([row["parent"] for row in contributing]),
                                _mean([row["candidate"] for row in contributing]), len(contributing))
    count = len(manifest["cases"])
    metrics["coverage"] = _paired(1.0, 1.0, count)
    result = {
        "target": target, "source": "predictions", "self_reported": False,
        "aggregation": "patient_macro", "metrics": metrics,
        "cohort": {"expected_cases": count, "evaluated_cases": count,
                   "patients": len(patients), "positive_reference_cases": positive_cases,
                   "empty_reference_cases": count - positive_cases},
        "per_patient": per_patient,
        "manifest": manifest, "manifest_sha256": _digest(manifest),
        "reference_sha256": _source_digest(manifest, "reference"),
        "parent_predictions_sha256": _source_digest(manifest, "parent"),
        "candidate_predictions_sha256": _source_digest(manifest, "candidate"),
    }
    # Recheck the ENTIRE cohort, including early cases, immediately before return.
    # A late change/deletion invalidates all metrics, never just the changed case.
    for row in manifest["cases"]:
        for role in _ROLES:
            _read_bound(row[role], row["case_id"] + "/" + role, snapshot=False)
    return result
