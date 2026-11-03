"""Freeze explicitly selected doctor references; never choose patients or approve a model.

``freeze_cohort(config_path, plan, destination, cache_path=None)`` is a trusted,
explicit local write capability, NOT a model-callable approval endpoint. Plan::

    {"schema": "segmentation-cohort-plan-v1", "owner_role": "lge_mi_specialist",
     "config_sha256": "...", "selections": [
       {"dataset": "site", "target": "lge_mi", "relative_id": "person/1.nii.gz",
        "case_id": "site:person", "patient_key": "stable-private-patient-key",
        "split": "validation"}]}

The caller supplies patient_key as a stable, cross-target/cross-release identity
(including any known cross-site aliases); its UTF-8 SHA-256 is patient_id. This
matches the job compiler when given the SAME private patient key. No identity
is inferred across sites. A source case cannot acquire multiple keys or splits.
Optional cache_path requires plan.cache_sha256 and an exact join to the existing
valid gold cache and its patient_split. No cache means explicitly supplied splits,
not permission to infer legacy model exposure. Gold designation is inherited
from the configured source, not assigned through a QC score or directory name.

Each nonempty target/split gets an exact segmentation-cycle-cohort-v1 JSON with
absolute final image/reference bindings. Metadata and source identities are in
separate provenance/private manifests. Test is only a caller-assigned split name;
baseline exposure stays UNKNOWN and final-test independence NOT_ESTABLISHED.
Original binary masks (including empty MI) and headers are copied byte-for-byte.
There is no resampling, registration, thresholding, clinical selection or source
write. Complete geometry/hash validation precedes atomic no-replace publication.

Everything is PRIVATE, including headers and hashed identifiers. Use an existing
private destination parent: POSIX modes restrict access; Windows relies on its
inherited ACL. Hashes/read-only modes are tamper detection, not authentication.
Only aggregate counts and manifest bindings are returned; source rows stay local.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np

from app import quality_scoring
from app.paired_dataset import _check_path, _discard, _publish_noreplace, _sha_file

TARGETS = frozenset(("cmr_myo", "lge_myo", "lge_mi"))
SPLITS = frozenset(("train", "validation", "test"))


class CohortError(ValueError):
    """Invalid explicit plan, source pair, or frozen provenance (no private IDs)."""


def _require(ok, reason):
    if not ok:
        raise CohortError(reason)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _text(value):
    _require(type(value) is str and bool(value.strip()) and value == value.strip()
             and "\x00" not in value, "invalid_identity_text")
    return value


def _keys(value, required, optional=()):
    _require(type(value) is dict and set(required) <= set(value)
             and set(value) <= set(required) | set(optional), "invalid_schema_keys")


def _path(value, *, directory=False):
    path = Path(os.path.abspath(value))
    _check_path(path)
    _require(path.is_dir() if directory else stat.S_ISREG(path.stat().st_mode),
             "missing_directory" if directory else "missing_regular_file")
    return path


def _identity(row):
    return row["dataset"], row["target"], row["relative_id"]


def _resolve(config_path, plan, cache_path):
    _keys(plan, ("schema", "owner_role", "config_sha256", "selections"), ("cache_sha256",))
    _require(plan["schema"] == "segmentation-cohort-plan-v1", "invalid_plan_schema")
    _text(plan["owner_role"])
    _require(_sha_file(config_path) == plan["config_sha256"], "config_hash_mismatch")
    _require(type(plan["selections"]) is list and plan["selections"], "empty_selection")
    _require((cache_path is not None) == ("cache_sha256" in plan), "cache_binding_required")
    specs, config = quality_scoring.load_specs(config_path)
    index = {}
    for spec in specs:
        key = (spec.dataset, spec.target, spec.relative_id)
        index.setdefault(key, []).append(spec)
    cache_index = None
    if cache_path is not None:
        _require(_sha_file(cache_path) == plan["cache_sha256"], "cache_hash_mismatch")
        from app.rsi_quality_benchmark import load_cached_gold, patient_split
        cached = load_cached_gold(cache_path)
        train, validation = patient_split(cached)
        cache_index = {}
        for split, records in (("train", train), ("validation", validation)):
            for row in records:
                cache_index.setdefault(_identity(row), []).append((row["case_id"], split))
    selected, seen, cases, patients = [], set(), {}, {}
    for row in plan["selections"]:
        _keys(row, ("dataset", "target", "relative_id", "case_id", "patient_key", "split"))
        for value in row.values():
            _text(value)
        _require(row["target"] in TARGETS and row["split"] in SPLITS, "invalid_target_or_split")
        relative = PurePosixPath(row["relative_id"])
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and "\\" not in row["relative_id"] and ":" not in row["relative_id"]
                 and relative.as_posix() == row["relative_id"], "invalid_relative_identity")
        key = _identity(row)
        _require(key not in seen, "duplicate_selection")
        seen.add(key)
        matches = index.get(key, [])
        _require(len(matches) == 1, "missing_or_duplicate_source_identity")
        spec = matches[0]
        _require(spec.tier == "gold", "source_not_configured_gold")
        _require(spec.case_id == row["case_id"], "source_patient_identity_mismatch")
        if cache_index is not None:
            _require(cache_index.get(key) == [(row["case_id"], row["split"])],
                     "missing_duplicate_or_conflicting_cache_join")
        case_key = (spec.dataset, spec.case_id)
        assignment = (row["patient_key"], row["split"])
        _require(cases.setdefault(case_key, assignment) == assignment, "source_patient_assignment_conflict")
        _require(patients.setdefault(row["patient_key"], row["split"]) == row["split"],
                 "patient_split_overlap")
        selected.append((row, spec))
    return sorted(selected, key=lambda item: _identity(item[0])), config


def _mkdir(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _write(path, payload):
    _mkdir(path.parent)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return hashlib.sha256(payload).hexdigest()


def _copy(source, destination, expected):
    _mkdir(destination.parent)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        _require(stat.S_ISREG(os.fstat(src.fileno()).st_mode), "source_not_regular")
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    destination.chmod(0o600)
    _require(digest.hexdigest() == expected, "source_changed_during_copy")


def _geometry(image_path, reference_path):
    shapes, affines = [], []
    for path, mask in ((image_path, False), (reference_path, True)):
        image = nib.load(str(path))
        _require(isinstance(image, (nib.Nifti1Image, nib.Nifti2Image))
                 and image.header["magic"].item() == image.header.single_magic,
                 "single_file_nifti_required")
        data = np.asanyarray(image.dataobj)
        _require(data.ndim >= 2 and data.size > 0 and np.issubdtype(data.dtype, np.number)
                 and not np.iscomplexobj(data) and np.isfinite(data).all(), "invalid_image_values")
        affine = np.asarray(image.affine)
        _require(affine.shape == (4, 4) and np.isfinite(affine).all(), "invalid_affine")
        if mask:
            _require(np.all((data == 0) | (data == 1)) or np.all((data == 0) | (data == 255)),
                     "reference_not_binary")
        shape = list(data.shape)
        while len(shape) > 2 and shape[-1] == 1:
            shape.pop()
        shapes.append(shape)
        affines.append(affine)
    _require(shapes[0] == shapes[1] and np.array_equal(*affines), "geometry_mismatch")
    return {"shape": shapes[0], "affine": affines[0].tolist()}


def freeze_cohort(config_path: Path, plan: dict, destination: Path, *, cache_path: Path | None = None) -> dict:
    """Freeze exactly the supplied members; failure publishes nothing and cleans staging.

    Return manifest/cohort bindings and aggregate counts only. The per-target,
    per-split manifests intentionally have no extra fields beyond the cycle API.
    The provenance manifest binds those files plus the private map and all assets.
    Caller must retain the same patient_key mapping when compiling training jobs.
    """
    staging = None
    try:
        plan = copy.deepcopy(plan)
        config_path = _path(config_path)
        cache_path = _path(cache_path) if cache_path is not None else None
        destination = Path(os.path.abspath(destination))
        parent = _path(destination.parent, directory=True)
        _require(not os.path.lexists(destination), "destination_exists")
        selected, config = _resolve(config_path, plan, cache_path)
        # Even an unselected source root must never become an export destination.
        for dataset in config["datasets"]:
            for item in dataset.get("targets", {}).values():
                for key in ("image_dir", "label_dir", "myo_dir"):
                    if item.get(key):
                        source_root = Path(item[key]).resolve()
                        _require(not destination.is_relative_to(source_root)
                                 and not source_root.is_relative_to(destination), "destination_overlaps_source")
        inputs = [(config_path, plan["config_sha256"])]
        if cache_path is not None:
            inputs.append((cache_path, plan["cache_sha256"]))
        _require(all(not p.is_relative_to(destination) for p, _ in inputs), "destination_overlaps_input")
        staging = Path(tempfile.mkdtemp(prefix=".segmentation-cohort-", dir=parent))
        staging.chmod(0o700)
        groups, source_rows, mappings, assets, copies = {}, [], {}, {}, []
        seen_images, image_patients, seen_references = set(), {}, set()
        for row, spec in selected:
            patient = hashlib.sha256(row["patient_key"].encode("utf-8")).hexdigest()
            case = "c_" + _hash(list(_identity(row)))
            bindings, original = {}, {}
            for role, value in (("image", spec.image), ("reference", spec.label)):
                source = _path(value)
                quality_scoring._assert_original(source)
                _require(source.name.endswith((".nii", ".nii.gz")), "single_file_nifti_required")
                digest = _sha_file(source)
                info = source.stat()
                file_id = (info.st_dev, info.st_ino) if info.st_ino else str(source)
                if role == "image":
                    _require((spec.target, digest) not in seen_images, "duplicate_image_for_target")
                    seen_images.add((spec.target, digest))
                    _require(image_patients.setdefault(digest, patient) == patient, "image_patient_alias")
                else:
                    _require((spec.target, file_id) not in seen_references, "duplicate_reference_for_target")
                    seen_references.add((spec.target, file_id))
                    _require(not os.path.samefile(source, spec.image), "image_reference_alias")
                suffix = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
                rel = (Path("images" if role == "image" else "references") /
                       spec.target / patient / (case + suffix)).as_posix()
                target = staging / rel
                _copy(source, target, digest)
                copies.append((source, target, digest))
                assets[rel] = digest
                bindings[role] = {"path": str(destination / rel), "sha256": digest}
                original[role] = {"path": str(source), "sha256": digest}
            geometry = _geometry(copies[-2][1], copies[-1][1])
            groups.setdefault((spec.target, row["split"]), []).append(
                {"case_id": case, "patient_id": patient, **bindings})
            mappings.setdefault(patient, {"patient_key": row["patient_key"], "split": row["split"], "source_cases": []})
            source_case = {"dataset": spec.dataset, "case_id": spec.case_id}
            if source_case not in mappings[patient]["source_cases"]:
                mappings[patient]["source_cases"].append(source_case)
            source_rows.append({"case_id": case, "selection": row, "sources": original,
                                "geometry": geometry, "provenance": "configured_doctor_reference"})
        private = {"schema": "segmentation-cohort-private-v1", "plan": plan,
                   "patients": mappings, "sources": source_rows,
                   "config_path": str(config_path), "cache_path": str(cache_path) if cache_path else None}
        private_sha = _write(staging / "private/manifest.json", _bytes(private))
        cohorts, counts, cohort_hashes = {}, {}, {}
        for (target, split), rows in sorted(groups.items()):
            rel = "cohorts/" + target + "/" + split + ".json"
            data = {"schema": "segmentation-cycle-cohort-v1", "target": target, "cases": rows}
            digest = _write(staging / rel, _bytes(data))
            cohort_hashes[rel] = digest
            cohorts.setdefault(target, {})[split] = {"path": str(destination / rel), "sha256": digest}
            counts.setdefault(target, {})[split] = {"cases": len(rows), "patients": len({r["patient_id"] for r in rows})}
        splits = [{"patient_id": patient, "split": value["split"]} for patient, value in sorted(mappings.items())]
        provenance = {"schema": "segmentation-cohort-provenance-v1", "plan_sha256": _hash(plan),
                      "config_sha256": plan["config_sha256"], "cache_sha256": plan.get("cache_sha256"),
                      "split_sha256": _hash(splits), "splits": splits,
                      "source_sha256": _hash(source_rows), "asset_hashes": assets,
                      "cohort_hashes": cohort_hashes, "private_manifest_sha256": private_sha,
                      "counts": counts, "baseline_exposure": "UNKNOWN",
                      "final_test_independence": "NOT_ESTABLISHED", "production_writes": 0}
        digest = _write(staging / "manifest.json", _bytes(provenance))
        _write(staging / "manifest.sha256", (digest + "  manifest.json\n").encode("ascii"))
        # Recheck the entire set after the LAST copy, not only each pair on arrival.
        for source, target, expected in copies:
            _require(_sha_file(_path(source)) == expected and _sha_file(_path(target)) == expected,
                     "source_or_copy_changed")
        for source, expected in inputs:
            _require(_sha_file(_path(source)) == expected, "config_or_cache_changed")
        expected_files = {**assets, **cohort_hashes, "private/manifest.json": private_sha, "manifest.json": digest}
        for relative, expected in expected_files.items():
            _require(_sha_file(_path(staging / relative)) == expected, "staged_manifest_changed")
        _require((staging / "manifest.sha256").read_bytes() ==
                 (digest + "  manifest.json\n").encode("ascii"), "staged_manifest_changed")
        for path in staging.rglob("*"):
            path.chmod(0o500 if path.is_dir() else 0o400)
        staging.chmod(0o500)
        _path(parent, directory=True)
        _publish_noreplace(staging, destination)
        staging = None
        return {"status": "FROZEN", "manifest": {"path": str(destination / "manifest.json"), "sha256": digest},
                "cohorts": cohorts, "counts": counts, "baseline_exposure": "UNKNOWN",
                "final_test_independence": "NOT_ESTABLISHED", "production_writes": 0}
    except CohortError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, nib.filebasedimages.ImageFileError) as exc:
        # Never leak patient paths through parser/IO exceptions in a board response.
        raise CohortError("cohort_input_or_io_error:" + type(exc).__name__) from None
    finally:
        if staging is not None:
            _discard(staging)
