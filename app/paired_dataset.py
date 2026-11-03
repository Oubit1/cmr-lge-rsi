"""Mechanical five-file bundle verification; no clinical adjudication.

Keys are exact, case-sensitive patient/relative-file names, including the NIfTI
extension. They are a pairing convention, NOT proof of anatomical registration.
Evidence is a caller-supplied, trusted record, not an authentication mechanism.
Only ACCEPTED whole_volume records bound to current bytes can authorize a target.
Historical/partial records never acquire acceptance from a directory name.

``summary`` contains counts and reason codes only. The rest of an audit is local,
private material. Freeze exports only releasable rows, revalidates the entire
audit, and publishes a new directory with an atomic no-replace rename. Public
manifest paths use random identifiers; source keys and provenance live in
private/manifest.json. Copied NIfTI headers are NOT anonymized. Read-only modes
and content digests provide application-level immutability/tamper detection,
not protection against a filesystem owner deliberately changing permissions.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
import zlib
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


LAYOUT = {
    "cmr_image": "images/cmr",
    "lge_image": "images/lge",
    "cmr_myo": "labels/cmr_checked/cmr_Myo_mask",
    "lge_myo": "labels/lge_checked/lge_Myo_labels",
    "lge_mi": "labels/lge_checked/lge_MI_labels",
}
TARGETS = ("cmr_myo", "lge_myo", "lge_mi")
PAIRING = "Exact relative key only; not proof of anatomical registration."
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_KINDS = ("human", "automatic_qc", "doctor_reference")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _is_nifti(name: str) -> bool:
    return name.endswith((".nii", ".nii.gz"))


def _valid_key(key: object) -> bool:
    if not isinstance(key, str) or not _is_nifti(key):
        return False
    parts = key.split("/")
    return (len(parts) >= 2 and not any(c in key for c in "\\:\x00")
            and all(p and p not in (".", "..") and p == p.rstrip(" .")
                    for p in parts))


def _check_path(path: Path) -> None:
    """Reject symlinks and Windows reparse points in *every* path component."""
    for component in reversed((path,) + tuple(path.parents)):
        info = component.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("unsafe_path")


def _root(path: Path) -> Path:
    # Do not resolve first: resolving would hide a symlink in the supplied path.
    path = Path(os.path.abspath(path))
    _check_path(path)
    if not path.is_dir():
        raise ValueError("expected_directory")
    return path


def _source(root: Path, path: Path) -> Path:
    path.relative_to(root)  # lexical and resolved containment are both required
    _check_path(path)
    path.resolve(strict=True).relative_to(root)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("not_regular_file")
    return path


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(root: Path, relative: str) -> tuple[dict, list]:
    base = root / relative
    files, issues = {}, []
    pending = [base]
    while pending:
        directory = pending.pop()
        try:
            _check_path(directory)
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
            for entry in children:
                path = directory / entry.name
                try:
                    _check_path(path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif _is_nifti(entry.name):
                        key = path.relative_to(base).as_posix()
                        if not _valid_key(key):
                            raise ValueError("invalid_key")
                        files[key] = _source(root, path)
                except (OSError, ValueError):
                    issues.append({"path": str(path), "reason": "unsafe_path"})
        except FileNotFoundError:
            if directory != base:
                issues.append({"path": str(directory), "reason": "scan_changed"})
        except (OSError, ValueError):
            issues.append({"path": str(directory), "reason": "unsafe_path"})
    return files, issues


def _inspect(root: Path, path: Path, mask: bool) -> tuple[dict, list]:
    details, errors = {}, []
    try:
        before = _sha_file(_source(root, path))
        details["sha256"] = before
        volume = nib.load(str(path), mmap=False)
        data = np.asanyarray(volume.dataobj)
        affine = np.asarray(volume.affine)
        details.update(shape=list(data.shape), affine=affine.tolist())
        if data.size == 0 or not np.isfinite(data).all():
            errors.append("nonfinite_or_empty_data")
        if not np.isfinite(affine).all():
            errors.append("nonfinite_affine")
            details.pop("affine")  # keep the audit JSON finite
        if mask and not (np.isin(data, (0, 1)).all()
                         or np.isin(data, (0, 255)).all()):
            errors.append("nonbinary_mask")
        if _sha_file(_source(root, path)) != before:
            errors.append("source_changed")
    except (OSError, ValueError, EOFError, zlib.error, nib.filebasedimages.ImageFileError):
        errors.append("unreadable_or_unsafe_file")
    return details, errors


def _evidence_errors(record: dict, target: str, hashes: dict) -> list[str]:
    errors = []
    if record.get("status") != "ACCEPTED":
        errors.append("evidence_not_accepted")
    if record.get("scope") != "whole_volume":
        errors.append("evidence_not_whole_volume")
    provenance = record.get("provenance")
    if (not isinstance(provenance, dict) or provenance.get("kind") not in _KINDS
            or not isinstance(provenance.get("record_id"), str)
            or not provenance["record_id"].strip()):
        errors.append("invalid_provenance")
    expected = {"image": target[:3] + "_image", "label": target}
    if target == "lge_myo":
        expected["mi"] = "lge_mi"
    elif target == "lge_mi":
        expected["myo"] = "lge_myo"
    provided = record.get("hashes")
    if not isinstance(provided, dict):
        return sorted(set(errors + ["invalid_evidence_hash"]))
    if set(provided) - {"image", "label", "myo", "mi"}:
        errors.append("invalid_evidence_hash")
    # Validate optional counterpart hashes too; never silently discard a binding.
    for field in ("myo", "mi"):
        if field in provided:
            expected[field] = target[:3] + "_" + field
    for field, role in expected.items():
        value = provided.get(field)
        dependency = field in ("myo", "mi")
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            errors.append("invalid_dependency_hash" if dependency else "invalid_evidence_hash")
        elif role not in hashes or value.lower() != hashes[role]:
            errors.append("stale_dependency_hash" if dependency else "stale_evidence_hash")
    return sorted(set(errors))


def audit_pairs(dataset: Path, evidence: list[dict], *,
                excluded_patients: set[str] | None = None) -> dict:
    """Return private rows plus a name-free summary; never alter source files.

    complete_files counts keys with five safe regular files, even when content
    or evidence fails. blockers_by_target counts blocked keys per reason, not
    records. A current valid record may coexist with stale historical records;
    all current valid records are retained. Malformed record identities and any
    unsafe scan paths fail closed for the audit as a whole.
    """
    root = _root(dataset)
    if not isinstance(evidence, list):
        raise ValueError("evidence_must_be_list")
    # Snapshot caller-owned data and enforce a portable, finite JSON contract.
    try:
        records = json.loads(_json_bytes(evidence))
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_must_be_json") from exc
    if excluded_patients is not None and not isinstance(excluded_patients, set):
        raise ValueError("excluded_patients_must_be_set")
    excluded = set(excluded_patients or ())
    if any(not isinstance(item, str) or not item or "/" in item or "\\" in item
           or item in (".", "..") for item in excluded):
        raise ValueError("invalid_excluded_patient")
    indexed, issues, files = {}, [], {}
    for record in records:
        if (not isinstance(record, dict) or not _valid_key(record.get("key"))
                or record.get("target") not in TARGETS):
            issues.append({"reason": "invalid_evidence_identity"})
            continue
        indexed.setdefault((record["key"], record["target"]), []).append(record)
    for role, directory in LAYOUT.items():
        files[role], found = _scan(root, directory)
        issues.extend(dict(item, role=role) for item in found)
    keys = sorted(set().union(*(set(group) for group in files.values()),
                              {key for key, _ in indexed}))
    rows = []
    counts = {target: Counter() for target in TARGETS}
    for key in keys:
        paths, hashes, metadata, file_errors = {}, {}, {}, {}
        for role in LAYOUT:
            path = files[role].get(key)
            if path is None:
                file_errors[role] = ["missing_file"]
                continue
            paths[role] = str(path)
            details, file_errors[role] = _inspect(root, path, role in TARGETS)
            if "sha256" in details:
                hashes[role] = details.pop("sha256")
            metadata[role] = details
        blockers, current = {}, {}
        for target in TARGETS:
            image = target[:3] + "_image"
            reasons = list(file_errors[target]) + list(file_errors[image])
            label_meta, image_meta = metadata.get(target, {}), metadata.get(image, {})
            if "shape" in label_meta and "shape" in image_meta:
                def spatial_shape(shape):
                    result = list(shape)
                    while len(result) > 2 and result[-1] == 1:
                        result.pop()
                    return result
                if spatial_shape(label_meta["shape"]) != spatial_shape(image_meta["shape"]):
                    reasons.append("shape_mismatch")
                if "affine" in label_meta and "affine" in image_meta and not np.allclose(
                        label_meta["affine"], image_meta["affine"], rtol=0, atol=1e-5):
                    reasons.append("affine_mismatch")
            candidates = indexed.get((key, target), [])
            valid, rejected = [], []
            for record in candidates:
                errors = _evidence_errors(record, target, hashes)
                if errors:
                    rejected.extend(errors)
                else:
                    valid.append(record)
            current[target] = valid
            if not valid:
                reasons.extend(rejected or ["missing_evidence"])
            if key.split("/", 1)[0] in excluded:
                reasons.append("excluded_patient")
            if issues:
                reasons.append("audit_input_unsafe")
            blockers[target] = sorted(set(reasons))
            counts[target].update(blockers[target])
        rows.append({"key": key, "paths": paths, "hashes": hashes,
                     "metadata": metadata, "file_errors": file_errors,
                     "complete_files": len(paths) == len(LAYOUT),
                     "releasable": not any(blockers.values()),
                     "blockers": blockers, "accepted_evidence": current,
                     "provenance": {target: [r["provenance"] for r in current[target]]
                                    for target in TARGETS}})
    result = {
        "schema": "paired-audit-v1", "dataset": str(root), "pairing": PAIRING,
        "evidence": records, "excluded_patients": sorted(excluded),
        "issues": issues, "rows": rows,
        "summary": {"keys": len(rows),
                    "files_by_role": {role: len(group) for role, group in files.items()},
                    "complete_files": sum(row["complete_files"] for row in rows),
                    "releasable_pairs": sum(row["releasable"] for row in rows),
                    "blocked_pairs": sum(not row["releasable"] for row in rows),
                    "excluded_pairs": sum(row["key"].split("/", 1)[0] in excluded for row in rows),
                    "blockers_by_target": {t: dict(sorted(c.items())) for t, c in counts.items()},
                    "input_issues": dict(sorted(Counter(i["reason"] for i in issues).items()))},
    }
    result["audit_sha256"] = _digest(result)
    return result


def _publish_noreplace(staging: Path, destination: Path) -> None:
    """A check followed by ordinary POSIX rename can overwrite an empty dir."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows rename refuses existing targets.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(staging), os.fsencode(destination), 4)
    else:
        raise OSError("atomic_no_replace_not_supported")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _seal(root: Path) -> None:
    # All bytes remain private, including original NIfTI headers and the
    # pseudonymized manifest. Read-only must not grant access to other users.
    # On Windows chmod only sets read-only; the parent must have a private ACL.
    for path in root.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def _discard(root: Path) -> None:
    # Only the freshly created staging tree is ever passed here.
    root.chmod(0o700)
    for directory, dirs, files in os.walk(root):
        for name in dirs:
            (Path(directory) / name).chmod(0o700)
        for name in files:
            (Path(directory) / name).chmod(0o600)
    shutil.rmtree(root)


def freeze_release(dataset: Path, audit: dict, destination: Path) -> dict:
    """Copy the audited releasable subset and atomically publish a new release.

    The destination's parent must already exist, outside the dataset. No empty
    release is published. All audit fields are digest-bound and re-derived;
    drift anywhere in the audit requires a new audit. Failure removes staging.
    manifest.sha256 binds canonical manifest.json bytes; that manifest also
    binds the private provenance manifest and every copied source byte.
    """
    root = _root(dataset)
    destination = Path(os.path.abspath(destination))
    parent = _root(destination.parent)
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("release_must_be_outside_dataset")
    if os.path.lexists(destination):
        raise FileExistsError("destination_exists")
    if not isinstance(audit, dict):
        raise ValueError("invalid_audit")
    try:
        snapshot = json.loads(_json_bytes(audit))
        expected_digest = snapshot.pop("audit_sha256")
        if _digest(snapshot) != expected_digest or snapshot["dataset"] != str(root):
            raise ValueError("audit_changed")
        fresh = audit_pairs(root, snapshot["evidence"],
                            excluded_patients=set(snapshot["excluded_patients"]))
        if fresh["audit_sha256"] != expected_digest:
            raise ValueError("audit_stale")
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid_audit") from exc
    selected = [row for row in fresh["rows"] if row["releasable"]]
    if not selected or fresh["issues"]:
        raise ValueError("no_releasable_pairs")
    staging = Path(tempfile.mkdtemp(prefix=".paired-release-", dir=str(parent)))
    try:
        patients, public_rows, private_rows, copies = {}, [], [], []
        for row in selected:
            patient = row["key"].split("/", 1)[0]
            patient_id = patients.setdefault(patient, "p_" + uuid.uuid4().hex)
            bundle_id = "b_" + uuid.uuid4().hex
            bundle = staging / "data" / patient_id / bundle_id
            bundle.mkdir(parents=True)
            exported = {}
            for role in LAYOUT:
                source = _source(root, Path(row["paths"][role]))
                expected = row["hashes"][role]
                if _sha_file(source) != expected:
                    raise ValueError("source_changed")
                suffix = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
                target = bundle / (role + suffix)
                # copyfile copies bytes, not inode links or source permissions.
                shutil.copyfile(source, target)
                with target.open("r+b") as stream:
                    os.fsync(stream.fileno())
                copies.append((source, target, expected))
                exported[role] = {"path": target.relative_to(staging).as_posix(),
                                  "sha256": expected, "bytes": target.stat().st_size}
            public_rows.append({"patient_id": patient_id, "bundle_id": bundle_id,
                                "files": exported})
            private_rows.append({"patient_id": patient_id, "bundle_id": bundle_id,
                                 "source_key": row["key"], "provenance": row["provenance"],
                                 "evidence": row["accepted_evidence"]})
        (staging / "private").mkdir(mode=0o700)
        private_bytes = _json_bytes({"schema": "paired-private-v1", "bundles": private_rows})
        _write(staging / "private" / "manifest.json", private_bytes)
        manifest = {"schema": "paired-release-v1", "pairing": PAIRING,
                    "bundle_count": len(selected), "bundles": public_rows,
                    "private_manifest_sha256": hashlib.sha256(private_bytes).hexdigest()}
        manifest_bytes = _json_bytes(manifest)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        _write(staging / "manifest.json", manifest_bytes)
        digest_bytes = (manifest_digest + "  manifest.json\n").encode("ascii")
        _write(staging / "manifest.sha256", digest_bytes)
        _seal(staging)
        # Verify *all* copies and sources after the last copy, not just each file
        # immediately after its own copy (which misses later dependency drift).
        for source, target, expected in copies:
            if (_sha_file(_source(root, source)) != expected
                    or _sha_file(_source(staging, target)) != expected):
                raise ValueError("source_or_copy_changed")
        if (_sha_file(staging / "manifest.json") != manifest_digest
                or _sha_file(staging / "private" / "manifest.json")
                != manifest["private_manifest_sha256"]
                or (staging / "manifest.sha256").read_bytes() != digest_bytes):
            raise ValueError("manifest_changed")
        _root(parent)  # Recheck ancestor links immediately before publication.
        _publish_noreplace(staging, destination)
    except BaseException:
        _discard(staging)
        raise
    return {"destination": str(destination), "releasable_pairs": len(selected),
            "manifest_sha256": manifest_digest, "manifest": manifest}
