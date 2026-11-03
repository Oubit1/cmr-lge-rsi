"""Compile frozen five-file releases into explicit vendor training jobs.

New compilation requires assignment_contract={path, sha256}, identifying a
PRIVATE, controller-owned frozen assignment file. freeze_assignment_contract()
explicitly records the caller's first plan; it never chooses a split, overwrites
an existing contract or silently enrolls new patients. The controller must reuse
the same authoritative binding across jobs, targets and releases. An arbitrary
Owner-selected replacement contract is not authority to change that history.

Keys are the release private source_key's first component. Their UTF-8 SHA-256
is the same patient_id used by segmentation_cohort for the SAME patient_key.
Caller assignments may be a subset of the contract but every supplied entry
must agree, and every released patient must be explicitly assigned.

For deliberate legacy callers only, allow_unbound_assignments=True permits the
old API without a contract and emits UnboundAssignmentsWarning: cross-job split
consistency is NOT enforced on that path. Invalid supplied contracts never fall
back. The trainer's v1 JSON schema stays unchanged: normalized splits/records
are job-hash bound; the host must retain the contract binding with its compilation
provenance. Direct hand-authored jobs are outside this compiler's enforcement.
Hashes identify bytes, not reviewer approval or filesystem access permissions.
Architecture/optimizer choices remain owned by the domain role's plan.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

from app import segmentation_training as training
from app.segmentation_training import sha256_file, validate_job


VENDORS = {"cmr_myo": "automated_annotation_of_cine", "lge_myo": "automated_annotation_of_lge",
           "lge_mi": "automated_annotation_of_lge_mi"}
ASSIGNMENT_SCHEMA = "segmentation-assignment-contract-v1"
PATIENT_IDENTITY = "sha256-utf8-private-patient-key-v1"


class UnboundAssignmentsWarning(UserWarning):
    """Explicit legacy compilation does not enforce assignments across jobs."""


def _assignments(value: dict) -> dict:
    if type(value) is not dict or not value:
        raise ValueError("invalid_patient_assignments")
    for patient, split in value.items():
        if (type(patient) is not str or not patient.strip() or patient != patient.strip()
                or "\x00" in patient or type(split) is not str
                or split not in {"train", "validation", "test"}):
            raise ValueError("invalid_patient_assignment")
    return dict(sorted(value.items()))


def _read_assignment_contract(binding: dict) -> dict:
    checked = training._file(binding)
    content = Path(checked["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != checked["sha256"]:
        raise ValueError("assignment_contract_changed")
    data = training._decode_json(content.decode("utf-8"))
    training._keys(data, ("schema", "patient_identity", "assignments"))
    if data["schema"] != ASSIGNMENT_SCHEMA or data["patient_identity"] != PATIENT_IDENTITY:
        raise ValueError("invalid_assignment_contract")
    return _assignments(data["assignments"])


def freeze_assignment_contract(path: Path, assignments: dict[str, str]) -> dict:
    """Explicit trusted first-plan registration; no clinical/split selection.

    The parent must already be a private controller-owned directory (Windows
    inherits its ACL). Publish complete bytes with atomic no-replace hard-link
    creation. Concurrent different plans cannot win the same contract path.
    Identical existing assignments are idempotent; different or added entries
    are rejected without changing the contract. No implicit extension/versioning.
    Return only the {path, sha256} binding, never the private patient map.
    """
    assignments = _assignments(assignments)
    path = training._absolute(str(path))
    if not path.parent.is_dir():
        raise ValueError("assignment_contract_parent_missing")

    def existing():
        binding = {"path": str(path), "sha256": sha256_file(path)}
        if _read_assignment_contract(binding) != assignments:
            raise ValueError("assignment_contract_conflict")
        return binding

    if path.exists():
        return existing()
    content = training._canonical({"schema": ASSIGNMENT_SCHEMA,
                                   "patient_identity": PATIENT_IDENTITY,
                                   "assignments": assignments})
    descriptor, name = tempfile.mkstemp(prefix=".assignment-contract-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        training._absolute(str(path))  # Recheck ancestors before publication.
        try:
            os.link(temporary, path)  # Atomic no-replace on supported local filesystems.
        except FileExistsError:
            return existing()
    finally:
        temporary.unlink()
    path.chmod(0o400)
    binding = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    _read_assignment_contract(binding)
    return binding


def _asset(root: Path, item: dict) -> dict:
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError("release_asset_changed")
    return {"path": str(path), "sha256": item["sha256"]}


def compile_job(release: Path, *, target: str, assignments: dict[str, str], config: Path,
                vendor_root: Path, output: Path, resources: dict,
                parent_weights: Path | None = None, myo_weights: Path | None = None,
                assignment_contract: dict | None = None,
                allow_unbound_assignments: bool = False) -> dict:
    """Read-only compile; new jobs require a controller-pinned assignment binding.

    Never creates or repairs a missing contract. Use freeze_assignment_contract
    separately and explicitly for a trusted first plan. Legacy opt-out is noisy
    and provides no cross-job guarantee; supplying a contract always enforces it.
    """
    if target not in VENDORS:
        raise ValueError("unknown_target")
    if type(allow_unbound_assignments) is not bool:
        raise ValueError("invalid_unbound_assignment_opt_out")
    assignments = _assignments(assignments)
    if assignment_contract is None:
        if not allow_unbound_assignments:
            raise ValueError("assignment_contract_required")
        warnings.warn("UNBOUND_PATIENT_ASSIGNMENTS: cross-job/cross-release split consistency "
                      "is not enforced; not suitable for autonomous training.",
                      UnboundAssignmentsWarning, stacklevel=2)
    else:
        if type(assignment_contract) is not dict:
            raise ValueError("invalid_assignment_contract_binding")
        assignment_contract = dict(assignment_contract)
        frozen = _read_assignment_contract(assignment_contract)
        if any(frozen.get(patient) != split for patient, split in assignments.items()):
            raise ValueError("patient_assignment_contract_mismatch")
        if Path(assignment_contract["path"]).resolve().is_relative_to(output.resolve()):
            raise ValueError("output_overlaps_assignment_contract")
    root = release.resolve()
    path = root / "manifest.json"
    expected = (root / "manifest.sha256").read_text().split()[0]
    if sha256_file(path) != expected:
        raise ValueError("release_manifest_changed")
    data = json.loads(path.read_text())
    private_path = root / "private/manifest.json"
    if data.get("schema") != "paired-release-v1" or sha256_file(private_path) != data["private_manifest_sha256"]:
        raise ValueError("release_provenance_changed")
    private = {r["bundle_id"]: r for r in json.loads(private_path.read_text())["bundles"]}
    records, groups, patient_names = [], {k: [] for k in ("train", "validation", "test")}, {}
    if set(private) != {r["bundle_id"] for r in data["bundles"]}:
        raise ValueError("release_provenance_incomplete")
    for row in data["bundles"]:
        original = private[row["bundle_id"]]["source_key"].split("/", 1)[0]
        split = assignments.get(original)
        if split not in groups:
            raise ValueError("patient_split_missing")
        # Stable internal group hash across releases; names never enter the trainer.
        patient = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if patient in patient_names and patient_names[patient] != split:
            raise ValueError("patient_split_conflict")
        patient_names[patient] = split
        if patient not in groups[split]:
            groups[split].append(patient)
        image_role = "cmr_image" if target == "cmr_myo" else "lge_image"
        records.append({"center": "released", "patient": patient,
            "relative_path": patient + "/" + row["bundle_id"] + ".nii.gz",
            "image": _asset(root, row["files"][image_role]),
            "label": _asset(root, row["files"][target])})
    vendor = vendor_root.resolve() / VENDORS[target]
    job = {"schema_version": 1, "target": target, "dataset_root": str(root),
        "output_dir": str(output.resolve()), "config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
        "vendor": {"root": str(vendor), "files": {p.relative_to(vendor).as_posix(): sha256_file(p) for p in sorted(vendor.rglob("*.py"))}},
        "splits": groups, "records": records, "resources": dict(resources)}
    for key, value in (("parent_weights", parent_weights), ("myo_weights", myo_weights)):
        if value:
            job[key] = {"path": str(value.resolve()), "sha256": sha256_file(value)}
    validate_job(job)
    if assignment_contract is not None:
        _read_assignment_contract(assignment_contract)  # Fail on drift during compilation.
    return job
