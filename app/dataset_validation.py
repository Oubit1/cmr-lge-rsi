"""Read-only current-byte QC using the deployed calibration and acceptance floor.

Only runtime journals are written. No labels, review buckets, split definitions,
or clinical policies are edited. Automatic QC is not clinical-gold annotation.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app import paired_dataset as paired

TARGETS = paired.TARGETS
_EVIDENCE_DEBT = {
    "missing_evidence", "stale_evidence_hash", "stale_dependency_hash",
    "invalid_evidence_hash", "invalid_dependency_hash", "evidence_not_accepted",
}


class PartialVolume(ValueError):
    """The scorer evaluated annotated phases, not the whole cine volume."""


def _sha(path):
    return paired._sha_file(Path(path))


def score_bundle(row: dict, profiles: dict) -> dict:
    """Use the scorer's native singleton handling; never reshape source data."""
    from app.quality_scoring import PairSpec, extract_pair, score_record
    scores = {}
    for target in TARGETS:
        p = row["paths"]
        features = extract_pair(PairSpec(
            dataset="release_validation", tier="unreviewed", target=target,
            image=Path(p[target[:3] + "_image"]), label=Path(p[target]),
            myo=Path(p["lge_myo"]) if target == "lge_mi" else None,
            relative_id="private", case_id="private", slice_position=0, reviewed=False))
        if features.get("valid") is not True:
            raise ValueError("invalid_features:" + target)
        if features.get("sparse_phase"):
            raise PartialVolume("sparse_phase_not_whole_volume:" + target)
        scores[target] = score_record(features, profiles[target])["score"]
    _check_scores(scores)
    return scores


def _check_scores(scores):
    if not isinstance(scores, dict) or set(scores) != set(TARGETS):
        raise ValueError("invalid_score_targets")
    for target, value in scores.items():
        if type(value) not in (int, float) or not 0 <= value <= 100 or not math.isfinite(value):
            raise ValueError("invalid_score:" + target)


def _safe_optional(path):
    """Check existing ancestors too, including dangling links and junctions."""
    for component in reversed((path,) + tuple(path.parents)):
        if os.path.lexists(component):
            paired._check_path(component)


def _audit(dataset, audit):
    # JSON snapshot rejects cycles, nonfinite values and non-serializable inputs.
    try:
        snapshot = json.loads(paired._json_bytes(audit))
        if (not isinstance(snapshot, dict) or snapshot.get("schema") != "paired-audit-v1"
                or snapshot.get("dataset") != str(dataset) or snapshot.get("issues") != []):
            raise ValueError("invalid_validation_audit")
        # dataset_release.run persists these annotations outside the core digest.
        # Strip only that known envelope; unknown fields must still fail the hash.
        for field in ("generated_at", "evidence_sources", "frozen_patient_count"):
            snapshot.pop(field, None)
        digest = snapshot.pop("audit_sha256")
        if paired._digest(snapshot) != digest:
            raise ValueError("validation_audit_changed")
        if (set(snapshot) != {"schema", "dataset", "pairing", "evidence", "excluded_patients",
                              "issues", "rows", "summary"}
                or not isinstance(snapshot["evidence"], list)
                or not isinstance(snapshot["rows"], list)
                or not isinstance(snapshot["summary"], dict)
                or not isinstance(snapshot["excluded_patients"], list)
                or any(not isinstance(p, str) or not p or "/" in p or "\\" in p or p in (".", "..")
                       for p in snapshot["excluded_patients"])):
            raise ValueError("invalid_validation_audit")
        if any(not isinstance(r, dict) or not paired._valid_key(r.get("key"))
               or r.get("target") not in TARGETS for r in snapshot["evidence"]):
            raise ValueError("invalid_validation_audit_evidence")
        keys = set()
        for row in snapshot["rows"]:
            if (not isinstance(row, dict) or set(row) != {
                    "key", "paths", "hashes", "metadata", "file_errors", "complete_files",
                    "releasable", "blockers", "accepted_evidence", "provenance"}
                    or not paired._valid_key(row["key"]) or row["key"] in keys):
                raise ValueError("invalid_validation_audit_row")
            keys.add(row["key"])
            for field in ("paths", "hashes", "metadata", "file_errors"):
                if not isinstance(row[field], dict) or set(row[field]) - set(paired.LAYOUT):
                    raise ValueError("invalid_validation_audit_roles")
            for role, path in row["paths"].items():
                if path != str(dataset / paired.LAYOUT[role] / row["key"]):
                    raise ValueError("source_role_mismatch")
            if (type(row["complete_files"]) is not bool or type(row["releasable"]) is not bool
                    or row["complete_files"] != (set(row["paths"]) == set(paired.LAYOUT))):
                raise ValueError("invalid_validation_audit_row")
            for field in ("blockers", "accepted_evidence", "provenance"):
                if (not isinstance(row[field], dict) or set(row[field]) != set(TARGETS)
                        or any(not isinstance(v, list) for v in row[field].values())):
                    raise ValueError("invalid_validation_audit_targets")
            if any(not isinstance(reason, str) for reasons in row["blockers"].values() for reason in reasons):
                raise ValueError("invalid_validation_audit_blockers")
        return snapshot
    except (KeyError, TypeError, RecursionError) as exc:
        raise ValueError("invalid_validation_audit") from exc


def _frozen(root, dataset):
    from app.check_validation import _digest as holdout_digest
    from app.segmentation_evidence import frozen_patients
    from app.rsi_kernel import canonical_hash
    # Reuse the existing patient mapping, while verifying its frozen contracts.
    for relative in ("runtime/check_iterations/holdout_v1.json", "runtime/lge_execution/holdout.json"):
        path = root / relative
        _safe_optional(path)
        if not path.exists():
            continue
        value = json.loads(paired._source(root, path).read_text(encoding="utf-8"))
        if "check_iterations" in relative:
            expected = holdout_digest(sorted(value["holdout"], key=lambda r: r["key"]))
            if value.get("holdout_sha256") != expected:
                raise ValueError("frozen_holdout_changed")
        elif value.get("sha256") != canonical_hash({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("frozen_holdout_changed")
    return frozen_patients(root, dataset, case_salt=os.environ.get("CMR_LGE_CASE_SALT"))


def _stem(name):
    return name[:-7] if name.endswith(".nii.gz") else name.removesuffix(".nii")


def _blocked_keys(dataset, keys):
    blocked = set()
    for modality in ("cmr", "lge"):
        names = set()
        for folder in ("待筛", "待处理", "垃圾桶"):
            base = dataset / "labels/check" / folder / modality
            _safe_optional(base)
            pending = [base] if base.exists() else []
            while pending:
                directory = pending.pop()
                paired._check_path(directory)
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = directory / entry.name
                        paired._check_path(path)  # No traversal through links/cycles.
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(path)
                        elif path.suffix.lower() == ".png":
                            paired._source(dataset, path)
                            relative = path.relative_to(base).with_suffix("").as_posix()
                            names.add(_stem(relative))
                            if modality == "cmr":
                                # Existing CMR renderers emit patient_volume_sNN.
                                names.add(_stem(re.sub(r"_(?:s|f|k|frame)\d+$", "", relative)))
        for key in keys:
            relative = Path(key)
            patient, stem = relative.parts[0], _stem(relative.name)
            aliases = {relative.with_name(stem).as_posix(),
                       relative.with_name(patient + "_" + stem).as_posix(),
                       patient + "_" + stem}
            if aliases & names:
                blocked.add(key)
    return blocked


def _current(dataset, row):
    if set(row["paths"]) != set(paired.LAYOUT):
        raise ValueError("incomplete_bundle")
    hashes = {}
    for role, directory in paired.LAYOUT.items():
        expected = dataset / directory / row["key"]
        if row["paths"][role] != str(expected):
            raise ValueError("source_role_mismatch")
        hashes[role] = _sha(paired._source(dataset, expected))
    return hashes


def _verify_selected(dataset, row, evidence):
    """Re-inspect just this five-file bundle, using the auditor's exact contract.

    A caller can recompute a digest after tampering, so its metadata and blockers
    cannot authorize acceptance. Geometry comparisons mirror audit_pairs; they
    compare metadata only, and do not reshape images or change clinical policy.
    """
    import numpy as np
    hashes, metadata, errors = {}, {}, {}
    for role, directory in paired.LAYOUT.items():
        path = paired._source(dataset, dataset / directory / row["key"])
        details, errors[role] = paired._inspect(dataset, path, role in TARGETS)
        if "sha256" in details:
            hashes[role] = details.pop("sha256")
        metadata[role] = details
    if (hashes != row["hashes"] or metadata != row["metadata"] or errors != row["file_errors"]):
        raise ValueError("validation_audit_stale_or_changed")
    if any(errors.values()):
        raise ValueError("invalid_bundle_files")
    for target in TARGETS:
        image = target[:3] + "_image"
        def spatial_shape(shape):
            result = list(shape)
            while len(result) > 2 and result[-1] == 1:
                result.pop()
            return result
        if (spatial_shape(metadata[target]["shape"]) != spatial_shape(metadata[image]["shape"])
                or not np.allclose(metadata[target]["affine"], metadata[image]["affine"],
                                   rtol=0, atol=1e-5)):  # Existing audit_pairs geometry tolerance.
            raise ValueError("invalid_bundle_geometry")
        candidates = [r for r in evidence if r.get("key") == row["key"] and r.get("target") == target]
        current = [r for r in candidates if not paired._evidence_errors(r, target, hashes)]
        reasons = sorted({reason for r in candidates for reason in paired._evidence_errors(r, target, hashes)})
        expected = [] if current else reasons or ["missing_evidence"]
        if (row["blockers"][target] != expected or row["accepted_evidence"][target] != current
                or row["provenance"][target] != [r["provenance"] for r in current]):
            raise ValueError("validation_audit_stale_or_changed")
    return hashes


def _read_lines(path):
    _safe_optional(path)
    if not path.exists():
        return []
    paired._source(path.parent, path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("invalid_validation_journal")
            paired._json_bytes(row)
            rows.append(row)
    return rows


def _append(path, records):
    from app.segmentation_registry import _sync_dir
    data = b"".join(paired._json_bytes(r) + b"\n" for r in records)
    _safe_optional(path)
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def validate_batch(root: Path, dataset: Path, audit: dict, *, limit: int = 16,
                   key_filter: list[str] | None = None) -> dict:
    """Serialize jobs; score each unchanged input at most once, including errors.

    STARTED is durable before scoring, so a crash is suppressed on the next run.
    Errors are journaled and re-raised; they never become accepted evidence. New
    bytes/calibration/wrapper code permit a new attempt; A->B->A does not retry A.
    """
    from app.segmentation_registry import _locked
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("invalid_validation_batch_limit")
    if key_filter is not None and (not isinstance(key_filter, list)
            or any(not paired._valid_key(key) for key in key_filter)
            or len(key_filter) != len(set(key_filter))):
        raise ValueError("invalid_validation_key_filter")
    root, dataset = paired._root(root), paired._root(dataset)
    runtime = root / "runtime/dataset_acceptance"
    if runtime == dataset or dataset in runtime.parents:
        raise ValueError("validation_runtime_inside_dataset")
    _safe_optional(runtime)
    with _locked(runtime):
        return _validate_batch(root, dataset, audit, limit=limit, key_filter=key_filter)


def _decision_records(attempt, ids, threshold, *, scores=None, status="PENDING", reason="validation_in_progress"):
    """Negative/provisional decisions supersede old evidence; never edit labels."""
    records = []
    for target in TARGETS:
        current = attempt["input_hashes"]
        hashes = {"image": current[target[:3] + "_image"], "label": current[target]}
        if target.startswith("lge"):
            other = "mi" if target == "lge_myo" else "myo"
            hashes[other] = current["lge_" + other]
        decision, target_reason = status, reason
        provenance = {"kind": "automatic_qc", "record_id": attempt["validation_signature"] + ":" + target,
                      "calibration_id": ids[target], "threshold": threshold, "label_write": False,
                      "policy": "existing_calibrated_grade_C_MI_requires_Myo"}
        if scores is not None:
            provenance["score"] = scores[target]
            decision = "ACCEPTED" if scores[target] >= threshold else "NEEDS_REVIEW"
            target_reason = "existing_quality_floor_met" if decision == "ACCEPTED" else "below_existing_quality_floor"
            if target == "lge_mi" and scores["lge_myo"] < threshold:
                decision, target_reason = "NEEDS_REVIEW", "myo_dependency_not_accepted"
        provenance.update(provisional=decision != "ACCEPTED", reason=target_reason)
        record = {**attempt, "target": target, "status": decision, "scope": "whole_volume",
                  "hashes": hashes, "provenance": provenance}
        expected_errors = [] if decision == "ACCEPTED" else ["evidence_not_accepted"]
        if paired._evidence_errors(record, target, current) != expected_errors:
            raise ValueError("invalid_acceptance_receipt")
        records.append(record)
    return records


def _validate_batch(root, dataset, audit, *, limit, key_filter=None):
    from app.lge_execution import calibration
    from app.check_auto_apply import MIN_QUALITY_SCORE
    audit = _audit(dataset, audit)
    selected_keys = None if key_filter is None else set(key_filter)
    if selected_keys is not None and selected_keys - {row["key"] for row in audit["rows"]}:
        raise ValueError("validation_key_not_in_audit")
    excluded = set(audit["excluded_patients"]) | _frozen(root, dataset)
    blocked = _blocked_keys(dataset, [row["key"] for row in audit["rows"]])
    profiles, ids = {}, {}
    for target in TARGETS:
        profiles[target], ids[target] = calibration(root, "cmr_paired" if target == "cmr_myo" else target)
        if not isinstance(profiles[target], dict) or not isinstance(ids[target], str) or not ids[target]:
            raise ValueError("invalid_calibration")
    runtime = root / "runtime/dataset_acceptance"
    ledger, attempts = runtime / "receipts.jsonl", runtime / "attempts.jsonl"
    # Read receipts as well: a damaged last line is an error, never ignored.
    old = _read_lines(attempts) + _read_lines(ledger)
    previous = {r["validation_signature"] for r in old if r.get("validation_signature")}
    counts, attempted = Counter(), 0
    wrapper = _sha(Path(__file__))
    for row in audit["rows"]:
        if selected_keys is not None and row["key"] not in selected_keys:
            continue
        if attempted >= limit:
            break
        if row["key"] in blocked:
            counts["existing_review_or_exclusion"] += 1
            continue
        invalid = {reason for reasons in row["blockers"].values() for reason in reasons}
        if (row["key"].split("/", 1)[0] in excluded
                or not row["complete_files"] or invalid - _EVIDENCE_DEBT):
            counts["structural_or_frozen_exclusion"] += 1
            continue
        signature = paired._digest({"dataset": str(dataset), "key": row["key"],
                                    "hashes": row["hashes"], "calibrations": ids,
                                    "wrapper": wrapper, "threshold": MIN_QUALITY_SCORE})
        if signature in previous:
            counts["unchanged_already_validated"] += 1
            continue
        attempt = {"dataset": str(dataset), "key": row["key"], "validation_signature": signature,
                   "timestamp": datetime.now(timezone.utc).isoformat(), "input_hashes": row["hashes"],
                   "calibrations": ids, "threshold": MIN_QUALITY_SCORE}
        _append(attempts, [dict(attempt, status="STARTED")])
        previous.add(signature)
        attempted += 1
        try:
            # Invalidate old acceptances before any fallible inspection/scoring.
            # A crash leaves explicitly provisional evidence, never stale success.
            _append(ledger, _decision_records(attempt, ids, MIN_QUALITY_SCORE))
            current = _verify_selected(dataset, row, audit["evidence"])
            scores = score_bundle(row, profiles)
            _check_scores(scores)
            if _current(dataset, row) != current:
                raise ValueError("source_changed")
            # The review workflow and holdout can change while scoring runs.
            if (row["key"] in _blocked_keys(dataset, [row["key"]])
                    or row["key"].split("/", 1)[0] in _frozen(root, dataset)):
                raise ValueError("exclusion_changed")
            for target in TARGETS:
                _, current_id = calibration(root, "cmr_paired" if target == "cmr_myo" else target)
                if current_id != ids[target]:
                    raise ValueError("calibration_changed")
            accepted = {t for t in TARGETS if scores[t] >= MIN_QUALITY_SCORE}
            if "lge_myo" not in accepted:
                accepted.discard("lge_mi")
            records = _decision_records(attempt, ids, MIN_QUALITY_SCORE, scores=scores)
            # Recheck after policy and receipt construction, immediately before append.
            if _current(dataset, row) != current:
                raise ValueError("source_changed")
            _append(ledger, records)
            status = "ACCEPTED" if len(accepted) == len(TARGETS) else "NEEDS_REVIEW"
            _append(attempts, [dict(attempt, status=status, scores=scores)])
            counts["accepted_pairs" if status == "ACCEPTED" else "needs_review_pairs"] += 1
            counts["accepted_targets"] += len(accepted)
        except PartialVolume:
            _append(ledger, _decision_records(attempt, ids, MIN_QUALITY_SCORE,
                    status="INELIGIBLE", reason="sparse_phase_not_whole_volume"))
            _append(attempts, [dict(attempt, status="INELIGIBLE", reason="sparse_phase_not_whole_volume")])
            counts["partial_volume_exclusion"] += 1
        except Exception as exc:
            try:
                _append(ledger, _decision_records(attempt, ids, MIN_QUALITY_SCORE,
                        status="ERROR", reason="validation_error:" + type(exc).__name__))
                _append(attempts, [dict(attempt, status="ERROR", error_type=type(exc).__name__)])
            except Exception as journal_error:
                # STARTED already prevents a blind retry; preserve the scoring error.
                raise exc from journal_error
            raise
    return {"status": "VALIDATED", "attempted": attempted, "counts": dict(counts),
            "calibrations": ids, "production_labels_written": 0, "provider_calls": 0}
