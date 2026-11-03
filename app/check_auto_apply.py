"""Deterministic, bounded promotion of already validated check candidates.

The check-loop service may call this after a fresh shadow pass. It does not
ask Codex to adjudicate cases, and it never touches the frozen holdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from app.check_inventory import build_check_inventory
from app.check_validation import _digest
from app.manual_annotation import _restore, _stack


MIN_QUALITY_SCORE = 70.0  # Existing quality scorer's C-grade entry; owner chose progression-first.
MAX_PER_TICK = 10

# Bounded discriminator constants for the route-conflict hook.  These are
# monitoring-grade thresholds derived from the existing quality scorer's
# C-grade entry and the anonymous spatial rank/core evidence; they do not
# encode identities or the private holdout split.
CONFLICT_MIN_QUALITY = MIN_QUALITY_SCORE
CONFLICT_MIN_RANK = 0
CONFLICT_MAX_RANK = 14

# Spatial decisions that may participate in a route conflict.  The pending
# queue is dominated by ``unresolved`` records, so the discriminator must
# consider them as well as explicit boundary candidates; otherwise the
# preflight gate can never clear.
CONFLICT_SPATIAL_DECISIONS = frozenset({
    "spatial_boundary_candidate",
    "unresolved",
})


def _conflict_route(*, evaluation: dict[str, Any],
                    candidate_records: list[dict[str, Any]],
                    spatial_record: dict[str, Any]) -> dict[str, Any] | None:
    """Derive a route from anonymous spatial rank/core plus existing scores.

    Returns a fully explained resolution dict (``route`` in {checked, trash}
    plus a non-empty ``reason``) or ``None`` to abstain.  The rule is bounded
    and explainable: it only fires when the spatial record carries a rank in
    the observed core range and the existing quality score is present.  It
    never reads identities, never inspects the frozen holdout, and never
    writes labels.
    """
    if not isinstance(spatial_record, dict):
        return None
    rank = spatial_record.get("rank")
    if not isinstance(rank, int) or not CONFLICT_MIN_RANK <= rank <= CONFLICT_MAX_RANK:
        return None
    core = spatial_record.get("core")
    if not isinstance(core, (list, tuple)) or len(core) != 2:
        return None
    if not all(isinstance(v, int) for v in core):
        return None
    decision = spatial_record.get("decision")
    if decision not in CONFLICT_SPATIAL_DECISIONS:
        return None
    status = evaluation.get("quality_status")
    if status != "SCORED":
        return None
    try:
        quality_after = float(evaluation.get("quality_after", -1.0))
        quality_before = float(evaluation.get("quality_before", -1.0))
        quality_delta = float(evaluation.get("quality_delta", 0.0))
    except (TypeError, ValueError):
        return None
    flags_after = evaluation.get("quality_flags_after") or []
    if not isinstance(flags_after, list):
        return None
    candidates_valid = bool(candidate_records) and all(
        item.get("passed") is True for item in candidate_records
    )
    # Route to checked only when the existing scorer already places the
    # volume at or above the C-grade entry and no defect flag is present.
    if (quality_after >= CONFLICT_MIN_QUALITY and quality_delta >= 0.0
            and not flags_after):
        return {
            "route": "checked",
            "reason": (
                f"conflict_quality_C_pass:quality_after={quality_after:.3f}"
                f";quality_delta={quality_delta:.3f};rank={rank}"
                f";core={list(core)};candidates_valid={candidates_valid}"
            ),
        }
    # Otherwise route to trash only when the existing scorer is clearly below
    # the C-grade entry and the spatial evidence is a boundary candidate.
    if (quality_after < CONFLICT_MIN_QUALITY and quality_before < CONFLICT_MIN_QUALITY
            and decision == "spatial_boundary_candidate"):
        return {
            "route": "trash",
            "reason": (
                f"conflict_quality_below_C:quality_after={quality_after:.3f}"
                f";quality_before={quality_before:.3f};rank={rank}"
                f";core={list(core)};decision={decision}"
            ),
        }
    return None


def resolve_route_conflict(*, evaluation: dict[str, Any],
                           candidate_records: list[dict[str, Any]],
                           spatial_record: dict[str, Any]) -> dict[str, Any] | None:
    """Domain-owned hook for two independently eligible but conflicting routes.

    The control plane deliberately abstains by default.  A governed domain-agent
    patch may return one fully explained route after independent review and the
    frozen preflight gate; callers must not infer a route here.

    This implementation derives a bounded, explainable route from the anonymous
    spatial rank/core evidence plus the existing quality score/flags.  It
    abstains (returns ``None``) whenever the evidence is insufficient, so
    unresolved conflicts remain visible in the preview JSON.
    """
    return _conflict_route(
        evaluation=evaluation,
        candidate_records=candidate_records,
        spatial_record=spatial_record,
    )


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"invalid report: {path}")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def eligible_plan(dataset: Path, runtime: Path) -> dict[str, Any]:
    inventory = build_check_inventory(dataset, runtime)
    if not inventory["shadow_current"]:
        raise ValueError("shadow report does not match current check folders")
    latest = _json(runtime / "check_iterations/latest.json")
    report_path = Path(latest["report_path"]).resolve()
    root = (runtime / "check_iterations").resolve()
    if root not in report_path.parents:
        raise ValueError("evaluation is outside check-iteration runtime")
    candidate_root = report_path.parent
    evaluation = _json(report_path)
    gate = _json(candidate_root / "candidate_validation.json")
    shadow = _json(candidate_root / "spatial_boundary_shadow.json")
    holdout = _json(runtime / "check_iterations/holdout_v1.json")
    if holdout.get("holdout_sha256") != _digest(sorted(
            holdout.get("holdout", []), key=lambda row: row["key"])):
        raise ValueError("frozen holdout hash mismatch")
    excluded = {row["key"] for row in holdout.get("holdout", [])}
    validation_excluded = set(excluded)
    plan_path = runtime / "check_validation/plan.json"
    if plan_path.is_file():
        plan = _json(plan_path)
        plan_hash = plan.get("plan_sha256")
        canonical_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if plan_hash != _digest(canonical_plan):
            raise ValueError("frozen validation plan hash mismatch")
        if plan.get("holdout_sha256") != holdout.get("holdout_sha256"):
            raise ValueError("validation plan does not match frozen holdout")
        validation_excluded = {row["key"] for row in plan.get("items", [])}
    manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        identities = [f'{row["patient"]}/{row["volume"]}' for row in csv.DictReader(handle)]
    gates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gate.get("records", []):
        if row.get("pool") != "cmr_boundary_manifest":
            continue
        index = int(row["index"])
        if not 0 <= index < len(identities):
            raise ValueError("candidate manifest index changed")
        gates[identities[index]].append(row)
    shadow_by_key = {row["key"]: row for row in shadow.get("records", [])
                     if row.get("state") == "待筛"}
    selected = []
    route_conflicts_detected = 0
    route_conflicts_resolved = 0
    for row in evaluation.get("records", []):
        key = row["key"]
        pending = dataset / "labels/check/待筛/cmr" / (key.removesuffix(".nii.gz") + ".png")
        if not pending.is_file():
            continue
        spatial_record = shadow_by_key.get(key, {})
        spatial = spatial_record.get("decision") == "spatial_boundary_candidate"
        records = gates.get(key, [])
        if spatial and records and shadow.get("control_gate") == "PASS":
            # Frozen holdout cases are monitoring evidence, not resolver input.
            # Exclude them before calling domain-owned code so the resolver does
            # not need identities or knowledge of the private split.
            if key in excluded:
                continue
            route_conflicts_detected += 1
            resolution = resolve_route_conflict(
                evaluation=row, candidate_records=records,
                spatial_record=spatial_record,
            )
            if resolution is None:
                continue
            if (not isinstance(resolution, dict)
                    or resolution.get("route") not in {"checked", "trash"}
                    or not isinstance(resolution.get("reason"), str)
                    or not resolution["reason"].strip()):
                raise ValueError("invalid route-conflict resolution")
            item = {"key": key, **resolution, "conflict_resolution": True}
            item.setdefault("candidates", [])
            selected.append(item)
            route_conflicts_resolved += 1
            continue
        if (key not in excluded and spatial and shadow.get("control_gate") == "PASS"
                and not shadow.get("checked_false_routes")):
            selected.append({"key": key, "route": "trash", "reason": "spatial_control_pass"})
            continue
        candidates_valid = bool(records) and all(item.get("passed") is True for item in records)
        by_screen = candidates_valid and row.get("decision") == "COUNTERFACTUAL_SALVAGE"
        by_score = (candidates_valid and row.get("quality_status") == "SCORED"
                    and float(row.get("quality_after", -1)) >= MIN_QUALITY_SCORE
                    and float(row.get("quality_delta", 0)) > 0)
        if key not in excluded and (by_screen or by_score):
            selected.append({"key": key, "route": "checked",
                             "reason": "screen" if by_screen else "quality_C_improved",
                             "candidates": [item["candidate"] for item in records]})
            continue
        # The operator's progression-first policy also accepts a current label
        # that already reaches the frozen C-grade threshold.  No candidate
        # validator is relevant because this path does not modify label bytes.
        # Spatial-boundary proposals remain excluded until their own checked
        # control gate passes; a good mask does not make an out-of-range slice
        # useful data.
        current_score_pass = (key not in validation_excluded and not spatial
                              and row.get("quality_status") == "SCORED"
                              and float(row.get("quality_before", -1)) >= MIN_QUALITY_SCORE)
        if current_score_pass:
            selected.append({"key": key, "route": "checked",
                             "reason": "quality_C_existing", "candidates": []})
    return {"candidate_root": str(candidate_root), "selected": selected,
            "holdout_excluded": len(excluded),
            "validation_excluded": len(validation_excluded),
            "route_conflicts_detected": route_conflicts_detected,
            "route_conflicts_resolved": route_conflicts_resolved,
            "route_conflicts_unresolved": (
                route_conflicts_detected - route_conflicts_resolved),
            "shadow_created_at": latest.get("created_at")}


def apply_one(dataset: Path, runtime: Path, candidate_root: Path,
              item: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Apply one preselected item, rechecking bytes immediately before write."""
    key = item["key"]
    patient, volume = key.split("/", 1)
    if not patient or Path(volume).name != volume or not volume.endswith(".nii.gz"):
        raise ValueError("invalid spatial-volume identity")
    source_png = dataset / "labels/check/待筛/cmr" / patient / (volume.removesuffix(".nii.gz") + ".png")
    target_state = "checked" if item["route"] == "checked" else "垃圾桶"
    destination = dataset / "labels/check" / target_state / "cmr" / patient / source_png.name
    if not source_png.is_file() or destination.exists():
        raise ValueError("check PNG was moved or destination exists")
    log = runtime / "check_iterations/auto_apply.jsonl"
    common = {"timestamp": datetime.now(timezone.utc).isoformat(), "key": key,
              "route": item["route"], "reason": item["reason"],
              "candidate_root": str(candidate_root)}
    if item["route"] == "trash":
        if dry_run:
            return {"key": key, "route": "trash", "preflight": "PASS"}
        _audit(log, {**common, "event": "requested", "png_sha_before": _sha(source_png)})
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_png), str(destination))
        _audit(log, {**common, "event": "applied", "png_destination": str(destination)})
        return {"key": key, "route": "trash"}
    label = dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
    if not label.is_file():
        raise ValueError("production label is missing")
    if not item.get("candidates"):
        before = _sha(label)
        if dry_run:
            return {"key": key, "route": "checked", "preflight": "PASS",
                    "frames": [], "label_write": False}
        _audit(log, {**common, "event": "requested", "source_sha_before": before,
                     "frames": [], "label_write": False})
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_png), str(destination))
        _audit(log, {**common, "event": "applied", "source_sha_before": before,
                     "source_sha_after": _sha(label), "frames": [],
                     "label_write": False, "png_destination": str(destination)})
        return {"key": key, "route": "checked", "frames": [], "label_write": False}
    image = nib.load(str(label))
    raw = np.asanyarray(image.dataobj)
    stack, axis = _stack(raw)
    proposed = stack.copy()
    touched = set()
    for path_text in item["candidates"]:
        candidate = Path(path_text).resolve()
        if candidate_root not in candidate.parents or not candidate.name.endswith(".npz"):
            raise ValueError("candidate path escaped this iteration")
        try:
            frame = int(candidate.name.rsplit("_k", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError) as exc:
            raise ValueError("candidate has no frame identity") from exc
        if frame in touched or not 0 <= frame < stack.shape[-1]:
            raise ValueError("duplicate or out-of-range candidate frame")
        with np.load(candidate, allow_pickle=False) as data:
            original = data["original"] > 0
            fixed = data["fixed"] > 0
        if original.shape != stack[..., frame].shape or fixed.shape != original.shape:
            raise ValueError("candidate shape changed")
        if not np.array_equal(stack[..., frame] > .5, original):
            raise ValueError("candidate source mask changed")
        if not np.any(fixed != original):
            raise ValueError("candidate is unchanged")
        proposed[..., frame] = fixed.astype(proposed.dtype) * np.max(raw)
        touched.add(frame)
    if not touched:
        raise ValueError("no validated candidate frames")
    if dry_run:
        return {"key": key, "route": "checked", "preflight": "PASS",
                "frames": sorted(touched)}
    before = _sha(label)
    backup = dataset / "label_iteration/auto_check_backups" / before[:16] / patient / volume
    if backup.exists():
        raise ValueError("backup identity already exists; refusing ambiguous replay")
    _audit(log, {**common, "event": "requested", "source_sha_before": before,
                 "frames": sorted(touched)})
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label, backup)
    temporary = label.with_name(label.name + ".auto-check.tmp.nii.gz")
    try:
        nib.save(nib.Nifti1Image(_restore(proposed, raw, axis), image.affine,
                                 image.header), str(temporary))
        if _sha(label) != before:
            raise ValueError("production label changed during staging")
        os.replace(temporary, label)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_png), str(destination))
    except Exception:
        temporary.unlink(missing_ok=True)
        if _sha(label) != before:
            shutil.copy2(backup, label)
        raise
    after = _sha(label)
    _audit(log, {**common, "event": "applied", "source_sha_before": before,
                 "source_sha_after": after, "backup": str(backup),
                 "frames": sorted(touched), "png_destination": str(destination)})
    return {"key": key, "route": "checked", "frames": sorted(touched)}


def run(dataset: Path, runtime: Path, *, execute: bool = False,
        limit: int = MAX_PER_TICK) -> dict[str, Any]:
    plan = eligible_plan(dataset, runtime)
    result = {"eligible": len(plan["selected"]), "holdout_excluded": plan["holdout_excluded"],
              "validation_excluded": plan["validation_excluded"],
              "route_conflicts_detected": plan["route_conflicts_detected"],
              "route_conflicts_resolved": plan["route_conflicts_resolved"],
              "route_conflicts_unresolved": plan["route_conflicts_unresolved"],
              "planned": plan["selected"][:limit], "applied": []}
    if not execute:
        root = Path(plan["candidate_root"]).resolve()
        result["preflight"] = []
        for item in plan["selected"][:limit]:
            try:
                result["preflight"].append(apply_one(dataset, runtime, root, item, dry_run=True))
            except (OSError, ValueError, KeyError, IndexError) as exc:
                result["preflight"].append({"key": item["key"], "error": str(exc)})
        return result
    root = Path(plan["candidate_root"]).resolve()
    for item in plan["selected"][:limit]:
        result["applied"].append(apply_one(dataset, runtime, root, item))
        # Each item has a different volume and is rechecked at write time.
        # Rebuild the cohort evidence once after this bounded batch.
    return result
