#!/usr/bin/env python3
"""Counterfactual review of gated repairs on pending CMR spatial volumes.

This never writes a label. It reports which pending volumes would satisfy the
existing SALVAGE rule after all validated candidate frames are applied in memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.manual_annotation import _restore, _stack  # noqa: E402
from app.quality_scoring import PairSpec, extract_pair, score_record  # noqa: E402


def features(stack: np.ndarray) -> tuple[float, float]:
    areas = []
    cavities = 0
    for k in range(stack.shape[-1]):
        mask = stack[..., k] > .5
        area = int(mask.sum())
        areas.append(area)
        if area >= 20 and int((ndi.binary_fill_holes(mask) & ~mask).sum()) >= 20:
            cavities += 1
    return float(np.median(areas)), cavities / max(stack.shape[-1], 1)


def compare_quality(image: Path, label: Path, proposed: np.ndarray | None,
                    candidate_root: Path, profile: dict) -> dict:
    """Score the source and, when supplied, an in-memory candidate.

    A pending item does not need a proposed edit before the existing quality
    model can determine that its current label already meets the configured
    threshold.  The function remains read-only in both modes.
    """
    if not image.is_file():
        return {"quality_status": "NO_IMAGE"}
    source = nib.load(str(label))
    original = np.asanyarray(source.dataobj)
    base = {"dataset": "check_shadow", "tier": "quasi_gold", "target": "cmr_myo",
            "image": image, "myo": None, "relative_id": "shadow",
            "case_id": "shadow", "slice_position": 0.0, "reviewed": False}
    before = extract_pair(PairSpec(label=label, **base))
    if not before.get("valid"):
        return {"quality_status": "INVALID_PAIR"}
    scored_before = score_record(before, profile)
    result = {
        "quality_status": "SCORED", "quality_before": scored_before["score"],
        "quality_flags_before": scored_before["flags"],
        "quality_subscores_before": scored_before.get("subscores", {}),
    }
    if proposed is None:
        return {**result, "quality_after": scored_before["score"], "quality_delta": 0.0,
                "quality_flags_after": scored_before["flags"],
                "quality_subscores_after": scored_before.get("subscores", {})}
    _, axis = _stack(original)
    candidate_raw = _restore((proposed > .5).astype(np.uint8), original, axis)
    with tempfile.TemporaryDirectory(dir=candidate_root) as tmp:
        candidate_path = Path(tmp) / "shadow_label.nii.gz"
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(candidate_raw, source.affine, header), str(candidate_path))
        after = extract_pair(PairSpec(label=candidate_path, **base))
        if not after.get("valid"):
            return {"quality_status": "INVALID_PAIR"}
        scored_after = score_record(after, profile)
    return {**result,
        "quality_after": scored_after["score"],
        "quality_delta": round(scored_after["score"] - scored_before["score"], 3),
        "quality_flags_after": scored_after["flags"],
        "quality_subscores_after": scored_after.get("subscores", {}),
    }


def evaluate(dataset: Path, candidate_root: Path, triage_path: Path,
             state: str = "待筛", quality_profile: Path | None = None) -> dict:
    profile = json.loads(quality_profile.read_text(encoding="utf-8")) if quality_profile else None
    if profile is not None and "cmr_myo" not in profile.get("targets", {}):
        raise ValueError("quality profile lacks cmr_myo calibration")
    check = dataset / "labels/check"
    with triage_path.open(encoding="utf-8-sig", newline="") as handle:
        triage = {r["relative_key"]: r for r in csv.DictReader(handle)}
    with (check / "manifest_backup/cmr_boundary_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    gate = json.loads((candidate_root / "candidate_validation.json").read_text())
    by_key = defaultdict(list)
    for rec in gate["records"]:
        if not rec.get("passed") or rec.get("pool") != "cmr_boundary_manifest":
            continue
        index = int(rec["index"])
        row = manifest[index]
        key = f'{row["patient"]}/{row["volume"]}'
        by_key[key].append(rec)

    records = []
    for png in sorted((check / state / "cmr").glob("*/*.png")):
        patient = png.parent.name
        volume = png.stem + ".nii.gz"
        key = f"{patient}/{volume}"
        row = triage.get(key)
        if row is None:
            records.append({"key": key, "decision": "NO_TRIAGE"})
            continue
        ref = float(row["ref_area"])
        label = dataset / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
        if not label.is_file():
            records.append({"key": key, "decision": "NO_LABEL"})
            continue
        stack, _ = _stack(np.asanyarray(nib.load(str(label)).dataobj))
        before_area, before_cavity = features(stack)
        proposed = stack.copy()
        reason = None
        changed_frames = set()
        for rec in by_key.get(key, []):
            path = Path(rec["candidate"])
            k = int(path.stem.rsplit("_k", 1)[1])
            with np.load(path, allow_pickle=False) as candidate:
                original = candidate["original"] > 0
                fixed = candidate["fixed"] > 0
            if k < 0 or k >= stack.shape[-1] or original.shape != stack[..., k].shape:
                reason = "FRAME_OR_SHAPE_MISMATCH"
                break
            if not np.array_equal(stack[..., k] > .5, original):
                reason = "STALE_SOURCE"
                break
            if k in changed_frames and not np.array_equal(proposed[..., k] > .5, fixed):
                reason = "CONFLICTING_CANDIDATES"
                break
            proposed[..., k] = fixed.astype(proposed.dtype) * np.max(stack)
            changed_frames.add(k)
        after_area, after_cavity = features(proposed)
        eligible = bool(changed_frames and not reason and ref > 0
                        and after_area >= .45 * ref and after_cavity >= .60)
        item = {
            "key": key, "decision": reason or ("COUNTERFACTUAL_SALVAGE" if eligible else "REVIEW"),
            "frames": sorted(changed_frames), "reference_area": ref,
            "before_area": before_area, "before_cavity": round(before_cavity, 3),
            "after_area": after_area, "after_cavity": round(after_cavity, 3),
        }
        if profile is not None and not reason:
            image = dataset / "images/cmr" / patient / volume
            item.update(compare_quality(
                image, label, proposed if changed_frames else None, candidate_root, profile
            ))
        records.append(item)
    counts = Counter(r["decision"] for r in records)
    scored = [r["quality_delta"] for r in records if r.get("quality_status") == "SCORED"]
    quality = {
        "status": "SCORED" if profile is not None else "NOT_RUN",
        "scored_volumes": len(scored),
        "improved": sum(delta > 0 for delta in scored),
        "degraded": sum(delta < 0 for delta in scored),
        "unchanged": sum(delta == 0 for delta in scored),
        "unscorable": sum(r.get("quality_status") in {"NO_IMAGE", "INVALID_PAIR"} for r in records),
        "median_delta": round(float(np.median(scored)), 3) if scored else None,
        "min_delta": round(min(scored), 3) if scored else None,
    }
    return {"state": state, "total_volumes": len(records),
            "gate_passed_frames": sum(len(v) for v in by_key.values()),
            "volumes_with_candidates": sum(bool(v) for v in by_key.values()),
            "decisions": dict(counts), "quality": quality, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--triage", type=Path, default=ROOT / "runtime/boundary_triage/slice_triage.csv")
    ap.add_argument("--state", choices=("待筛", "待处理"), default="待筛")
    ap.add_argument("--quality-profile", type=Path,
                    help="Frozen existing calibration profile; scores are evidence, not approval")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(args.dataset, args.candidates, args.triage, args.state,
                      args.quality_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
