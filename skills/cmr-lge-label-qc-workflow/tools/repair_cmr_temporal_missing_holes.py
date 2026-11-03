#!/usr/bin/env python3
"""Repair short runs of missing CMR blood-pool holes across cine time.

Only volumes present in the current middle-Issue manifest are scanned. A run
of 1--3 solid single-component masks is repaired when valid ring frames on
both sides contain nearly identical holes. The conservative intersection of
the two supporting holes is removed from each affected frame. Every edit is
backed up and logged.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def as_stack(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[..., None], None
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1), axis


def restore(stack, original, axis):
    out = stack[..., 0] if axis is None else np.moveaxis(stack, -1, axis)
    return out.reshape(original.shape) if out.size == original.size else out


def topology(mask):
    structure = np.ones((3, 3), dtype=np.uint8)
    components = int(ndi.label(mask, structure=structure)[1])
    holes = ndi.binary_fill_holes(mask) & ~mask
    labels, n = ndi.label(holes, structure=structure)
    if n:
        sizes = [(labels == i).sum() for i in range(1, n + 1)]
        hole = labels == (int(np.argmax(sizes)) + 1)
    else:
        hole = np.zeros_like(mask, dtype=bool)
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return components, int(n), border, hole


def dice(a, b):
    denom = int(a.sum() + b.sum())
    return 2.0 * int((a & b).sum()) / denom if denom else 0.0


def candidate_runs(stack, max_run, min_hole_dice, min_hole_px):
    stats = [topology(stack[..., k]) for k in range(stack.shape[-1])]
    solid = [c == 1 and h == 0 and not b for c, h, b, _ in stats]
    runs, k = [], 0
    while k < len(solid):
        if not solid[k]:
            k += 1
            continue
        start = k
        while k + 1 < len(solid) and solid[k + 1]:
            k += 1
        end = k
        k += 1
        if end - start + 1 > max_run or start == 0 or end == len(solid) - 1:
            continue
        left = stats[start - 1]
        right = stats[end + 1]
        if not (left[0] == left[1] == 1 and not left[2] and right[0] == right[1] == 1 and not right[2]):
            continue
        overlap = dice(left[3], right[3])
        proposal = left[3] & right[3]
        if overlap < min_hole_dice or int(proposal.sum()) < min_hole_px:
            continue
        valid = True
        for frame in range(start, end + 1):
            repaired = stack[..., frame] & ~proposal
            c, h, border, _ = topology(repaired)
            if c != 1 or h != 1 or border:
                valid = False
                break
        if valid:
            runs.append((start, end, proposal, overlap))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--max-run", type=int, default=3)
    ap.add_argument("--min-hole-dice", type=float, default=0.85)
    ap.add_argument("--min-hole-px", type=int, default=30)
    ap.add_argument("--patient")
    ap.add_argument("--volume")
    args = ap.parse_args()

    root = args.root.resolve()
    manifest = args.manifest or root / "labels/check/cmr_middle_issue_slices/issue_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    volumes = sorted({(r["patient"], r["volume"]) for r in rows})
    if args.patient:
        volumes = [x for x in volumes if x[0] == args.patient]
    if args.volume:
        volumes = [x for x in volumes if x[1] == args.volume]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_temporal_hole_runs" / stamp
    backup_root = run / "backups"
    decisions = []

    for patient, volume in volumes:
        path = root / "labels/cmr/cmr_Myo_mask" / patient / volume
        if not path.exists():
            continue
        nii = nib.load(str(path))
        raw = np.asarray(nii.get_fdata())
        stack, axis = as_stack(raw > 0.5)
        runs = candidate_runs(stack, args.max_run, args.min_hole_dice, args.min_hole_px)
        if not runs:
            continue
        updated = stack.copy()
        for start, end, proposal, overlap in runs:
            before = [int(updated[..., k].sum()) for k in range(start, end + 1)]
            for k in range(start, end + 1):
                updated[..., k] &= ~proposal
            after = [int(updated[..., k].sum()) for k in range(start, end + 1)]
            decisions.append({
                "patient": patient, "volume": volume,
                "frames": f"{start + 1}-{end + 1}",
                "hole_pixels": int(proposal.sum()), "support_dice": round(overlap, 4),
                "before_pixels": before, "after_pixels": after,
                "action": "restore_temporally_supported_blood_pool_hole",
                "status": "applied" if args.mode == "apply" else "candidate",
            })
        if args.mode == "apply":
            backup = backup_root / patient / volume
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            restored = restore(updated.astype(np.uint8), raw, axis)
            out = nib.Nifti1Image(restored, nii.affine, nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(path))

    run.mkdir(parents=True, exist_ok=True)
    with (run / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    summary = {
        "root": str(root), "mode": args.mode, "scanned_volumes": len(volumes),
        "candidate_runs": len(decisions),
        "candidate_volumes": len({(d["patient"], d["volume"]) for d in decisions}),
        "candidate_frames": sum(int(d["frames"].split("-")[1]) - int(d["frames"].split("-")[0]) + 1 for d in decisions),
        "rule": "1-3 zero-hole frames bracketed by single-ring frames with hole Dice >= 0.85",
        "run": str(run),
    }
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
