#!/usr/bin/env python3
"""Repair short CMR cine runs with an open myocardial ring.

The repair is deliberately conservative: a 1--3 frame run must contain one
connected myocardium component but no enclosed blood-pool hole, and must be
bracketed by two closed single-ring frames whose masks and holes agree.  Only
the intersection of the two supporting myocardium masks is added.  The result
must be one connected component with one enclosed hole and may add at most a
configured fraction of the mean supporting-mask area.
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


STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def as_stack(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[..., None], None
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1), axis


def restore(stack, original, axis):
    out = stack[..., 0] if axis is None else np.moveaxis(stack, -1, axis)
    return out.reshape(original.shape)


def topology(mask):
    components = int(ndi.label(mask, structure=STRUCTURE)[1])
    holes = ndi.binary_fill_holes(mask) & ~mask
    labels, count = ndi.label(holes, structure=STRUCTURE)
    if count:
        sizes = ndi.sum(holes, labels, range(1, count + 1))
        main_hole = labels == (int(np.argmax(sizes)) + 1)
    else:
        main_hole = np.zeros_like(mask, dtype=bool)
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return components, int(count), border, main_hole


def dice(a, b):
    denominator = int(a.sum() + b.sum())
    return 2.0 * int((a & b).sum()) / denominator if denominator else 0.0


def candidate_runs(stack, max_run=3, min_hole_dice=.85, min_mask_dice=.70,
                   max_added_fraction=.35):
    stats = [topology(stack[..., k]) for k in range(stack.shape[-1])]
    open_ring = [c == 1 and h == 0 and not border for c, h, border, _ in stats]
    result = []
    k = 0
    while k < len(open_ring):
        if not open_ring[k]:
            k += 1
            continue
        start = k
        while k + 1 < len(open_ring) and open_ring[k + 1]:
            k += 1
        end = k
        k += 1
        if end - start + 1 > max_run or start == 0 or end == stack.shape[-1] - 1:
            continue
        left, right = start - 1, end + 1
        ls, rs = stats[left], stats[right]
        if not (ls[0] == ls[1] == 1 and not ls[2] and rs[0] == rs[1] == 1 and not rs[2]):
            continue
        hole_score = dice(ls[3], rs[3])
        mask_score = dice(stack[..., left], stack[..., right])
        if hole_score < min_hole_dice or mask_score < min_mask_dice:
            continue
        support = stack[..., left] & stack[..., right]
        support_area = (int(stack[..., left].sum()) + int(stack[..., right].sum())) / 2.0
        proposals = []
        valid = True
        for frame in range(start, end + 1):
            repaired = stack[..., frame] | support
            added = int((repaired & ~stack[..., frame]).sum())
            c, h, border, _ = topology(repaired)
            if (added < 3 or added / max(support_area, 1.0) > max_added_fraction
                    or c != 1 or h != 1 or border):
                valid = False
                break
            proposals.append((frame, repaired, added))
        if valid:
            result.append((start, end, proposals, hole_score, mask_score))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    ap.add_argument("--patient")
    ap.add_argument("--volume")
    ap.add_argument("--manifest", type=Path,
                    help="restrict scanning to patient/volume pairs in a QC manifest; defaults to the current middle-Issue manifest when present")
    ap.add_argument("--max-run", type=int, default=3)
    args = ap.parse_args()
    root = args.root.resolve()
    label_root = root / "labels/cmr/cmr_Myo_mask"
    manifest = args.manifest or root / "labels/check/cmr_middle_issue_slices/issue_manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            keys = {(row["patient"], row["volume"]) for row in csv.DictReader(handle)}
        paths = [label_root / patient / volume for patient, volume in sorted(keys)]
        paths = [path for path in paths if path.exists()]
    else:
        paths = sorted(label_root.glob("*/*.nii.gz"))
    if args.patient:
        paths = [p for p in paths if p.parent.name == args.patient]
    if args.volume:
        paths = [p for p in paths if p.name == args.volume]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_temporal_open_ring_runs" / stamp
    records = []
    for path in paths:
        nii = nib.load(str(path))
        raw = np.asarray(nii.get_fdata())
        stack, axis = as_stack(raw > .5)
        runs = candidate_runs(stack, max_run=args.max_run)
        if not runs:
            continue
        updated = stack.copy()
        actions = []
        for start, end, proposals, hole_score, mask_score in runs:
            for frame, repaired, added in proposals:
                updated[..., frame] = repaired
                actions.append({"frame": frame + 1, "added_pixels": added})
            records.append({
                "patient": path.parent.name, "volume": path.name,
                "frames": f"{start + 1}-{end + 1}", "actions": actions[-len(proposals):],
                "support_hole_dice": round(hole_score, 4),
                "support_mask_dice": round(mask_score, 4),
                "status": "applied" if args.mode == "apply" else "candidate",
            })
        if args.mode == "apply":
            backup = run / "backups" / path.parent.name / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            out = nib.Nifti1Image(restore(updated.astype(np.uint8), raw, axis), nii.affine, nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(path))
    run.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode, "scanned_volumes": len(paths), "candidate_runs": len(records),
        "candidate_frames": sum(len(r["actions"]) for r in records), "records": records,
        "rule": "1-3 open-ring frames + two-sided closed-ring agreement; add support-mask intersection",
        "source_manifest": str(manifest) if manifest.exists() else "full_label_tree",
        "run": str(run),
    }
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
