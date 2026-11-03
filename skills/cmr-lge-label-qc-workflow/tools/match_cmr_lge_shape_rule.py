#!/usr/bin/env python3
"""Batch-match the reviewed CMR false-bridge/shape-overflow rule to all volumes."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, zoom

from cmr_qc import as_stack, components, load, render, restore_stack


def numeric_stem(path: Path) -> float:
    try:
        return float(path.name[:-7])
    except ValueError:
        return float("nan")


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = zoom(mask.astype(np.uint8), (shape[0] / mask.shape[0], shape[1] / mask.shape[1]), order=0)
    if out.shape != shape:
        raise ValueError(f"cannot resize LGE mask {mask.shape} to CMR shape {shape}")
    return out.astype(bool)


def center(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.where(mask)
    return (float(xx.mean()), float(yy.mean())) if len(xx) else (float("nan"), float("nan"))


def candidate_for_slice(mask: np.ndarray, lge: np.ndarray, dilation: int,
                        min_coverage: float, min_extra: float, min_extra_pixels: int) -> dict | None:
    if not mask.any() or not lge.any():
        return None
    _, lge_cs = components(lge)
    if len(lge_cs) != 1:
        return None
    reference = binary_dilation(lge, iterations=dilation)
    refined = mask & reference
    if not refined.any():
        return None
    coverage = float(np.logical_and(mask, lge).sum() / max(int(lge.sum()), 1))
    extra_pixels = int(mask.sum() - refined.sum())
    extra_fraction = float(extra_pixels / max(int(mask.sum()), 1))
    _, refined_cs = components(refined)
    if coverage < min_coverage or extra_fraction < min_extra or extra_pixels < min_extra_pixels:
        return None
    if len(refined_cs) != 1:
        return None
    cx, cy = center(mask)
    rx, ry = center(refined)
    return {"coverage": round(coverage, 5), "extra_fraction": round(extra_fraction, 5),
            "extra_pixels": extra_pixels, "before_pixels": int(mask.sum()),
            "after_pixels": int(refined.sum()), "before_centroid": [round(cx, 2), round(cy, 2)],
            "after_centroid": [round(rx, 2), round(ry, 2)], "refined": refined}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dilation", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.45)
    parser.add_argument("--min-extra-fraction", type=float, default=0.20)
    parser.add_argument("--min-extra-pixels", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_lge_shape_matching_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_lge_shape_matching_backups" / stamp
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    records = []
    changed = 0
    for patient_dir in sorted((root / "images/cmr").iterdir()):
        if not patient_dir.is_dir():
            continue
        lge_dir = root / "labels/lge_checked/lge_Myo_labels" / patient_dir.name
        lge_files = sorted(lge_dir.glob("*.nii.gz"), key=numeric_stem) if lge_dir.exists() else []
        for image_path in sorted(patient_dir.glob("*.nii.gz")):
            label_path = root / "labels/cmr/cmr_Myo_mask" / patient_dir.name / image_path.name
            if not label_path.exists() or not lge_files:
                continue
            lge_path = min(lge_files, key=lambda p: abs(numeric_stem(p) - numeric_stem(image_path)))
            _, image_raw = load(image_path)
            label_nii, label_raw = load(label_path)
            _, lge_raw = load(lge_path)
            image_stack, _ = as_stack(image_raw)
            cmr_stack, label_axis = as_stack(label_raw)
            lge = np.squeeze(lge_raw).astype(bool)
            if lge.ndim != 2:
                continue
            lge = resize_mask(lge, cmr_stack.shape[:2])
            actions = []
            for k in range(cmr_stack.shape[-1]):
                result = candidate_for_slice(cmr_stack[..., k].astype(bool), lge, args.dilation,
                                             args.min_coverage, args.min_extra_fraction, args.min_extra_pixels)
                if result is None:
                    continue
                # Require continuity of the proposed LGE-shaped result in at
                # least one neighboring frame when a neighbor exists.
                supported = False
                for j in (k - 1, k + 1):
                    if 0 <= j < cmr_stack.shape[-1]:
                        neighbor = candidate_for_slice(cmr_stack[..., j].astype(bool), lge, args.dilation,
                                                        0.0, 0.0, 0)
                        if neighbor is not None:
                            a, b = result["after_centroid"], neighbor["after_centroid"]
                            dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                            ratio = max(result["after_pixels"], neighbor["after_pixels"]) / max(min(result["after_pixels"], neighbor["after_pixels"]), 1)
                            if dist <= 30 and ratio <= 3:
                                supported = True
                if not supported and cmr_stack.shape[-1] > 1:
                    continue
                result.pop("refined", None)
                result["slice_index"] = k + 1
                actions.append(result)
            if not actions:
                continue
            row = {"patient": patient_dir.name, "cmr_volume": image_path.name,
                   "lge_reference": lge_path.name, "actions": actions, "status": "candidate"}
            records.append(row)
            if args.mode != "apply":
                continue
            backup = backup_root / label_path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, backup)
            new_stack = cmr_stack.astype(bool).copy()
            for action in actions:
                k = action["slice_index"] - 1
                result = candidate_for_slice(cmr_stack[..., k].astype(bool), lge, args.dilation,
                                             args.min_coverage, args.min_extra_fraction, args.min_extra_pixels)
                new_stack[..., k] = result["refined"]
                if args.render:
                    check = issue_root / f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}_lge_shape_batch.png"
                    render(image_stack[..., k], new_stack[..., k], check,
                           f"{patient_dir.name} | {image_path.name} | s{k + 1} | LGE shape batch")
            restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
            out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(label_path))
            row.update(status="applied", backup=str(backup))
            changed += 1
    (run / "matches.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "cmr_volume", "lge_reference", "slices", "status", "backup"])
        writer.writeheader()
        for row in records:
            writer.writerow({"patient": row["patient"], "cmr_volume": row["cmr_volume"],
                             "lge_reference": row["lge_reference"],
                             "slices": ",".join(str(x["slice_index"]) for x in row["actions"]),
                             "status": row["status"], "backup": row.get("backup", "")})
    summary = {"mode": args.mode, "candidate_volumes": len(records), "changed_volumes": changed,
               "candidate_slices": sum(len(x["actions"]) for x in records), "run": str(run),
               "rule": "LGE-shaped CMR refinement with adjacent-frame support"}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
