#!/usr/bin/env python3
"""Unify CMR myocardial position across frames using LGE and neighboring frames."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from cmr_qc import as_stack, components, load, render, restore_stack


def num(path: Path) -> float:
    try:
        return float(path.name[:-7])
    except ValueError:
        return float("nan")


def center(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.where(mask)
    return (float(x.mean()), float(y.mean())) if len(x) else (float("nan"), float("nan"))


def nearest_component(mask: np.ndarray, reference: tuple[float, float], min_area: int = 50):
    lab, cs = components(mask)
    choices = []
    for cid, area, cx, cy in cs:
        if area < min_area:
            continue
        distance = float(((cx - reference[0]) ** 2 + (cy - reference[1]) ** 2) ** 0.5)
        choices.append((distance, cid, area))
    return min(choices) if choices else None


def clean_small_holes(mask: np.ndarray, max_hole: int = 120) -> tuple[np.ndarray, int]:
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    lab, n = components(holes)
    out = mask.copy()
    filled_pixels = 0
    for cid, area, _, _ in components(holes)[1]:
        if area <= max_hole:
            out[lab == cid] = True
            filled_pixels += area
    return out, filled_pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-reference-distance", type=float, default=45.0)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_temporal_lge_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_temporal_lge_backups" / stamp
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    records, changed_volumes, changed_slices = [], 0, 0

    for patient_dir in sorted((root / "images/cmr").iterdir()):
        if not patient_dir.is_dir():
            continue
        lge_dir = root / "labels/lge_checked/lge_Myo_labels" / patient_dir.name
        lge_files = sorted(lge_dir.glob("*.nii.gz"), key=num) if lge_dir.exists() else []
        if not lge_files:
            continue
        for image_path in sorted(patient_dir.glob("*.nii.gz")):
            label_path = root / "labels/cmr/cmr_Myo_mask" / patient_dir.name / image_path.name
            if not label_path.exists():
                continue
            lge_path = min(lge_files, key=lambda p: abs(num(p) - num(image_path)))
            _, image_raw = load(image_path)
            label_nii, label_raw = load(label_path)
            _, lge_raw = load(lge_path)
            image_stack, _ = as_stack(image_raw)
            cmr_stack, label_axis = as_stack(label_raw)
            lge = np.squeeze(lge_raw).astype(bool)
            if lge.ndim != 2:
                continue
            # LGE and CMR can have different in-plane resolutions.
            yy, xx = np.where(lge)
            if len(xx) == 0:
                continue
            lge_ref = (float(xx.mean() / lge.shape[1] * cmr_stack.shape[1]),
                       float(yy.mean() / lge.shape[0] * cmr_stack.shape[0]))
            first = nearest_component(cmr_stack[..., 0].astype(bool), lge_ref)
            if first is None or first[0] > args.max_reference_distance:
                continue
            reference = center((components(cmr_stack[..., 0].astype(bool))[0] == first[1]))
            proposed = cmr_stack.astype(bool).copy()
            actions = []
            selected = {}
            for k in range(cmr_stack.shape[-1]):
                current = cmr_stack[..., k].astype(bool)
                choice = nearest_component(current, reference)
                if choice is not None and choice[0] <= args.max_reference_distance:
                    lab, _ = components(current)
                    selected[k] = lab == choice[1]
            for k in range(cmr_stack.shape[-1]):
                current = cmr_stack[..., k].astype(bool)
                target = selected.get(k)
                source = "component"
                if target is None:
                    # Fill an ambiguous frame from the nearest temporally
                    # supported frame, as in the reviewed DAI example.
                    neighbors = sorted((abs(k - j), j) for j in selected if j != k)
                    if not neighbors or neighbors[0][0] > 2:
                        continue
                    target = selected[neighbors[0][1]].copy()
                    source = f"neighbor_frame_{neighbors[0][1] + 1}"
                target, filled = clean_small_holes(target)
                if not np.array_equal(target, current) or filled:
                    actions.append({"slice_index": k + 1, "source": source,
                                    "before_pixels": int(current.sum()), "after_pixels": int(target.sum()),
                                    "filled_hole_pixels": int(filled),
                                    "reference_distance": round(float(nearest_component(current, reference)[0]) if nearest_component(current, reference) else -1, 2)})
                    proposed[..., k] = target
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
            restored = restore_stack(proposed.astype(np.uint8), label_raw, label_axis)
            out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(label_path))
            if args.render:
                for action in actions:
                    k = action["slice_index"] - 1
                    check = issue_root / f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}_temporal_lge.png"
                    render(image_stack[..., k], proposed[..., k], check,
                           f"{patient_dir.name} | {image_path.name} | s{k + 1} | temporal/LGE reference")
            row.update(status="applied", backup=str(backup))
            changed_volumes += 1
            changed_slices += len(actions)

    (run / "matches.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "cmr_volume", "lge_reference", "slices", "status", "backup"])
        writer.writeheader()
        for row in records:
            writer.writerow({"patient": row["patient"], "cmr_volume": row["cmr_volume"], "lge_reference": row["lge_reference"],
                             "slices": ",".join(str(a["slice_index"]) for a in row["actions"]),
                             "status": row["status"], "backup": row.get("backup", "")})
    summary = {"mode": args.mode, "candidate_volumes": len(records), "candidate_slices": sum(len(r["actions"]) for r in records),
               "changed_volumes": changed_volumes, "changed_slices": changed_slices, "run": str(run),
               "rules": ["first CMR frame must agree with nearest single-component LGE reference", "unified component position across frames", "fill only small internal holes <=120 pixels", "ambiguous frames require a neighboring supported frame"]}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
