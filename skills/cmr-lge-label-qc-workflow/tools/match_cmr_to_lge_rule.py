#!/usr/bin/env python3
"""Use a geometrically matched single-component LGE Myo mask as CMR QC evidence."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from cmr_qc import as_stack, components, load, render, restore_stack


def numeric_stem(path: Path) -> float:
    try:
        return float(path.name[:-7])
    except ValueError:
        return float("nan")


def centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.where(mask)
    return (float(xx.mean() / mask.shape[1]), float(yy.mean() / mask.shape[0])) if len(xx) else (float("nan"), float("nan"))


def match_volume(cmr_stack: np.ndarray, lge_mask: np.ndarray) -> tuple[list[dict], str]:
    _, lge_components = components(lge_mask)
    if len(lge_components) != 1:
        return [], f"lge_components={len(lge_components)}"
    lge_center = centroid(lge_mask)
    actions = []
    for k in range(cmr_stack.shape[-1]):
        mask = cmr_stack[..., k].astype(bool)
        _, cs = components(mask)
        if len(cs) <= 1:
            continue
        candidates = []
        for cid, area, cx, cy in cs:
            center = (cx / mask.shape[1], cy / mask.shape[0])
            distance = float(((center[0] - lge_center[0]) ** 2 + (center[1] - lge_center[1]) ** 2) ** 0.5)
            candidates.append((distance, cid, area, center))
        distance, keep_id, keep_area, keep_center = min(candidates)
        # The 0.15 normalized-image threshold is deliberately conservative;
        # unresolved geometry remains a manual candidate.
        if distance > 0.15 or keep_area < 50:
            continue
        actions.append({"slice_index": k + 1, "component_areas": [x[1] for x in cs],
                        "kept_component": int(keep_id), "kept_area": int(keep_area),
                        "lge_centroid": [round(x, 5) for x in lge_center],
                        "cmr_centroid": [round(x, 5) for x in keep_center],
                        "normalized_distance": round(distance, 5)})
    return actions, f"lge_centroid={lge_center[0]:.3f},{lge_center[1]:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_lge_reference_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_lge_reference_backups" / stamp
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
            lge_mask = np.squeeze(lge_raw).astype(bool)
            if lge_mask.ndim != 2:
                continue
            actions, reason = match_volume(cmr_stack, lge_mask)
            if not actions:
                continue
            row = {"patient": patient_dir.name, "cmr_volume": image_path.name,
                   "lge_reference": lge_path.name, "reason": reason,
                   "actions": actions, "status": "candidate"}
            records.append(row)
            if args.mode != "apply":
                continue
            backup = backup_root / label_path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, backup)
            new_stack = cmr_stack.astype(bool).copy()
            for action in actions:
                k = action["slice_index"] - 1
                mask = cmr_stack[..., k].astype(bool)
                lab, _ = components(mask)
                new_stack[..., k] = lab == action["kept_component"]
                if args.render:
                    check = issue_root / f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}_lge_ref.png"
                    render(image_stack[..., k], new_stack[..., k], check,
                           f"{patient_dir.name} | {image_path.name} | s{k + 1} | LGE ref {lge_path.name}")
            restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
            out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(label_path))
            row.update(status="applied", backup=str(backup))
            changed += 1
    (run / "matches.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "cmr_volume", "lge_reference", "reason", "slices", "status", "backup"])
        writer.writeheader()
        for row in records:
            writer.writerow({"patient": row["patient"], "cmr_volume": row["cmr_volume"],
                             "lge_reference": row["lge_reference"], "reason": row["reason"],
                             "slices": ",".join(str(x["slice_index"]) for x in row["actions"]),
                             "status": row["status"], "backup": row.get("backup", "")})
    summary = {"mode": args.mode, "candidate_volumes": len(records), "changed_volumes": changed,
               "run": str(run), "rule": "single-component, nearest-position LGE Myo reference; normalized centroid distance <=0.15"}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
