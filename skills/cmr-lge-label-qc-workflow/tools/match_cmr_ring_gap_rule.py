#!/usr/bin/env python3
"""Detect and conservatively close small CMR myocardial ring-wall gaps."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

from cmr_qc import as_stack, components, load, render, restore_stack


def gap_candidate(mask: np.ndarray, prev: np.ndarray | None, nxt: np.ndarray | None,
                  max_added: int = 100, min_support: float = 0.20):
    if not mask.any() or prev is None or nxt is None:
        return None
    closed = binary_closing(mask, structure=np.ones((5, 5)))
    added = closed & ~mask
    added_n = int(added.sum())
    if added_n < 3 or added_n > max_added:
        return None
    support = binary_dilation(prev, iterations=2) & binary_dilation(nxt, iterations=2)
    support_fraction = float((added & support).sum() / added_n)
    if support_fraction < min_support:
        return None
    _, component_list = components(closed)
    n_components = len(component_list)
    holes = binary_fill_holes(closed) & ~closed
    _, hole_list = components(holes)
    n_holes = len(hole_list)
    if n_components != 1 or n_holes > 1:
        return None
    return {"added_pixels": added_n, "support_fraction": round(support_fraction, 4), "result": closed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--manifest", type=Path, help="restrict screening to patient/volume/slice rows in an existing QC manifest")
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_ring_gap_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_ring_gap_backups" / stamp
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    allowed = None
    if args.manifest:
        with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
            allowed = {(row["patient"], row["volume"], int(row["slice_index"]))
                       for row in csv.DictReader(handle) if row.get("slice_index", "").isdigit()}
    records, changed_volumes, changed_slices = [], 0, 0
    for patient_dir in sorted((root / "images/cmr").iterdir()):
        if not patient_dir.is_dir():
            continue
        for image_path in sorted(patient_dir.glob("*.nii.gz")):
            label_path = root / "labels/cmr/cmr_Myo_mask" / patient_dir.name / image_path.name
            if not label_path.exists():
                continue
            _, image_raw = load(image_path)
            label_nii, label_raw = load(label_path)
            image_stack, _ = as_stack(image_raw)
            mask_stack, label_axis = as_stack(label_raw)
            proposed = mask_stack.astype(bool).copy()
            actions = []
            for k in range(1, mask_stack.shape[-1] - 1):
                if allowed is not None and (patient_dir.name, image_path.name, k + 1) not in allowed:
                    continue
                result = gap_candidate(mask_stack[..., k].astype(bool), mask_stack[..., k - 1].astype(bool), mask_stack[..., k + 1].astype(bool))
                if result is None:
                    continue
                actions.append({"slice_index": k + 1, "added_pixels": result["added_pixels"], "support_fraction": result["support_fraction"]})
                proposed[..., k] = result["result"]
            if not actions:
                continue
            row = {"patient": patient_dir.name, "cmr_volume": image_path.name, "actions": actions, "status": "candidate"}
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
                    check = issue_root / f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}_ring_gap.png"
                    render(image_stack[..., k], proposed[..., k], check,
                           f"{patient_dir.name} | {image_path.name} | s{k + 1} | ring gap closed")
            row.update(status="applied", backup=str(backup))
            changed_volumes += 1
            changed_slices += len(actions)
    (run / "matches.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run / "matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "cmr_volume", "slices", "status", "backup"])
        writer.writeheader()
        for row in records:
            writer.writerow({"patient": row["patient"], "cmr_volume": row["cmr_volume"],
                             "slices": ",".join(str(a["slice_index"]) for a in row["actions"]),
                             "status": row["status"], "backup": row.get("backup", "")})
    summary = {"mode": args.mode, "candidate_volumes": len(records), "candidate_slices": sum(len(r["actions"]) for r in records),
               "changed_volumes": changed_volumes, "changed_slices": changed_slices, "run": str(run),
               "rule": "5x5 ring closing, <=100 pixels, >=20% adjacent-frame support, preserve one central hole"}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
