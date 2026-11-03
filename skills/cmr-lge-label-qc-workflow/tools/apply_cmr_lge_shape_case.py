#!/usr/bin/env python3
"""Remove a CMR mask lobe outside a matched LGE Myo reference for a reviewed case."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

from cmr_qc import as_stack, load, render, restore_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--lge-reference", required=True)
    parser.add_argument("--distance", type=float, default=5.0)
    args = parser.parse_args()
    root = args.root.resolve()
    label_path = root / "labels/cmr/cmr_Myo_mask" / args.patient / args.volume
    image_path = root / "images/cmr" / args.patient / args.volume
    lge_path = root / "labels/lge_checked/lge_Myo_labels" / args.patient / args.lge_reference
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "label_iteration/cmr_lge_shape_backups" / stamp / label_path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, backup)
    label_nii, label_raw = load(label_path)
    _, image_raw = load(image_path)
    _, lge_raw = load(lge_path)
    label_stack, label_axis = as_stack(label_raw)
    image_stack, _ = as_stack(image_raw)
    lge = np.squeeze(lge_raw).astype(bool)
    k = args.slice - 1
    if lge.ndim != 2 or k < 0 or k >= label_stack.shape[-1]:
        raise ValueError("invalid LGE shape or CMR slice")
    lge_resized = zoom(lge, (label_stack.shape[0] / lge.shape[0], label_stack.shape[1] / lge.shape[1]), order=0) > 0.5
    if lge_resized.shape != label_stack[..., k].shape:
        raise ValueError(f"LGE/CMR shape mismatch after resize: {lge_resized.shape} vs {label_stack[..., k].shape}")
    mask = label_stack[..., k].astype(bool)
    reference_distance = distance_transform_edt(~lge_resized)
    refined = mask & (reference_distance <= args.distance)
    if refined.sum() == 0 or refined.sum() >= mask.sum():
        raise ValueError(f"shape rule made no useful change: before={int(mask.sum())}, after={int(refined.sum())}")
    new_stack = label_stack.astype(bool).copy()
    new_stack[..., k] = refined
    restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
    out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(label_path))
    check = root / "labels/check/cmr_issue_slices" / f"{args.patient}_{args.volume[:-7]}_s{args.slice:02d}_lge_shape.png"
    render(image_stack[..., k], refined, check, f"{args.patient} | {args.volume} | s{args.slice} | LGE shape reference")
    run = root / "label_iteration/cmr_lge_shape_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), "patient": args.patient, "volume": args.volume,
              "slice_index": args.slice, "lge_reference": args.lge_reference,
              "rule": "keep CMR pixels within distance threshold of resized LGE Myo",
              "distance": args.distance, "before_pixels": int(mask.sum()),
              "after_pixels": int(refined.sum()), "removed_pixels": int(mask.sum() - refined.sum()),
              "backup": str(backup), "check": str(check), "status": "applied"}
    (run / "case_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
