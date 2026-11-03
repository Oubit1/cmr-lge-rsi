#!/usr/bin/env python3
"""Apply an explicitly reviewed, volume-scoped CMR case rule with backup."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from cmr_qc import as_stack, components, load, render, restore_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--volume", required=True, help="e.g. 60.6223.nii.gz")
    parser.add_argument("--rule", choices=["keep-largest-component"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    image_path = root / "images/cmr" / args.patient / args.volume
    label_path = root / "labels/cmr/cmr_Myo_mask" / args.patient / args.volume
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"missing image or label: {image_path} / {label_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "label_iteration/cmr_case_backups" / stamp / label_path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, backup)
    _, image_raw = load(image_path)
    label_nii, label_raw = load(label_path)
    image_stack, _ = as_stack(image_raw)
    label_stack, label_axis = as_stack(label_raw)
    new_stack = label_stack.astype(bool).copy()
    changed = []
    for k in range(label_stack.shape[-1]):
        mask = label_stack[..., k].astype(bool)
        _, cs = components(mask)
        areas = [area for _, area, _, _ in cs]
        if len(areas) <= 1:
            continue
        keep_id = max(cs, key=lambda x: x[1])[0]
        lab, _ = components(mask)
        new = lab == keep_id
        removed = int(mask.sum() - new.sum())
        new_stack[..., k] = new
        changed.append({"slice_index": k + 1, "component_areas": areas,
                        "kept_area": int(new.sum()), "removed_pixels": removed})

        check_dir = root / "labels/check/cmr_issue_slices"
        check_dir.mkdir(parents=True, exist_ok=True)
        check = check_dir / f"{args.patient}_{args.volume[:-7]}_s{k + 1:02d}_case_largest.png"
        render(image_stack[..., k], new, check,
               f"{args.patient} | {args.volume} | slice {k + 1} | keep-largest-component")

    if not changed:
        print(json.dumps({"status": "no_change", "backup": str(backup)}, ensure_ascii=False))
        return

    restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
    out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(label_path))
    run_dir = root / "label_iteration/cmr_case_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), "modality": "cmr",
              "patient": args.patient, "volume": args.volume,
              "rule": args.rule, "action": "keep_largest_component_per_slice",
              "changed_slices": changed, "backup": str(backup),
              "status": "applied"}
    (run_dir / "case_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
