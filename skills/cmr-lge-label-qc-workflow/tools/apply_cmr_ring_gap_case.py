#!/usr/bin/env python3
"""Apply a manually confirmed small CMR ring-gap closure to selected slices."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing

from cmr_qc import as_stack, load, render, restore_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--slices", required=True, help="comma-separated 1-based slice/frame numbers")
    args = parser.parse_args()
    root = args.root.resolve()
    label_path = root / "labels/cmr/cmr_Myo_mask" / args.patient / args.volume
    image_path = root / "images/cmr" / args.patient / args.volume
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "label_iteration/cmr_ring_gap_case_backups" / stamp / label_path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, backup)
    label_nii, label_raw = load(label_path)
    _, image_raw = load(image_path)
    masks, axis = as_stack(label_raw)
    images, _ = as_stack(image_raw)
    new = masks.astype(bool).copy()
    actions = []
    for value in args.slices.split(","):
        k = int(value) - 1
        before = new[..., k].copy()
        after = binary_closing(before, structure=np.ones((5, 5)))
        added = int((after & ~before).sum())
        if added <= 0 or added > 100:
            raise ValueError(f"slice {k + 1} unexpected closure size: {added}")
        new[..., k] = after
        check = root / "labels/check/cmr_middle_issue_slices" / f"{args.patient}_{args.volume[:-7]}_s{k + 1:02d}_ring_gap_case.png"
        render(images[..., k], after, check, f"{args.patient} | {args.volume} | s{k + 1} | confirmed ring gap")
        actions.append({"slice_index": k + 1, "before_pixels": int(before.sum()), "after_pixels": int(after.sum()), "added_pixels": added, "check": str(check)})
    restored = restore_stack(new.astype(np.uint8), label_raw, axis)
    out = nib.Nifti1Image(restored, label_nii.affine, label_nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(label_path))
    run = root / "label_iteration/cmr_ring_gap_case_runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), "patient": args.patient, "volume": args.volume,
              "rule": "confirmed ring-wall gap; 5x5 closure supported by adjacent frames",
              "actions": actions, "backup": str(backup), "status": "applied"}
    (run / "case_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
