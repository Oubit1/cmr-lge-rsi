#!/usr/bin/env python3
"""Conservatively merge nearby same-territory MI components.

Clinical rationale:
When a transmural or subendocardial infarct is segmented, slight image noise or
sub-pixel intensity drops can artificially split a single contiguous infarct territory
into 2 or 3 fragments separated by a 1-3 pixel gap.

Conservative repair rule:
1. Candidate slice must have multiple MI components.
2. A small 3x3 morphological closing is applied strictly constrained to the Myocardium mask.
3. Added pixels must be small (<= 30 pixels).
4. Slices with distant, clearly separate infarct territories (e.g. LAD + RCA > 5px apart)
   remain separate and are preserved.
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


def load_nii(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    nii = nib.load(str(path))
    arr = np.squeeze(nii.get_fdata())
    return nii, arr, arr > 0.5


def count_components(mask: np.ndarray) -> int:
    _, n = ndi.label(mask, structure=np.ones((3, 3)))
    return n


def run_merge_nearby(root: Path, mode: str, max_added_px: int = 30) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/mi_merge_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/mi_merge_backups" / stamp

    myo_dir = root / "labels/lge_checked/lge_Myo_labels"
    mi_dir = root / "labels/lge_checked/lge_MI_labels"

    candidates = []
    modified_count = 0

    patient_dirs = sorted([p for p in mi_dir.iterdir() if p.is_dir()])

    for pd in patient_dirs:
        patient = pd.name
        for mi_file in sorted(pd.glob("*.nii.gz")):
            myo_file = myo_dir / patient / mi_file.name
            if not mi_file.exists() or not myo_file.exists():
                continue

            mi_nii, _, mi_mask = load_nii(mi_file)
            _, _, myo_mask = load_nii(myo_file)

            before_comps = count_components(mi_mask)
            if before_comps <= 1:
                continue

            # Conservative 3x3 morphological closing clipped to Myo
            closed = ndi.binary_closing(mi_mask, structure=np.ones((3, 3))) & myo_mask
            after_comps = count_components(closed)
            added_pixels = int(closed.sum() - mi_mask.sum())

            # Only accept if components decreased and added pixels is conservative
            if after_comps < before_comps and 0 < added_pixels <= max_added_px:
                cand = {
                    "patient": patient,
                    "slice": mi_file.name,
                    "before_components": before_comps,
                    "after_components": after_comps,
                    "added_pixels": added_pixels,
                    "fully_resolved_to_single": (after_comps == 1)
                }
                candidates.append(cand)

                if mode == "apply-safe":
                    # Backup and save
                    backup = backup_root / mi_file.relative_to(root)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mi_file, backup)

                    out_nii = nib.Nifti1Image(closed.astype(np.uint8), mi_nii.affine, mi_nii.header)
                    nib.save(out_nii, str(mi_file))
                    modified_count += 1

    csv_path = run_dir / "candidates.csv"
    if candidates:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
            w.writeheader()
            w.writerows(candidates)

    fully_resolved = sum(1 for c in candidates if c["fully_resolved_to_single"])
    summary = {
        "root": str(root),
        "mode": mode,
        "run": str(run_dir),
        "candidates": len(candidates),
        "fully_resolved_to_single": fully_resolved,
        "partially_reduced_components": len(candidates) - fully_resolved,
        "modified_files": modified_count,
        "max_added_px": max_added_px
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply-safe"], default="dry-run")
    parser.add_argument("--max-added-px", type=int, default=30)
    args = parser.parse_args()

    summary = run_merge_nearby(args.root.resolve(), args.mode, args.max_added_px)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
