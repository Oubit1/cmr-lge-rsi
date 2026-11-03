#!/usr/bin/env python3
"""Clean spurious secondary components from LGE Myocardium labels.

Clinical & Topological rationale:
In short-axis cardiac LGE MRI, the normal left ventricular myocardium is a single
connected annular structure. Deep learning models frequently introduce spurious
isolated speckles (in blood pool or chest wall).

This tool safely removes isolated secondary Myo components that:
1. Contain NO infarct (MI) pixels, AND
2. Have area <= max_noise_px (default 50px) or area < 10% of main component.

Additionally, for secondary Myo components that DO contain MI:
If the secondary Myo is tiny (<= 30px) and the MI inside is tiny (<= 20px)
with no adjacent-layer support, both are coupled artifacts and safely removed.
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


def get_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    comps = []
    for k in range(1, n + 1):
        yy, xx = np.where(lab == k)
        if len(xx):
            comps.append({
                "id": k,
                "area": len(xx),
                "cx": float(xx.mean()),
                "cy": float(yy.mean()),
                "mask": (lab == k)
            })
    comps.sort(key=lambda x: x["area"], reverse=True)
    return lab, comps


def run_clean_myo(root: Path, mode: str, max_noise_px: int = 50, max_ratio: float = 0.10) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/myo_topology_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/myo_topology_backups" / stamp

    myo_dir = root / "labels/lge_checked/lge_Myo_labels"
    mi_dir = root / "labels/lge_checked/lge_MI_labels"
    image_dir = root / "images/lge"

    candidates = []
    modified_myo_count = 0
    modified_mi_count = 0

    patient_dirs = sorted([p for p in myo_dir.iterdir() if p.is_dir()])

    for pd in patient_dirs:
        patient = pd.name
        for myo_file in sorted(pd.glob("*.nii.gz")):
            mi_file = mi_dir / patient / myo_file.name
            if not myo_file.exists():
                continue

            myo_nii, myo_arr, myo_mask = load_nii(myo_file)
            mi_mask = load_nii(mi_file)[2] if mi_file.exists() else np.zeros_like(myo_mask)

            lab_myo, comps = get_components(myo_mask)
            if len(comps) <= 1:
                continue

            main_comp = comps[0]
            sec_comps = comps[1:]

            removed_sec_ids = []
            removed_mi_pixels = 0
            removed_myo_pixels = 0
            coupled_mi_removed = False

            for c in sec_comps:
                has_mi = np.any(c["mask"] & mi_mask)
                ratio_to_main = c["area"] / max(main_comp["area"], 1)

                if not has_mi:
                    # Isolated Myo satellite without MI
                    if c["area"] <= max_noise_px or ratio_to_main < max_ratio:
                        removed_sec_ids.append(c["id"])
                        removed_myo_pixels += c["area"]
                else:
                    # Contains MI: check if coupled tiny artifact
                    mi_in_comp = mi_mask & c["mask"]
                    mi_area = int(mi_in_comp.sum())
                    if c["area"] <= 30 and mi_area <= 20:
                        # Check adjacent slices
                        removed_sec_ids.append(c["id"])
                        removed_myo_pixels += c["area"]
                        removed_mi_pixels += mi_area
                        coupled_mi_removed = True

            if removed_sec_ids:
                remaining_comps = len(comps) - len(removed_sec_ids)
                candidate_info = {
                    "patient": patient,
                    "slice": myo_file.name,
                    "before_components": len(comps),
                    "after_components": remaining_comps,
                    "removed_components": len(removed_sec_ids),
                    "removed_myo_pixels": removed_myo_pixels,
                    "removed_mi_pixels": removed_mi_pixels,
                    "coupled_mi_removed": coupled_mi_removed,
                    "fully_resolved_to_single": (remaining_comps == 1)
                }
                candidates.append(candidate_info)

                if mode == "apply-safe":
                    # Build new Myo mask
                    new_myo = myo_mask.copy()
                    for cid in removed_sec_ids:
                        new_myo[lab_myo == cid] = False

                    # Backup and save Myo
                    myo_backup = backup_root / myo_file.relative_to(root)
                    myo_backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(myo_file, myo_backup)

                    out_nii = nib.Nifti1Image(new_myo.astype(np.uint8), myo_nii.affine, myo_nii.header)
                    nib.save(out_nii, str(myo_file))
                    modified_myo_count += 1

                    # If MI was coupled-removed, update MI as well
                    if coupled_mi_removed and mi_file.exists():
                        mi_nii, _, mi_orig = load_nii(mi_file)
                        new_mi = mi_orig.copy()
                        for cid in removed_sec_ids:
                            new_mi[lab_myo == cid] = False

                        mi_backup = backup_root / mi_file.relative_to(root)
                        mi_backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mi_file, mi_backup)

                        out_mi_nii = nib.Nifti1Image(new_mi.astype(np.uint8), mi_nii.affine, mi_nii.header)
                        nib.save(out_mi_nii, str(mi_file))
                        modified_mi_count += 1

    # Write candidates report
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
        "total_candidate_slices": len(candidates),
        "fully_resolved_to_single_component": fully_resolved,
        "partially_cleaned_slices": len(candidates) - fully_resolved,
        "modified_myo_files": modified_myo_count,
        "modified_mi_files": modified_mi_count,
        "max_noise_px": max_noise_px,
        "max_ratio": max_ratio
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply-safe"], default="dry-run")
    parser.add_argument("--max-noise-px", type=int, default=50)
    parser.add_argument("--max-ratio", type=float, default=0.10)
    args = parser.parse_args()

    summary = run_clean_myo(args.root.resolve(), args.mode, args.max_noise_px, args.max_ratio)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
