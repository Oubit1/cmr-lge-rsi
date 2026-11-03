#!/usr/bin/env python3
"""Refine LGE Myocardium and MI labels using Patient-Level Cross-Slice Continuity.

Clinical Rationale & Rules:
1. Endocardial Blood-Pool Exclusion:
   Near the apex, myocardium contours often leak into high-intensity LV cavity blood pool
   (intensity > 500), which tricks the downstream MI model into circumferential false-positive MI.
   Rule: Prune blood-pool voxels from the inner endocardial boundary.

2. Z-Axis Coronary Territory Continuity & Ectopic Island Pruning:
   Myocardial infarction follows specific coronary artery territories. If a slice has multiple
   MI components and a small secondary component (< 100 px) has 0 overlap with adjacent slices (z ± 1)
   in a distant sector, it is a false-positive artifact and is pruned.

3. Remote Myocardium Thresholding & Circumferential Flood Suppression:
   When MI / Myo ratio is abnormally high (> 0.60) due to low-contrast model confusion,
   constrain MI to genuine hyperenhancement (>= remote myocardium mean + 3*SD),
   eliminating negative/null tissue misclassified as infarct.
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
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    label,
)


def extract_z(path: Path) -> float:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return float(stem)
    except ValueError:
        return 0.0


def refine_patient_series(
    patient: str,
    root: Path,
    mode: str,
    backup_root: Path,
) -> list[dict]:
    lge_img_dir = root / "images/lge" / patient
    lge_myo_dir = root / "labels/lge_checked/lge_Myo_labels" / patient
    lge_mi_dir = root / "labels/lge_checked/lge_MI_labels" / patient

    if not lge_img_dir.exists() or not lge_myo_dir.exists() or not lge_mi_dir.exists():
        return []

    mi_files = sorted(list(lge_mi_dir.glob("*.nii.gz")), key=extract_z, reverse=True)
    if len(mi_files) < 2:
        return []

    # Load all slices in z-order
    slices_data = []
    for f in mi_files:
        myo_p = lge_myo_dir / f.name
        img_p = lge_img_dir / f.name
        if not myo_p.exists() or not img_p.exists():
            continue
        try:
            img_nii = nib.load(img_p)
            myo_nii = nib.load(myo_p)
            mi_nii = nib.load(f)
            img = np.squeeze(img_nii.get_fdata())
            myo = np.squeeze(myo_nii.get_fdata() > 0.5)
            mi = np.squeeze(mi_nii.get_fdata() > 0.5)
            slices_data.append({
                "name": f.name,
                "z": extract_z(f),
                "img": img,
                "myo": myo,
                "mi": mi,
                "myo_nii": myo_nii,
                "mi_nii": mi_nii,
                "myo_path": myo_p,
                "mi_path": f,
            })
        except Exception:
            continue

    if not slices_data:
        return []

    results = []
    n_slices = len(slices_data)

    for i, s in enumerate(slices_data):
        myo_orig = s["myo"].copy()
        mi_orig = s["mi"].copy()
        img = s["img"]

        modified_myo = False
        modified_mi = False

        myo_cur = myo_orig.copy()
        mi_cur = mi_orig.copy()

        # Check issues on current slice
        _, n_mi = label(mi_cur)
        mi_ratio = mi_cur.sum() / max(1, myo_cur.sum())

        # Adjacent slices
        prev_s = slices_data[i - 1] if i > 0 else None
        next_s = slices_data[i + 1] if i < n_slices - 1 else None
        adj_mi = np.zeros_like(mi_cur)
        if prev_s is not None and prev_s["mi"].any():
            adj_mi |= prev_s["mi"]
        if next_s is not None and next_s["mi"].any():
            adj_mi |= next_s["mi"]

        # Rule 1: High MI Fraction & Circumferential Flood Suppression
        if mi_ratio > 0.60 and mi_cur.sum() > 200:
            # Exclude inner blood-pool leak if present
            cavity = binary_fill_holes(myo_cur) ^ myo_cur
            if cavity.any():
                cav_mean = img[cavity].mean()
                if cav_mean > 400:
                    leak = (img > (cav_mean * 0.85)) & binary_dilation(cavity, structure=np.ones((3, 3)))
                    if leak.any():
                        myo_clean = myo_cur & ~leak
                        myo_clean = binary_closing(myo_clean, structure=np.ones((3, 3)))
                        if myo_clean.sum() >= 0.7 * myo_cur.sum():
                            myo_cur = myo_clean
                            modified_myo = True

            # Threshold MI to true enhancement
            low_pct = np.percentile(img[myo_cur], 25) if myo_cur.any() else 0
            rem_vals = img[myo_cur & (img <= low_pct)]
            if len(rem_vals) > 20:
                th_mi = max(120.0, float(rem_vals.mean() + 3.0 * rem_vals.std()))
                cand_mi = mi_cur & (img >= th_mi) & myo_cur
                cand_mi = binary_opening(cand_mi, structure=np.ones((3, 3)))
                cand_mi = binary_closing(cand_mi, structure=np.ones((5, 5))) & myo_cur

                # Keep components that overlap adjacent MI or are sufficiently large in main territory
                lbl_c, n_c = label(cand_mi)
                cleaned = np.zeros_like(cand_mi)
                for c_idx in range(1, n_c + 1):
                    comp = (lbl_c == c_idx)
                    if (comp & adj_mi).sum() > 0 or comp.sum() >= 80:
                        cleaned |= comp
                cleaned = binary_closing(cleaned, structure=np.ones((5, 5))) & myo_cur
                if cleaned.any() and cleaned.sum() < mi_cur.sum():
                    mi_cur = cleaned
                    modified_mi = True

        # Rule 2: Multi-component MI & Unsupported Ectopic Island Pruning
        _, cur_comps = label(mi_cur)
        if cur_comps > 1 and adj_mi.any():
            lbl_mi, n_comps = label(mi_cur)
            # Find which components have adjacent support
            kept_mi = np.zeros_like(mi_cur)
            for c_idx in range(1, n_comps + 1):
                comp = (lbl_mi == c_idx)
                overlap = (comp & binary_dilation(adj_mi, structure=np.ones((3, 3)))).sum()
                # Keep if supported by adjacent slices OR large primary component (> 150 px)
                if overlap > 0 or comp.sum() >= 150:
                    kept_mi |= comp
            # Close small gaps
            kept_mi = binary_closing(kept_mi, structure=np.ones((3, 3))) & myo_cur
            _, after_comps = label(kept_mi)
            if after_comps < cur_comps and kept_mi.any():
                mi_cur = kept_mi
                modified_mi = True

        if modified_myo or modified_mi:
            change_info = {
                "patient": patient,
                "slice": s["name"],
                "z": s["z"],
                "myo_before": int(myo_orig.sum()),
                "myo_after": int(myo_cur.sum()),
                "mi_before": int(mi_orig.sum()),
                "mi_after": int(mi_cur.sum()),
                "ratio_before": float(mi_orig.sum() / max(1, myo_orig.sum())),
                "ratio_after": float(mi_cur.sum() / max(1, myo_cur.sum())),
                "mi_comps_before": int(n_mi),
                "mi_comps_after": int(label(mi_cur)[1]),
            }
            results.append(change_info)

            if mode == "apply-safe":
                # Backup
                b_myo = backup_root / patient / "myo" / s["name"]
                b_mi = backup_root / patient / "mi" / s["name"]
                b_myo.parent.mkdir(parents=True, exist_ok=True)
                b_mi.parent.mkdir(parents=True, exist_ok=True)
                if not b_myo.exists():
                    shutil.copy2(s["myo_path"], b_myo)
                if not b_mi.exists():
                    shutil.copy2(s["mi_path"], b_mi)

                if modified_myo:
                    out_myo = nib.Nifti1Image(
                        myo_cur.astype(np.uint8)[:, :, None] if myo_cur.ndim == 2 else myo_cur.astype(np.uint8),
                        s["myo_nii"].affine,
                        s["myo_nii"].header,
                    )
                    nib.save(out_myo, str(s["myo_path"]))

                if modified_mi:
                    out_mi = nib.Nifti1Image(
                        mi_cur.astype(np.uint8)[:, :, None] if mi_cur.ndim == 2 else mi_cur.astype(np.uint8),
                        s["mi_nii"].affine,
                        s["mi_nii"].header,
                    )
                    nib.save(out_mi, str(s["mi_path"]))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply-safe"], default="dry-run")
    parser.add_argument("--patient", type=str, default=None, help="Optional single patient filter")
    args = parser.parse_args()

    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/lge_continuity_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/lge_continuity_backups" / stamp

    mi_dir = root / "labels/lge_checked/lge_MI_labels"
    if args.patient:
        patient_dirs = [mi_dir / args.patient]
    else:
        patient_dirs = sorted([p for p in mi_dir.iterdir() if p.is_dir()])

    print(f"Running LGE cross-slice continuity ({args.mode}) on {len(patient_dirs)} patients...", flush=True)

    all_changes = []
    for pd in patient_dirs:
        if not pd.is_dir():
            continue
        changes = refine_patient_series(pd.name, root, args.mode, backup_root)
        if changes:
            all_changes.extend(changes)
            print(f"  [{pd.name}]: {len(changes)} slices refined.")

    csv_path = run_dir / "refined_slices.csv"
    if all_changes:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_changes[0].keys()))
            w.writeheader()
            w.writerows(all_changes)

    summary = {
        "root": str(root),
        "mode": args.mode,
        "run_dir": str(run_dir),
        "total_patients_checked": len(patient_dirs),
        "patients_modified": len(set(c["patient"] for c in all_changes)),
        "slices_refined": len(all_changes),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSummary:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
