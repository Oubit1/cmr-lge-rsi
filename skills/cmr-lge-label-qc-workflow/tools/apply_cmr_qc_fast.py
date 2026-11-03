#!/usr/bin/env python3
"""Fast parallel execution of CMR QC audit and conservative safe repair.

Processes patients across multiple CPU cores to avoid timeouts on large
datasets with thousands of 2D/3D CMR series.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmr_qc import (as_stack, components, load, nearby, numeric_stem,
                     restore_stack, ring_stats, supported_small_ring,
                     temporal_missing_hole_candidate)


def process_single_patient(root: Path, patient_name: str, mode: str, backup_root: Path) -> tuple[list[dict], list[dict], int]:
    """Process all CMR volumes for a single patient."""
    image_dir = root / "images/cmr" / patient_name
    label_dir = root / "labels/cmr/cmr_Myo_mask" / patient_name
    lge_label_dir = root / "labels/lge_checked/lge_Myo_labels" / patient_name
    rows = []
    modifications = []
    modified_vols = 0

    if not image_dir.exists():
        return rows, modifications, modified_vols

    image_paths = sorted(image_dir.glob("*.nii.gz"), key=numeric_stem)
    for volume_index, image_path in enumerate(image_paths):
        label_path = label_dir / image_path.name
        try:
            _, image_raw = load(image_path)
            image_stack, _ = as_stack(image_raw)
        except Exception as e:
            continue

        if label_path.exists():
            try:
                label_nii, label_raw = load(label_path)
                label_stack, label_axis = as_stack(label_raw)
            except Exception:
                continue
            if image_stack.shape != label_stack.shape:
                rows.append({
                    "patient": patient_name,
                    "volume": image_path.name,
                    "slice_index": "",
                    "issues": "cmr_shape_mismatch",
                    "evidence": f"image={image_stack.shape};label={label_stack.shape}",
                    "check": ""
                })
                continue
        else:
            label_raw = np.zeros_like(image_raw)
            label_stack, label_axis = np.zeros_like(image_stack), None

        n_slices = image_stack.shape[-1]
        spatial_boundary_volume = volume_index < 2 or volume_index >= len(image_paths) - 2
        lge_path = lge_label_dir / image_path.name
        lge_mask = None
        if lge_path.exists():
            try:
                _, lge_raw = load(lge_path)
                lge_stack, _ = as_stack(lge_raw)
                lge_mask = lge_stack[..., 0].astype(bool)
            except Exception:
                lge_mask = None
        nonempty_indices = [k for k in range(n_slices) if np.any(label_stack[..., k])]
        first_nonempty = min(nonempty_indices) if nonempty_indices else None
        last_nonempty = max(nonempty_indices) if nonempty_indices else None
        boundary_slices = set(range(n_slices)) if spatial_boundary_volume else set()
        metrics = {}
        new_stack = label_stack.astype(bool).copy()
        volume_changed = False

        for k in range(n_slices):
            mask = label_stack[..., k].astype(bool)
            tags, evidence = [], []
            if not label_path.exists():
                tags += ["missing_cmr_myo"]
                evidence += ["label_missing"]
            else:
                _, cs = components(mask)
                if not mask.any():
                    tags.append("cmr_myo_empty")

                if mask.any():
                    if k in (0, n_slices - 1) or k in boundary_slices:
                        if len(cs) > 2:
                            tags.append("cmr_myo_excess_components")
                            evidence.append(f"components={len(cs)}")
                    else:
                        if len(cs) > 2:
                            tags.append("cmr_myo_excess_components")
                            evidence.append(f"components={len(cs)}")

                    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
                        tags.append("cmr_myo_touches_image_border")

                    small_supported = (not spatial_boundary_volume and
                                       supported_small_ring(label_stack, k, lge_mask))
                    if mask.sum() < 400 and not small_supported:
                        tags.append("cmr_myo_small_area")
                        evidence.append(f"pixels={int(mask.sum())}")

                    # Hole count
                    filled = ndi.binary_fill_holes(mask)
                    _, holes = components(filled & ~mask)
                    hole_n = len(holes)
                    if hole_n > 2:
                        tags.append("cmr_myo_multiple_holes")
                        evidence.append(f"holes={hole_n}")
                    missing_hole, hole_dice, supported_px = temporal_missing_hole_candidate(label_stack, k)
                    if missing_hole:
                        tags.append("cmr_myo_missing_temporal_hole")
                        evidence.append(f"adjacent_hole_dice={hole_dice:.3f};supported_hole_pixels={supported_px}")

                    yy, xx = np.where(mask)
                    metrics[k] = (int(mask.sum()), float(xx.mean()) if len(xx) else np.nan, float(yy.mean()) if len(yy) else np.nan)

                    neighbours = [label_stack[..., j].astype(bool) for j in (k - 1, k + 1) if 0 <= j < n_slices]
                    unsupported = [(cid, area) for cid, area, cx, cy in cs if area <= 20 and not any(nearby(nm, cx, cy) for nm in neighbours)]
                    if unsupported:
                        tags.append("cmr_tiny_unsupported_component")
                        evidence.append(str(unsupported))

                    if mode == "apply-safe" and unsupported:
                        lab, _ = components(mask)
                        new = mask.copy()
                        for cid, _ in unsupported:
                            new[lab == cid] = False
                        new_stack[..., k] = new
                        volume_changed = True

            if tags:
                rows.append({
                    "patient": patient_name,
                    "volume": image_path.name,
                    "slice_index": k + 1,
                    "issues": ";".join(tags),
                    "evidence": ";".join(evidence),
                    "check": ""
                })

        # Repair internal dropout
        if mode == "apply-safe":
            for k in range(1, n_slices - 1):
                if k not in metrics or k - 1 not in metrics or k + 1 not in metrics:
                    continue
                area, cx, cy = metrics[k]
                a0, x0, y0 = metrics[k - 1]
                a1, x1, y1 = metrics[k + 1]
                if area >= 400 or min(a0, a1) < 400:
                    continue
                if max(a0, a1) / max(min(a0, a1), 1) > 1.6:
                    continue
                if max(((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5, 0) > 20:
                    continue
                current = label_stack[..., k].astype(bool)
                prev = label_stack[..., k - 1].astype(bool)
                nxt = label_stack[..., k + 1].astype(bool)
                _, cc = components(current)
                _, cp = components(prev)
                _, cn = components(nxt)
                if not cc or len(cp) > 2 or len(cn) > 2:
                    continue
                _, current_comp, current_holes, current_border, _ = ring_stats(current)
                if current_comp == 1 and current_holes == 1 and not current_border:
                    continue
                reference = prev if a0 <= a1 else nxt
                new_stack[..., k] = reference
                volume_changed = True
                modifications.append({
                    "timestamp": datetime.now().isoformat(),
                    "modality": "cmr",
                    "patient": patient_name,
                    "volume": image_path.name,
                    "slice_index": k + 1,
                    "rule": "internal_small_area_supported_by_adjacent_rings",
                    "action": "replace_with_adjacent_reference",
                    "before_pixels": int(area),
                    "after_pixels": int(reference.sum()),
                    "status": "applied"
                })

        for k in range(1, n_slices - 1):
            if k not in metrics or k - 1 not in metrics or k + 1 not in metrics:
                continue
            area, cx, cy = metrics[k]
            a0, x0, y0 = metrics[k - 1]
            a1, x1, y1 = metrics[k + 1]
            if min(area, a0, a1) <= 0:
                continue
            neighbour_area = (a0 + a1) / 2.0
            area_ratio = max(area, neighbour_area) / max(min(area, neighbour_area), 1)
            drift = max(((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5, ((cx - x1) ** 2 + (cy - y1) ** 2) ** 0.5)
            tags, evidence = [], []
            if area_ratio > 3.0 and k not in boundary_slices:
                tags.append("cmr_layer_area_jump")
                evidence.append(f"area={area};adjacent_mean={neighbour_area:.1f};ratio={area_ratio:.2f}")
            if drift > 20.0 and k not in boundary_slices:
                tags.append("cmr_centroid_drift")
                evidence.append(f"max_neighbor_distance={drift:.1f}")

            if tags:
                rows.append({
                    "patient": patient_name,
                    "volume": image_path.name,
                    "slice_index": k + 1,
                    "issues": ";".join(tags),
                    "evidence": ";".join(evidence),
                    "check": ""
                })

        if volume_changed and mode == "apply-safe":
            backup = backup_root / label_path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, backup)
            src = nib.load(str(label_path))
            restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
            out = nib.Nifti1Image(restored, src.affine, src.header)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(label_path))
            modified_vols += 1
            modifications.append({
                "timestamp": datetime.now().isoformat(),
                "modality": "cmr",
                "patient": patient_name,
                "volume": image_path.name,
                "rule": "tiny_unsupported_cmr_component",
                "action": "remove",
                "backup": str(backup),
                "status": "applied"
            })

    return rows, modifications, modified_vols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "apply-safe"], default="apply-safe")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/cmr_qc_runs" / f"fast_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_root = root / "label_iteration/cmr_qc_backups" / f"fast_{stamp}"

    image_root = root / "images/cmr"
    patients = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    print(f"Starting parallel CMR QC ({args.mode}) on {len(patients)} patients with {args.workers} workers...", flush=True)

    all_rows = []
    all_mods = []
    total_modified = 0

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_patient, root, p, args.mode, backup_root): p for p in patients}
        for fut in as_completed(futures):
            p = futures[fut]
            completed += 1
            if completed % 25 == 0 or completed == len(patients):
                print(f"Progress: {completed}/{len(patients)} patients processed ({len(all_rows)} issue rows found so far)...", flush=True)
            try:
                rows, mods, modified_vols = fut.result()
                all_rows.extend(rows)
                all_mods.extend(mods)
                total_modified += modified_vols
            except Exception as e:
                print(f"Error processing {p}: {e}", flush=True)

    # Write modifications log
    log_path = run_dir / "modifications.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for m in all_mods:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Write issue manifest
    fields = ["patient", "volume", "slice_index", "issues", "evidence", "check"]
    public_manifest = root / "labels/check/cmr_issue_slices/issue_manifest.csv"
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    for p_out in (run_dir / "issue_manifest.csv", public_manifest):
        with p_out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "root": str(root),
        "mode": args.mode,
        "patients": len(patients),
        "candidate_slices": len(all_rows),
        "modified_volumes": total_modified,
        "total_modifications": len(all_mods),
        "run": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
