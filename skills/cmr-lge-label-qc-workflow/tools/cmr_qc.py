#!/usr/bin/env python3
"""QC and conservative repair for CMR myocardial masks.

CMR files in the site_b processed dataset are usually 3-D (H, W, slices),
while LGE files are one slice per NIfTI. This tool audits CMR one slice at a
time and writes any safe repair back to the original 3-D volume orientation.
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def load(path: Path):
    nii = nib.load(str(path))
    return nii, np.asarray(nii.get_fdata())


def as_stack(array: np.ndarray) -> tuple[np.ndarray, int | None]:
    arr = np.squeeze(array)
    if arr.ndim == 2:
        return arr[..., None], None
    if arr.ndim != 3:
        raise ValueError(f"CMR image/mask must be 2-D or 3-D after squeeze, got {arr.shape}")
    candidates = [i for i, n in enumerate(arr.shape) if n <= 64 and all(arr.shape[j] > n for j in range(3) if j != i)]
    axis = candidates[0] if len(candidates) == 1 else int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1), axis


def restore_stack(stack: np.ndarray, original: np.ndarray, axis: int | None) -> np.ndarray:
    arr = stack[..., 0] if axis is None else np.moveaxis(stack, -1, axis)
    if arr.shape == original.shape:
        return arr
    squeezed = np.squeeze(original)
    if arr.shape == squeezed.shape and original.ndim == 3 and original.shape[-1] == 1:
        return arr[..., None]
    raise ValueError(f"cannot restore CMR shape {arr.shape} to {original.shape}")


def components(mask: np.ndarray):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        yy, xx = np.where(lab == k)
        if len(xx):
            out.append((k, len(xx), float(xx.mean()), float(yy.mean())))
    return lab, out


def nearby(mask: np.ndarray, cx: float, cy: float, radius: float = 15.0) -> bool:
    _, cs = components(mask)
    return any(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= radius for _, _, x, y in cs)


def disk_struct(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x**2 + y**2 <= r**2).astype(bool)


def numeric_stem(path: Path):
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem)


def ring_stats(mask: np.ndarray):
    mask = mask.astype(bool)
    _, cs = components(mask)
    holes = ndi.binary_fill_holes(mask) & ~mask
    _, hole_n = ndi.label(holes, structure=np.ones((3, 3)))
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    yy, xx = np.where(mask)
    center = (float(xx.mean()), float(yy.mean())) if len(xx) else (np.nan, np.nan)
    return int(mask.sum()), len(cs), int(hole_n), border, center


def supported_small_ring(stack: np.ndarray, k: int, lge_mask: np.ndarray | None) -> bool:
    """True for a small but valid ring supported by cine time and LGE."""
    n_frames = stack.shape[-1]
    if lge_mask is None or n_frames < 3 or k < 0 or k >= n_frames:
        return False
    stats = [ring_stats(stack[..., j]) for j in range(n_frames)]
    area, comp, holes, border, center = stats[k]
    p_k = (k - 1) % n_frames
    n_k = (k + 1) % n_frames
    p_area, p_comp, p_holes, p_border, p_center = stats[p_k]
    n_area, n_comp, n_holes, n_border, n_center = stats[n_k]
    lge_area, lge_comp, lge_holes, lge_border, _ = ring_stats(lge_mask)
    valid_fraction = sum(c == 1 and h == 1 and not b for _, c, h, b, _ in stats) / len(stats)
    local_ratio = area / max((p_area + n_area) / 2.0, 1.0)
    area_fraction = area / float(stack.shape[0] * stack.shape[1])
    drift = max(np.hypot(center[0] - p_center[0], center[1] - p_center[1]),
                np.hypot(center[0] - n_center[0], center[1] - n_center[1]))
    return bool(
        comp == holes == 1 and not border
        and p_comp == p_holes == 1 and not p_border
        and n_comp == n_holes == 1 and not n_border
        and valid_fraction >= 0.90 and area_fraction >= 0.004
        and 0.75 <= local_ratio <= 1.33 and drift <= 8.0
        and lge_area > 0 and lge_comp == lge_holes == 1 and not lge_border
    )


def temporal_missing_hole_candidate(stack: np.ndarray, k: int, min_dice: float = 0.85,
                                    max_run: int = 3) -> tuple[bool, float, int]:
    """Detect a short solid-frame run bracketed by matching blood-pool holes."""
    if k <= 0 or k >= stack.shape[-1] - 1:
        return False, 0.0, 0
    _, comp, holes, border, _ = ring_stats(stack[..., k])
    if comp != 1 or holes != 0 or border:
        return False, 0.0, 0
    left = k - 1
    while left >= 0 and k - left <= max_run and ring_stats(stack[..., left])[2] == 0:
        left -= 1
    right = k + 1
    while right < stack.shape[-1] and right - k <= max_run and ring_stats(stack[..., right])[2] == 0:
        right += 1
    if left < 0 or right >= stack.shape[-1] or right - left - 1 > max_run:
        return False, 0.0, 0
    candidates = []
    for j in (left, right):
        mask = stack[..., j].astype(bool)
        _, c, h, b, _ = ring_stats(mask)
        if c != 1 or h != 1 or b:
            return False, 0.0, 0
        candidates.append(ndi.binary_fill_holes(mask) & ~mask)
    overlap = int((candidates[0] & candidates[1]).sum())
    denom = int(candidates[0].sum() + candidates[1].sum())
    score = 2.0 * overlap / denom if denom else 0.0
    return score >= min_dice and overlap >= 30, score, overlap


def render(image: np.ndarray, mask: np.ndarray, out: Path, title: str) -> None:
    arr = np.asarray(image).astype(float)
    finite = np.isfinite(arr)
    lo, hi = np.percentile(arr[finite], [1, 99]) if finite.any() else (0, 1)
    gray = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[mask] = (0.42 * rgb[mask] + 0.58 * np.array([30, 220, 80])).astype(np.uint8)
    im = Image.fromarray(rgb).convert("RGB")
    im.thumbnail((640, 640))
    canvas = Image.new("RGB", (im.width, im.height + 30), "white")
    canvas.paste(im, (0, 30))
    ImageDraw.Draw(canvas).text((5, 7), title, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def audit(root: Path, mode: str, patient_filter: str | None = None, max_volumes: int | None = None, render_checks: bool = True, start_patient: int = 0, max_patients: int | None = None) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_qc_runs" / stamp
    checks = run / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    image_root = root / "images/cmr"
    label_root = root / "labels/cmr/cmr_Myo_mask"
    lge_label_root = root / "labels/lge_checked/lge_Myo_labels"
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    for old_png in issue_root.glob("*.png"):
        old_png.unlink()
    backup_root = root / "label_iteration/cmr_qc_backups" / stamp
    log_path = run / "modifications.jsonl"
    rows, modified = [], 0

    volume_count = 0
    patient_dirs = sorted(p for p in image_root.iterdir() if p.is_dir())
    if patient_filter:
        patient_dirs = [p for p in patient_dirs if p.name == patient_filter]
    else:
        patient_dirs = patient_dirs[start_patient: start_patient + max_patients if max_patients is not None else None]
    total_patients = len(patient_dirs)
    for patient_index, patient_dir in enumerate(patient_dirs, start=1):
        if patient_index == 1 or patient_index % 10 == 0 or patient_index == total_patients:
            print(json.dumps({"progress": "cmr_qc", "patient": patient_index,
                              "patients": total_patients, "volumes": volume_count,
                              "candidates": len(rows)}, ensure_ascii=False), flush=True)
        image_paths = sorted(patient_dir.glob("*.nii.gz"), key=numeric_stem)
        for volume_index, image_path in enumerate(image_paths):
            if max_volumes is not None and volume_count >= max_volumes:
                break
            volume_count += 1
            label_path = label_root / patient_dir.name / image_path.name
            _, image_raw = load(image_path)
            image_stack, _ = as_stack(image_raw)
            if label_path.exists():
                label_nii, label_raw = load(label_path)
                label_stack, label_axis = as_stack(label_raw)
                if image_stack.shape != label_stack.shape:
                    rows.append({"patient": patient_dir.name, "volume": image_path.name, "slice_index": "", "issues": "cmr_shape_mismatch", "evidence": f"image={image_stack.shape};label={label_stack.shape}", "check": ""})
                    continue
            else:
                label_raw = np.zeros_like(image_raw)
                label_stack, label_axis = np.zeros_like(image_stack), None

            n_slices = image_stack.shape[-1]
            spatial_boundary_volume = volume_index < 2 or volume_index >= len(image_paths) - 2
            lge_path = lge_label_root / patient_dir.name / image_path.name
            lge_mask = None
            if lge_path.exists():
                _, lge_raw = load(lge_path)
                lge_stack, _ = as_stack(lge_raw)
                lge_mask = lge_stack[..., 0].astype(bool)
            nonempty_indices = [k for k in range(n_slices) if np.any(label_stack[..., k])]
            first_nonempty = min(nonempty_indices) if nonempty_indices else None
            last_nonempty = max(nonempty_indices) if nonempty_indices else None
            boundary_slices = set(range(n_slices)) if spatial_boundary_volume else set()
            metrics = {}
            new_stack = label_stack.astype(bool).copy()
            volume_changed = False
            for k in range(n_slices):
                image = image_stack[..., k]
                mask = label_stack[..., k].astype(bool)
                tags, evidence = [], []
                if not label_path.exists():
                    tags += ["missing_cmr_myo"]
                    evidence += ["label_missing"]
                else:
                    _, cs = components(mask)
                    if not mask.any():
                        tags.append("cmr_myo_empty")
                    # CMR masks may intentionally contain separate LV and RV
                    # myocardial rings. One or two major components are valid;
                    # only excess components are suspicious.
                    if len(cs) > 2:
                        tags.append("cmr_myo_excess_components")
                        evidence.append(f"components={len(cs)}")
                    # The third axis is temporal in this dataset. A fixed
                    # pixel threshold alone must not reject a stable ring.
                    small_supported = (not spatial_boundary_volume and
                                       supported_small_ring(label_stack, k, lge_mask))
                    if mask.sum() < 400 and not small_supported:
                        tags.append("cmr_myo_small_area")
                        evidence.append(f"pixels={int(mask.sum())}")
                    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
                        tags.append("cmr_myo_touches_image_border")
                    holes = ndi.binary_fill_holes(mask) & ~mask
                    _, hole_n = ndi.label(holes, structure=np.ones((3, 3)))
                    # With two normal ventricular rings, two blood-pool holes
                    # are expected. Extra holes are the review signal.
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
                        before = int(mask.sum())
                        for cid, _ in unsupported:
                            new[lab == cid] = False
                        new_stack[..., k] = new
                        volume_changed = True
                if tags:
                    filename = f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}.png"
                    png = checks / filename
                    if render_checks:
                        render(image, new_stack[..., k] if volume_changed else mask, png, f"{patient_dir.name} | {image_path.name} | slice {k + 1} | {';'.join(tags)}")
                        shutil.copy2(png, issue_root / filename)
                    rows.append({"patient": patient_dir.name, "volume": image_path.name, "slice_index": k + 1, "issues": ";".join(tags), "evidence": ";".join(evidence), "check": str(png) if render_checks else ""})

            # Repair a high-confidence internal dropout: the current layer is
            # unusually small, while both adjacent layers contain compatible
            # complete myocardium. This is distinct from an apex/base boundary.
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
                    # Keep the adjacent mask with area closest to their local
                    # median; this preserves the observed ring morphology.
                    reference = prev if a0 <= a1 else nxt
                    new_stack[..., k] = reference
                    volume_changed = True
                    if log_path.parent.exists():
                        rec = {"timestamp": datetime.now().isoformat(), "modality": "cmr", "patient": patient_dir.name, "volume": image_path.name, "slice_index": k + 1, "rule": "internal_small_area_supported_by_adjacent_rings", "action": "replace_with_adjacent_reference", "before_pixels": int(area), "after_pixels": int(reference.sum()), "status": "applied"}
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Repair temporal missing hole (unclosed myocardium ring)
                for k in range(n_slices):
                    missing_hole, _, _ = temporal_missing_hole_candidate(new_stack, k)
                    if not missing_hole:
                        continue
                    m = new_stack[..., k].astype(bool)
                    adj_masks = []
                    for j in (k - 1, k + 1):
                        if 0 <= j < n_slices:
                            adj_m = new_stack[..., j].astype(bool)
                            adj_h = ndi.binary_fill_holes(adj_m) & ~adj_m
                            if adj_h.sum() > 200:
                                adj_masks.append(adj_m)
                    if not adj_masks:
                        continue
                    env = ndi.binary_dilation(np.bitwise_or.reduce(adj_masks), structure=disk_struct(3))
                    for rad in range(2, 6):
                        closed = ndi.binary_closing(m, structure=disk_struct(rad))
                        ch = ndi.binary_fill_holes(closed) & ~closed
                        if ch.sum() > 200:
                            h_bound = ndi.binary_dilation(ch, structure=disk_struct(1)) & ~ch
                            bridge = closed & ~m & h_bound
                            if 0 < bridge.sum() <= 60 and env[bridge].mean() >= 0.80:
                                new_stack[..., k] = m | bridge
                                volume_changed = True
                                if log_path.parent.exists():
                                    rec = {"timestamp": datetime.now().isoformat(), "modality": "cmr", "patient": patient_dir.name, "volume": image_path.name, "slice_index": k + 1, "rule": "temporal_missing_hole_ring_closure", "action": "bridge_narrow_gap", "before_pixels": int(m.sum()), "after_pixels": int(m.sum() + bridge.sum()), "status": "applied"}
                                    with log_path.open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                break

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
                    filename = f"{patient_dir.name}_{image_path.name[:-7]}_s{k + 1:02d}.png"
                    png = checks / filename
                    if render_checks:
                        render(image_stack[..., k], new_stack[..., k], png, f"{patient_dir.name} | {image_path.name} | slice {k + 1} | {';'.join(tags)}")
                        shutil.copy2(png, issue_root / filename)
                    rows.append({"patient": patient_dir.name, "volume": image_path.name, "slice_index": k + 1, "issues": ";".join(tags), "evidence": ";".join(evidence), "check": str(png) if render_checks else ""})

            if volume_changed and mode == "apply-safe":
                backup = backup_root / label_path.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, backup)
                src = nib.load(str(label_path))
                restored = restore_stack(new_stack.astype(np.uint8), label_raw, label_axis)
                out = nib.Nifti1Image(restored, src.affine, src.header)
                out.set_data_dtype(np.uint8)
                nib.save(out, str(label_path))
                modified += 1
                record = {"timestamp": datetime.now().isoformat(), "modality": "cmr", "patient": patient_dir.name, "volume": image_path.name, "rule": "tiny_unsupported_cmr_component", "action": "remove", "backup": str(backup), "status": "applied"}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = ["patient", "volume", "slice_index", "issues", "evidence", "check"]
    for path in (run / "issue_manifest.csv", issue_root / "issue_manifest.csv"):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {"root": str(root), "mode": mode, "candidate_slices": len(rows), "modified_volumes": modified, "run": str(run), "rules": ["one or two major components (LV/RV) are valid; excess components are review-only", "tiny <=20px component without 15px support above/below may be removed", "small single/dual-component apex/base boundary layers within five occupied slices of a volume edge are valid unless both adjacent layers are large (internal dropout)", "internal small-area layer may be replaced only when both adjacent rings are large, position-compatible and structurally unambiguous", "area ratio >3 or centroid drift >20px is review-only outside the apex/base boundary"]}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "apply-safe"], default="audit")
    parser.add_argument("--patient", help="debug/review only: restrict to one patient")
    parser.add_argument("--max-volumes", type=int, help="debug/review only: cap processed volumes")
    parser.add_argument("--no-render", action="store_true", help="audit only; skip PNG generation for large batches")
    parser.add_argument("--start-patient", type=int, default=0)
    parser.add_argument("--max-patients", type=int)
    args = parser.parse_args()
    audit(args.root.resolve(), args.mode, args.patient, args.max_volumes, not args.no_render, args.start_patient, args.max_patients)


if __name__ == "__main__":
    main()
