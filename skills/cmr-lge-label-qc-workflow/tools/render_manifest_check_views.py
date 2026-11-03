#!/usr/bin/env python3
"""Render overlay PNG check views for an issue manifest.

Follows dataset standard conventions:
- CMR: 3-panel triptych (Original, Mask overlay, Mask contour) with red myocardium
  matching /srv/rsi/site_b/original/labels/check/cmr/cmr_Myo_mask/SUBJECT_001/78.6223.png
- LGE: 6-panel grid (Original, Myo overlay/contour, Myo+MI overlay/contour) with green myocardium and red infarct
  matching /srv/rsi/site_b/original/Processed_dataset_V1/labels/check/lge_issue_slices/LI_FENG_HE_077Y_-82.7630.png
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cmr_qc import as_stack, load


def mask_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = np.zeros_like(mask)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    boundary = mask & ~eroded
    for _ in range(max(0, thickness - 1)):
        padded = np.pad(boundary, 1, mode="constant")
        boundary = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return boundary


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    values = image[finite]
    lower, upper = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
    normalized[~finite] = 0.0
    return np.rint(normalized * 255.0).astype(np.uint8)


def render_cmr_triptych(
    image: np.ndarray,
    mask: np.ndarray,
    title: str,
    out: Path,
    alpha: float = 0.42,
    contour_thickness: int = 1,
) -> None:
    gray = normalize_grayscale(image)
    base = np.repeat(gray[..., None], 3, axis=-1)
    mask_bool = np.asarray(mask) > 0.5

    overlay = base.copy()
    if mask_bool.any():
        overlay_values = (1.0 - alpha) * overlay[mask_bool].astype(np.float32)
        overlay_values += alpha * np.array([255.0, 0.0, 0.0], dtype=np.float32)
        overlay[mask_bool] = np.rint(overlay_values).astype(np.uint8)

    contour = base.copy()
    b_mask = mask_boundary(mask_bool, thickness=contour_thickness)
    if b_mask.any():
        contour[b_mask] = np.array([255, 0, 0], dtype=np.uint8)

    panel_height, panel_width = gray.shape
    separator = 6
    header_height = 48
    canvas_width = panel_width * 3 + separator * 2
    canvas = Image.new("RGB", (canvas_width, panel_height + header_height), "white")
    panels = [Image.fromarray(base), Image.fromarray(overlay), Image.fromarray(contour)]
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * (panel_width + separator), header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ["Original", "Mask overlay", "Mask contour"]
    for index, label in enumerate(labels):
        x0 = index * (panel_width + separator)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((x0 + max(3, (panel_width - text_width) // 2), 5), label, fill="black", font=font)
    display_title = title if len(title) <= 150 else title[:147] + "..."
    draw.text((5, 25), display_title, fill="black", font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out))


def process_volume(task: tuple[str, str, list[int], Path, Path, bool]) -> list[tuple[str, str, int, str]]:
    patient, volume, slice_indices, root, out_dir, overwrite = task
    image_path = root / "images/cmr" / patient / volume
    label_path = root / "labels/cmr/cmr_Myo_mask" / patient / volume
    if not image_path.exists() or not label_path.exists():
        return []
    try:
        _, image_raw = load(image_path)
        _, label_raw = load(label_path)
        image_stack, _ = as_stack(image_raw)
        label_stack, _ = as_stack(label_raw)
    except Exception:
        return []

    vol_stem = volume[:-7] if volume.endswith(".nii.gz") else volume
    results = []
    n_slices = image_stack.shape[-1]
    for s_idx in slice_indices:
        k = s_idx - 1
        if 0 <= k < n_slices:
            filename = f"{patient}_{vol_stem}_s{s_idx:02d}.png"
            target = out_dir / filename
            if overwrite or not target.exists():
                title = f"cmr_Myo_mask | {patient}/{volume} | slice {s_idx}"
                render_cmr_triptych(image_stack[..., k], label_stack[..., k], title, target)
            results.append((patient, volume, s_idx, str(target)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    out_dir = manifest_path.parent

    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Manifest is empty.")
        return

    vol_to_slices: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        try:
            s_idx = int(r["slice_index"])
            vol_to_slices.setdefault((r["patient"], r["volume"]), set()).add(s_idx)
        except (ValueError, TypeError):
            continue

    tasks = [
        (patient, volume, sorted(slices), root, out_dir, args.overwrite)
        for (patient, volume), slices in vol_to_slices.items()
    ]

    print(f"Rendering standard 3-panel check views for {len(tasks)} volumes ({len(rows)} rows) with {args.workers} workers...", flush=True)

    rendered_map: dict[tuple[str, str, int], str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_volume, t) for t in tasks]
        for fut in as_completed(futures):
            for patient, volume, s_idx, target_path in fut.result():
                rendered_map[(patient, volume, s_idx)] = target_path

    fields = list(rows[0].keys())
    if "check" not in fields:
        fields.append("check")
    if "check_view" not in fields:
        fields.append("check_view")

    for r in rows:
        try:
            key = (r["patient"], r["volume"], int(r["slice_index"]))
            if key in rendered_map:
                r["check"] = rendered_map[key]
                r["check_view"] = rendered_map[key]
        except (ValueError, TypeError):
            pass

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Rendered {len(rendered_map)} check view PNGs. Updated {manifest_path}.", flush=True)


if __name__ == "__main__":
    main()
