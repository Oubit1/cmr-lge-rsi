#!/usr/bin/env python3
"""Render CMR/LGE QC checks with modality-specific legacy V1 layouts.

The renderer is read-only with respect to NIfTI labels.  It only refreshes
generated PNGs and the check manifest paths.

CMR uses the three-panel standalone myocardium convention (red mask), while
LGE uses the six-panel combined convention (green myocardium + red infarct).
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


GREEN = np.array([0, 230, 115], dtype=np.float32)
RED = np.array([255, 40, 40], dtype=np.float32)
ALPHA = 0.42


def font(size: int):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


TITLE = font(16)
SUB = font(13)


def load(path: Path):
    return np.squeeze(nib.load(str(path)).get_fdata())


def norm_u8(arr):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    p1, p99 = np.percentile(arr[finite], [1, 99])
    if p99 <= p1:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip((arr - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)


def contour(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask)
    p = np.pad(mask, 1, constant_values=False)
    eroded = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~eroded


def panel_set(image, myo, mi=None):
    base = np.stack([norm_u8(image)] * 3, axis=-1).astype(np.float32)
    myo = myo.astype(bool)
    mi = np.zeros_like(myo) if mi is None else mi.astype(bool)

    overlay_myo = base.copy()
    overlay_myo[myo] = (1 - ALPHA) * overlay_myo[myo] + ALPHA * GREEN
    overlay_both = base.copy()
    overlay_both[myo & ~mi] = (1 - ALPHA) * overlay_both[myo & ~mi] + ALPHA * GREEN
    overlay_both[mi] = (1 - ALPHA) * overlay_both[mi] + ALPHA * RED

    myo_c = contour(myo)
    mi_c = contour(mi)
    contour_myo = base.copy(); contour_myo[myo_c] = GREEN
    contour_both = base.copy(); contour_both[myo_c] = GREEN; contour_both[mi_c] = RED
    return [
        base, overlay_myo, overlay_both,
        base, contour_myo, contour_both,
    ]


def cmr_panel_set(image, myo):
    """Legacy standalone-mask view: original, red overlay, red contour."""
    base = np.stack([norm_u8(image)] * 3, axis=-1).astype(np.float32)
    myo = myo.astype(bool)
    overlay = base.copy()
    overlay[myo] = (1 - ALPHA) * overlay[myo] + ALPHA * RED
    outlined = base.copy()
    outlined[contour(myo)] = RED
    return [base, overlay, outlined]


def draw_lge_grid(image, myo, mi, issue, title):
    panels = [np.clip(x, 0, 255).astype(np.uint8) for x in panel_set(image, myo, mi)]
    h, w = panels[0].shape[:2]
    gap, header, row_gap, sub_h = 6, 48, 6, 28
    total_w = w * 3 + gap * 2
    total_h = header + h + row_gap + sub_h + h
    canvas = Image.new("RGB", (total_w, total_h), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, total_w, 26), fill=(255, 235, 180))
    d.text((8, 5), f"QC ISSUE: {issue}", fill=(150, 0, 0), font=SUB)
    max_title_w = max(total_w - 245, 120)
    title_text = title
    while d.textbbox((0, 0), title_text, font=SUB)[2] > max_title_w and len(title_text) > 12:
        title_text = title_text[:-4] + "..."
    d.text((8, 29), title_text, fill=(0, 0, 0), font=SUB)
    d.rectangle((total_w - 220, 30, total_w - 208, 42), fill=tuple(GREEN.astype(int)))
    d.text((total_w - 204, 28), "Myo (心肌)", fill=(0, 0, 0), font=SUB)
    d.rectangle((total_w - 105, 30, total_w - 93, 42), fill=tuple(RED.astype(int)))
    d.text((total_w - 89, 28), "MI (梗死)", fill=(0, 0, 0), font=SUB)

    labels_bottom = ["Original", "Mask contour (Myo)", "Mask contour (Myo+MI)"]
    for i in range(3):
        x = i * (w + gap)
        canvas.paste(Image.fromarray(panels[i]), (x, header))
    r2_y = header + h + row_gap
    for i, label in enumerate(labels_bottom):
        x = i * (w + gap)
        bb = TITLE.getbbox(label)
        d.text((x + (w - (bb[2] - bb[0])) // 2, r2_y + 4), label, fill=(0, 0, 0), font=TITLE)
        canvas.paste(Image.fromarray(panels[3 + i]), (x, r2_y + sub_h))
    return canvas


def draw_cmr_grid(image, myo, issue, title):
    panels = [np.clip(x, 0, 255).astype(np.uint8) for x in cmr_panel_set(image, myo)]
    h, w = panels[0].shape[:2]
    gap, header, sub_h = 6, 48, 28
    total_w = w * 3 + gap * 2
    total_h = header + sub_h + h
    canvas = Image.new("RGB", (total_w, total_h), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, total_w, 26), fill=(255, 235, 180))
    d.text((8, 5), f"QC ISSUE: {issue}", fill=(150, 0, 0), font=SUB)
    title_text = title
    while d.textbbox((0, 0), title_text, font=SUB)[2] > total_w - 16 and len(title_text) > 12:
        title_text = title_text[:-4] + "..."
    d.text((8, 29), title_text, fill=(0, 0, 0), font=SUB)
    labels = ["Original", "Mask overlay (Myo)", "Mask contour (Myo)"]
    for i, (label, panel) in enumerate(zip(labels, panels)):
        x = i * (w + gap)
        bb = TITLE.getbbox(label)
        d.text((x + (w - (bb[2] - bb[0])) // 2, header + 4), label,
               fill=(0, 0, 0), font=TITLE)
        canvas.paste(Image.fromarray(panel), (x, header + sub_h))
    return canvas


def stack_slice(arr, index):
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1)[..., index]


def render_cmr_volume(task):
    root_s, out_s, patient, name, entries = task
    root, out = Path(root_s), Path(out_s)
    image_path = root / "images/cmr" / patient / name
    myo_path = root / "labels/cmr/cmr_Myo_mask" / patient / name
    image_raw, myo_raw = load(image_path), load(myo_path)
    rendered = []
    for index, issue in entries:
        image = stack_slice(image_raw, index)
        myo = stack_slice(myo_raw, index) > .5
        if myo.shape != image.shape:
            myo = np.zeros(image.shape, bool)
        stem = f"{Path(name).name[:-7]}_s{index + 1:02d}"
        output = out / f"{patient}_{stem}.png"
        title = f"cmr | {patient}/{name} | slice {index + 1}"
        draw_cmr_grid(image, myo, issue, title).save(output)
        rendered.append(((patient, name, index + 1), str(output)))
    return rendered


def render_lge_slice(task):
    root_s, out_s, patient, name, issue = task
    root, out = Path(root_s), Path(out_s)
    image_path = root / "images/lge" / patient / name
    myo_path = root / "labels/lge_checked/lge_Myo_labels" / patient / name
    mi_path = root / "labels/lge_checked/lge_MI_labels" / patient / name
    image = load(image_path)
    myo = load(myo_path) > .5 if myo_path.exists() else np.zeros(image.shape, bool)
    mi = load(mi_path) > .5 if mi_path.exists() else np.zeros(image.shape, bool)
    if myo.shape != image.shape:
        myo = np.zeros(image.shape, bool)
    if mi.shape != image.shape:
        mi = np.zeros(image.shape, bool)
    output = out / f"{patient}_{Path(name).name[:-7]}.png"
    draw_lge_grid(image, myo, mi, issue, f"lge_checked | {patient}/{name}").save(output)
    return (patient, name), str(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--modality", choices=["lge", "cmr"], required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = args.root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in out.glob("*.png"):
            p.unlink()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    rendered = {}
    tasks = []
    if args.modality == "cmr":
        grouped = {}
        for row in rows:
            key = (row["patient"], row["volume"])
            grouped.setdefault(key, {})[int(row["slice_index"])] = row.get("issues", "")
        tasks = [
            (str(root), str(out), patient, name,
             [(index - 1, issue) for index, issue in sorted(entries.items())])
            for (patient, name), entries in grouped.items()
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(render_cmr_volume, task) for task in tasks]
            for future in as_completed(futures):
                rendered.update(dict(future.result()))
        for row in rows:
            row["check"] = rendered.get((row["patient"], row["volume"], int(row["slice_index"])), "")
            if "check_view" in row:
                row["check_view"] = row["check"]
    else:
        unique = {(row["patient"], row["slice"]): row.get("issues", "") for row in rows}
        tasks = [(str(root), str(out), patient, name, issue)
                 for (patient, name), issue in unique.items()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(render_lge_slice, task) for task in tasks]
            for future in as_completed(futures):
                key, value = future.result()
                rendered[key] = value
        for row in rows:
            row["check"] = rendered.get((row["patient"], row["slice"]), "")
            if "check_view" in row:
                row["check_view"] = row["check"]
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["check"])
        writer.writeheader(); writer.writerows(rows)
    print({"modality": args.modality, "rows": len(rows), "rendered": len(rendered), "output": str(out)})


if __name__ == "__main__": main()
