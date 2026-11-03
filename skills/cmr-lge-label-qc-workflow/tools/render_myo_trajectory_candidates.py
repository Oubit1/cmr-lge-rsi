#!/usr/bin/env python3
"""Render before/after views for myocardial trajectory QC candidates."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


RED = np.array([255, 40, 40], dtype=float)
GREEN = np.array([0, 220, 110], dtype=float)


def norm(image):
    arr = np.asarray(image, float); finite = np.isfinite(arr)
    lo, hi = np.percentile(arr[finite], [1, 99]) if finite.any() else (0, 1)
    gray = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def overlay(base, mask, colour):
    out = base.astype(float).copy(); out[mask] = .55 * out[mask] + .45 * colour
    return np.clip(out, 0, 255).astype(np.uint8)


def contour(mask):
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)))


def stack(raw):
    arr = np.squeeze(raw)
    if arr.ndim == 2: return arr[..., None]
    return np.moveaxis(arr, int(np.argmin(arr.shape)), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--clean", action="store_true", help="remove stale PNGs and manifest from output first")
    args = ap.parse_args(); root = args.root.resolve()
    if args.clean and args.output.exists():
        for path in args.output.iterdir():
            if path.is_file() and (path.suffix.lower() == ".png" or path.name == "issue_manifest.csv"):
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    count = 0; rendered_rows = []
    for row in rows:
        modality, patient, name = row["modality"], row["patient"], row["volume"]
        image_path = root / "images" / modality / patient / name
        label_path = (root / "labels/lge_checked/lge_Myo_labels" / patient / name if modality == "lge"
                      else root / "labels/cmr/cmr_Myo_mask" / patient / name)
        images = stack(nib.load(str(image_path)).get_fdata()); masks = stack(nib.load(str(label_path)).get_fdata() > .5)
        frame = int(row["frame_index"]) - 1; image = images[..., frame]; current = masks[..., frame]
        labels, _ = ndi.label(current, structure=np.ones((3, 3)))
        selected = row.get("selected_component", "")
        proposed = labels == int(selected) if selected else current
        base = norm(image); p0 = base; p1 = overlay(base, current, RED); p2 = overlay(base, proposed, GREEN)
        p3 = base.copy(); p3[contour(current)] = RED; p3[contour(proposed)] = GREEN
        h, w = image.shape; gap, header = 6, 48
        canvas = Image.new("RGB", (w * 4 + gap * 3, h + header), "white")
        title = (f"{modality} | {patient}/{name} | frame {frame + 1} | {row['decision']} | "
                 f"selected={row['selected_distance_px']}px second={row['second_distance_px']}px")
        draw = ImageDraw.Draw(canvas); draw.text((6, 6), title, fill="black")
        for i, panel in enumerate((p0, p1, p2, p3)):
            canvas.paste(Image.fromarray(panel), (i * (w + gap), header))
        stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
        target = args.output / f"{modality}_{patient}_{stem}_f{frame + 1:02d}.png"
        # Mark the patient-specific majority-trajectory centre on the final
        # panel without altering the mask visualization.
        ex, ey = float(row["expected_cx"]), float(row["expected_cy"])
        ox = 3 * (w + gap)
        draw.ellipse((ox + ex - 5, header + ey - 5, ox + ex + 5, header + ey + 5), outline=(255, 220, 0), width=2)
        draw.line((ox + ex - 7, header + ey, ox + ex + 7, header + ey), fill=(255, 220, 0), width=1)
        draw.line((ox + ex, header + ey - 7, ox + ex, header + ey + 7), fill=(255, 220, 0), width=1)
        canvas.save(target)
        rendered_rows.append({**row, "check": str(target)})
        count += 1
    if rendered_rows:
        fields = list(rendered_rows[0].keys())
        with (args.output / "issue_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rendered_rows)
    print({"rendered": count, "output": str(args.output)})


if __name__ == "__main__": main()
