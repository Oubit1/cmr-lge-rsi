#!/usr/bin/env python3
"""Remove small unsupported MI components that exceed casebook_qc's 20px threshold.

This script extends casebook_qc's conservative tiny-removal rule:
  - casebook_qc removes components <= 20px with no adjacent-layer support.
  - This script handles the 20 < area <= max_tiny_px range (default 40px),
    applying the same adjacency test with the same 15px distance criterion.

Always dry-run first. Records a backup and modifications.jsonl entry per edit.
Only operates on components=2 slices where the smaller component has no
adjacent-layer support. Does NOT touch components>=3 or large components.

Usage:
    python remove_small_unsupported_mi.py --root ROOT [--max-tiny-px 40] [--max-area-ratio 0.15] [--mode dry-run|apply-safe]
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


def load(path: Path):
    img = nib.load(str(path))
    return img, np.squeeze(img.get_fdata()) > 0.5


def sort_key(p: Path):
    stem = p.name[:-7]
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem)


def get_components(mask):
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        yy, xx = np.where(labels == k)
        if len(xx):
            out.append((k, len(xx), float(xx.mean()), float(yy.mean())))
    return labels, out


def nearby(mask, cx, cy, distance=15.0):
    _, comps = get_components(mask)
    return any(
        ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= distance
        for _, _, x, y in comps
    )


def run(root: Path, mode: str, max_tiny_px: int, max_area_ratio: float = 0.15):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = root / 'label_iteration' / 'casebook_runs' / f'extended_tiny_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_root = root / 'label_iteration' / 'casebook_backups' / f'extended_tiny_{stamp}'
    log_path = run_dir / 'modifications.jsonl'

    mi_root = root / 'labels/lge_checked/lge_MI_labels'
    myo_root = root / 'labels/lge_checked/lge_Myo_labels'

    rows = []
    modified = 0
    skipped_no_neighbour = 0

    for patient_dir in sorted(p for p in (root / 'images/lge').iterdir() if p.is_dir()):
        patient = patient_dir.name
        imgs = sorted((root / 'images/lge' / patient).glob('*.nii.gz'), key=sort_key)

        for idx, image in enumerate(imgs):
            mip = mi_root / patient / image.name
            myop = myo_root / patient / image.name
            if not mip.exists() or not myop.exists():
                continue

            _, mi = load(mip)
            _, myo = load(myop)
            labels, comps = get_components(mi)

            # Only handle exactly 2 components
            if len(comps) != 2:
                continue

            areas = sorted(comps, key=lambda c: c[1])
            small_k, small_area, small_cx, small_cy = areas[0]
            large_k, large_area, _, _ = areas[1]

            # Must be in the extended range (above casebook_qc threshold)
            if small_area <= 20 or small_area > max_tiny_px:
                continue

            # Skip if components are similarly sized (likely two real MI territories)
            area_ratio = small_area / max(large_area, 1)
            if area_ratio >= max_area_ratio:
                continue  # keep for manual review

            # Gather neighbour MI masks
            neighbours = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(imgs):
                    np_ = mi_root / patient / imgs[j].name
                    if np_.exists():
                        neighbours.append(load(np_)[1])

            if not neighbours:
                skipped_no_neighbour += 1
                continue

            # Only remove if NO neighbour has MI within 15px of small component centroid
            if any(nearby(n, small_cx, small_cy) for n in neighbours):
                continue  # supported — keep for manual review

            # Candidate for removal
            row = {
                'patient': patient,
                'slice': image.name,
                'layer_index': idx + 1,
                'small_component_area': small_area,
                'large_component_area': large_area,
                'area_ratio': round(small_area / max(large_area, 1), 4),
                'action': 'remove_small_unsupported',
                'status': 'planned' if mode == 'dry-run' else 'applied',
            }
            rows.append(row)

            if mode != 'apply-safe':
                continue

            # Apply: remove small component, keep large, clip to Myo
            refined = mi.copy()
            refined[labels == small_k] = False
            refined &= myo
            before, after = int(mi.sum()), int(refined.sum())

            backup = backup_root / mip.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mip, backup)

            src = nib.load(str(mip))
            out_img = nib.Nifti1Image(refined.astype(np.uint8), src.affine, src.header)
            out_img.set_data_dtype(np.uint8)
            nib.save(out_img, str(mip))
            modified += 1

            rec = {
                'timestamp': datetime.now().isoformat(),
                'patient': patient,
                'slice': image.name,
                'label': str(mip.relative_to(root)),
                'rule': f'remove_small_unsupported_mi (area={small_area}, threshold={max_tiny_px})',
                'action': 'remove_small_component',
                'before_pixels': before,
                'after_pixels': after,
                'changed_pixels': before - after,
                'backup': str(backup),
                'status': 'applied',
            }
            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Write CSV
    fields = ['patient', 'slice', 'layer_index', 'small_component_area',
              'large_component_area', 'area_ratio', 'action', 'status']
    with (run_dir / 'candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'root': str(root),
        'mode': mode,
        'run': str(run_dir),
        'max_tiny_px': max_tiny_px,
        'max_area_ratio': max_area_ratio,
        'candidates': len(rows),
        'modified_files': modified,
        'skipped_no_neighbour': skipped_no_neighbour,
        'rule': (
            f'components=2, small component area in (20, {max_tiny_px}], '
            f'area_ratio < {max_area_ratio}, '
            'no adjacent-layer MI within 15px → remove small component'
        ),
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--max-tiny-px', type=int, default=40,
                    help='Upper bound (inclusive) for "small" component area (default: 40)')
    ap.add_argument('--max-area-ratio', type=float, default=0.15,
                    help='Skip if small/large area ratio >= this (default: 0.15, i.e. skip near-equal pairs)')
    ap.add_argument('--mode', choices=['dry-run', 'apply-safe'], default='dry-run')
    args = ap.parse_args()
    run(args.root.resolve(), args.mode, args.max_tiny_px, args.max_area_ratio)


if __name__ == '__main__':
    main()
