#!/usr/bin/env python3
"""Match LGE labels to the accumulated casebook and apply only safe rules.

This is the single casebook-oriented entry point.  It is intentionally more
conservative than the historical patient-specific patches: those patches are
kept under legacy_case_patches/ for provenance and are never imported here.
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


def files(root: Path, patient: str):
    def sort_key(p):
        stem = p.name[:-7]
        try:
            return (0, float(stem))
        except ValueError:
            return (1, stem)
    return sorted((root / 'images/lge' / patient).glob('*.nii.gz'), key=sort_key)


def components(mask):
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        yy, xx = np.where(labels == k)
        if len(xx): out.append((k, len(xx), float(xx.mean()), float(yy.mean())))
    return labels, out


def nearby_component(mask, cx, cy, distance=15.0):
    _, comps = components(mask)
    return any(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= distance for _, _, x, y in comps)


def intensity_score(image: Path, component: np.ndarray, myo: np.ndarray) -> float:
    arr = np.squeeze(nib.load(str(image)).get_fdata()).astype(float)
    vals = arr[np.isfinite(arr) & myo]
    if not len(vals): return 0.0
    hi = float(np.percentile(vals, 95)); med = float(np.median(vals))
    comp_vals = arr[component]
    return float((np.mean(comp_vals) - med) / max(hi - med, 1e-6))


def audit(root: Path, mode: str):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = root / 'label_iteration/casebook_runs' / stamp
    run.mkdir(parents=True, exist_ok=True)
    mi_root = root / 'labels/lge_checked/lge_MI_labels'
    myo_root = root / 'labels/lge_checked/lge_Myo_labels'
    backup_root = root / 'label_iteration/casebook_backups' / stamp
    log_path = run / 'modifications.jsonl'
    rows = []
    modified = 0

    for patient_dir in sorted(p for p in (root / 'images/lge').iterdir() if p.is_dir()):
        patient = patient_dir.name; imgs = files(root, patient)
        for idx, image in enumerate(imgs):
            mip = mi_root / patient / image.name; myop = myo_root / patient / image.name
            if not mip.exists() or not myop.exists(): continue
            _, mi = load(mip); _, myo = load(myop)
            labels, comps = components(mi)
            unsupported = []
            neighbours = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(imgs):
                    p = mi_root / patient / imgs[j].name
                    if p.exists(): neighbours.append(load(p)[1])
            for k, area, cx, cy in comps:
                if area <= 20 and not any(nearby_component(n, cx, cy) for n in neighbours):
                    unsupported.append((k, area, intensity_score(image, labels == k, myo)))
            outside = int((mi & ~myo).sum())
            actions = []
            if unsupported: actions.append('remove_tiny_unsupported')
            if outside: actions.append('clip_mi_outside_myo')
            if len(comps) > 1 and not unsupported: actions.append('retain_or_merge_review')
            if actions:
                row = {'patient': patient, 'slice': image.name, 'layer_index': idx + 1,
                       'components': len(comps), 'tiny_unsupported': len(unsupported),
                       'outside_myo_pixels': outside,
                       'component_intensity_scores': json.dumps([round(x[2], 4) for x in unsupported]),
                       'recommended_action': ';'.join(actions), 'status': 'planned' if mode == 'dry-run' else 'applied'}
                rows.append(row)
            if mode != 'apply-safe' or (not unsupported and not outside): continue
            refined = mi.copy()
            for k, _, _ in unsupported: refined[labels == k] = False
            refined &= myo
            before, after = int(mi.sum()), int(refined.sum())
            backup = backup_root / mip.relative_to(root); backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mip, backup)
            src = nib.load(str(mip)); out = nib.Nifti1Image(refined.astype(np.uint8), src.affine, src.header)
            out.set_data_dtype(np.uint8); nib.save(out, str(mip)); modified += 1
            rec = {'timestamp': datetime.now().isoformat(), 'patient': patient, 'slice': image.name,
                   'label': str(mip.relative_to(root)), 'rule': ';'.join(actions),
                   'action': 'remove_tiny_unsupported_and_clip_outside_myo',
                   'before_pixels': before, 'after_pixels': after, 'changed_pixels': before - after,
                   'backup': str(backup), 'status': 'applied'}
            with log_path.open('a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    fields = ['patient', 'slice', 'layer_index', 'components', 'tiny_unsupported',
              'outside_myo_pixels', 'component_intensity_scores', 'recommended_action', 'status']
    with (run / 'casebook_match.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {'root': str(root), 'mode': mode, 'run': str(run), 'candidate_slices': len(rows), 'modified_files': modified,
               'rule': 'remove only tiny MI components unsupported by both adjacent layers; clip MI outside Myo; retain ambiguous merges for review'}
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--mode', choices=['dry-run', 'apply-safe'], default='dry-run')
    args = ap.parse_args(); audit(args.root.resolve(), args.mode)


if __name__ == '__main__': main()
