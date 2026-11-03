#!/usr/bin/env python3
"""Portable, conservative LGE label audit and safe-repair workflow.

Default mode is read-only.  The script deliberately leaves ambiguous cases in
the review manifest instead of guessing clinical labels.
"""
from __future__ import annotations

import argparse, csv, json, os, shutil
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS_DIR))
from cmr_qc import audit as audit_cmr
from render_check_panels import draw_lge_grid


def load_mask(path):
    nii = nib.load(str(path))
    arr = np.squeeze(nii.get_fdata())
    return nii, arr, arr > 0.5


def ordered(root, patient):
    # Some datasets use numeric slice names while others use names such as
    # ``patient001_slice01``.  Returning mixed float/string keys makes Python
    # 3 fail during sorting, so keep both branches in one comparable tuple.
    def sort_key(p):
        stem = p.name[:-7]
        try:
            return (0, float(stem))
        except ValueError:
            return (1, stem)
    return sorted((root / 'images/lge' / patient).glob('*.nii.gz'), key=sort_key)


def comps(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        yy, xx = np.where(lab == k)
        if len(xx): out.append((k, len(xx), float(xx.mean()), float(yy.mean())))
    return lab, out


def neighbour_masks(root, patient, files, index, label_root):
    out = []
    for j in (index - 1, index + 1):
        if 0 <= j < len(files):
            p = root / label_root / patient / files[j].name
            if p.exists(): out.append(load_mask(p)[2])
    return out


def has_nearby_centroid(mask, cx, cy, max_distance=15.0):
    _, items = comps(mask)
    return any(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= max_distance for _, _, x, y in items)


def check_png(image_path, myo, mi, out, tags, title):
    arr = np.squeeze(nib.load(str(image_path)).get_fdata())
    grid = draw_lge_grid(arr, myo, mi, ";".join(tags), title)
    grid.save(str(out))


def audit(root, run_dir, apply_safe=False, no_render=False):
    imgroot = root / 'images/lge'
    myoroot = root / 'labels/lge_checked/lge_Myo_labels'
    miroot = root / 'labels/lge_checked/lge_MI_labels'
    rows, summary = [], {'patients': 0, 'slices': 0, 'issues': 0, 'by_issue': {}}
    backup_root = root / 'label_iteration/backups' / run_dir.name
    log_path = root / 'label_iteration' / f'modifications_{run_dir.name}.jsonl'
    check_dir = run_dir / 'checks'; check_dir.mkdir(parents=True, exist_ok=True)
    public_check_dir = root / 'labels/check/lge_issue_slices'
    public_check_dir.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted(p for p in imgroot.iterdir() if p.is_dir())
    for patient_index, pd in enumerate(patient_dirs, start=1):
        if patient_index == 1 or patient_index % 25 == 0 or patient_index == len(patient_dirs):
            print(json.dumps({'progress': 'lge_qc', 'patient': patient_index,
                              'patients': len(patient_dirs), 'slices': summary['slices'],
                              'issues': len(rows)}, ensure_ascii=False), flush=True)
        summary['patients'] += 1
        files = ordered(root, pd.name)
        for idx, image in enumerate(files):
            summary['slices'] += 1
            myop = myoroot / pd.name / image.name; mip = miroot / pd.name / image.name
            tags, evidence = [], []
            if not myop.exists() or not mip.exists():
                tags.append('missing_label'); evidence.append('missing LGE Myo or MI')
                myo = np.zeros(np.squeeze(nib.load(str(image)).get_fdata()).shape, bool)
                mi = np.zeros_like(myo)
            else:
                myo = load_mask(myop)[2]; mi = load_mask(mip)[2]
                if myo.shape != mi.shape: tags.append('shape_or_affine_mismatch')
                lab_myo, cm = comps(myo)
                if len(cm) > 1: tags.append('myo_topology'); evidence.append(f'myo_components={len(cm)}')
                outside = int((mi & ~myo).sum())
                if outside: tags.append('mi_outside_myo'); evidence.append(f'outside_pixels={outside}')
                lab_mi, cmi = comps(mi)
                neigh = neighbour_masks(root, pd.name, files, idx, 'labels/lge_checked/lge_MI_labels')
                unsupported = []
                for k, area, cx, cy in cmi:
                    near = any(has_nearby_centroid(x, cx, cy) for x in neigh)
                    if area <= 20 and not near: unsupported.append((k, area))
                if unsupported: tags.append('mi_tiny_unsupported'); evidence.append(str(unsupported))
                if len(cmi) > 1: tags.append('mi_multiple_components'); evidence.append(f'mi_components={len(cmi)}')
                frac = float(mi.sum() / max(myo.sum(), 1))
                if frac >= 0.70: tags.append('mi_fraction_high'); evidence.append(f'mi_myo_ratio={frac:.3f}')
                if apply_safe and unsupported:
                    before = int(mi.sum()); new = mi.copy()
                    for k, _ in unsupported: new[lab_mi == k] = False
                    backup = backup_root / mip.relative_to(root); backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mip, backup)
                    nib.save(nib.Nifti1Image(new.astype(np.uint8), load_mask(mip)[0].affine, load_mask(mip)[0].header), str(mip))
                    rec = {'timestamp': datetime.now().isoformat(), 'patient': pd.name, 'slice': image.name, 'label': str(mip.relative_to(root)), 'rule': 'tiny_unsupported_mi', 'action': 'remove', 'before_pixels': before, 'after_pixels': int(new.sum()), 'changed_pixels': before-int(new.sum()), 'backup': str(backup), 'evidence': evidence, 'status': 'applied'}
                    with log_path.open('a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    mi = new
            if tags:
                stem = image.name[:-7] if image.name.endswith('.nii.gz') else image.name
                check_name = f'{pd.name}_{stem}.png'
                run_check = check_dir / check_name
                public_check = public_check_dir / check_name
                row = {'modality': 'lge', 'patient': pd.name, 'slice': image.name, 'layer_index': idx + 1, 'issues': ';'.join(tags), 'evidence': ';'.join(evidence), 'check': str(public_check)}
                rows.append(row)
                if not no_render:
                    check_png(image, myo, mi, run_check, tags, f'{pd.name} | layer {idx+1} | {";".join(tags)}')
                    shutil.copy2(run_check, public_check)
                    idx_link = public_check_dir / f'{pd.name}_{idx+1:02d}.png'
                    if idx_link.exists() or idx_link.is_symlink(): idx_link.unlink()
                    try:
                        os.link(public_check, idx_link)
                    except OSError:
                        pass
                for tag in tags: summary['by_issue'][tag] = summary['by_issue'].get(tag, 0) + 1
    summary['issues'] = len(rows)
    fields = ['modality', 'patient', 'slice', 'layer_index', 'issues', 'evidence', 'check']
    with (run_dir / 'audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    issue_dir = root / 'labels/check/lge_issue_slices'; issue_dir.mkdir(parents=True, exist_ok=True)
    with (issue_dir / 'issue_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--mode', choices=['audit', 'apply-safe'], default='audit')
    ap.add_argument('--no-render', action='store_true', help='Skip check PNG rendering for fast audit')
    ap.add_argument('--skip-cmr', action='store_true', help='Audit only LGE and do not launch the independent CMR audit')
    args = ap.parse_args(); root = args.root.resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = root / 'label_iteration/qc_runs' / stamp; run_dir.mkdir(parents=True, exist_ok=True)
    summary = audit(root, run_dir, args.mode == 'apply-safe', no_render=args.no_render)
    (run_dir / 'command.json').write_text(json.dumps({'root': str(root), 'mode': args.mode, 'timestamp': stamp}, indent=2) + '\n')
    cmr_summary = audit_cmr(root, args.mode, render_checks=not args.no_render) if ((root / 'images/cmr').is_dir() and not args.skip_cmr) else None
    print(json.dumps({'run_dir': str(run_dir), 'cmr_run': cmr_summary.get('run') if cmr_summary else None, **summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
