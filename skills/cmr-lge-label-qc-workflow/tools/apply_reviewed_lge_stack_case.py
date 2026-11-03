#!/usr/bin/env python3
"""Apply an explicitly reviewed LGE stack correction from a JSON action spec.

This is a case-level tool, not an unrestricted matcher.  Every changed mask is
backed up and every operation is logged.  Supported actions are intentionally
small and topology-checked: clear labels, retain the Myo component nearest a
reference trajectory, retain the largest MI component, clear MI, fill an open
Myo ring from two adjacent supports, and lightly smooth a reviewed Myo ring.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def load_mask(path: Path):
    nii = nib.load(str(path))
    raw = np.asarray(nii.get_fdata())
    return nii, raw, np.squeeze(raw) > .5


def restore(mask, raw):
    out = mask.astype(np.uint8)
    return out.reshape(raw.shape) if out.size == raw.size else out


def components(mask):
    labels, count = ndi.label(mask, structure=STRUCTURE)
    items = []
    for index in range(1, count + 1):
        region = labels == index
        yy, xx = np.where(region)
        items.append((index, int(region.sum()), float(xx.mean()), float(yy.mean())))
    return labels, items


def largest(mask):
    labels, items = components(mask)
    if not items:
        return np.zeros_like(mask, dtype=bool)
    index = max(items, key=lambda item: item[1])[0]
    return labels == index


def topology(mask):
    _, comps = components(mask)
    holes = ndi.binary_fill_holes(mask) & ~mask
    _, hole_count = ndi.label(holes, structure=STRUCTURE)
    return len(comps), int(hole_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    args = ap.parse_args()
    root = args.root.resolve()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    patient = spec["patient"]
    myo_root = root / "labels/lge_checked/lge_Myo_labels" / patient
    mi_root = root / "labels/lge_checked/lge_MI_labels" / patient
    cache = {}

    def get(name):
        if name not in cache:
            myo_nii, myo_raw, myo = load_mask(myo_root / name)
            mi_nii, mi_raw, mi = load_mask(mi_root / name)
            if myo.shape != mi.shape:
                raise ValueError(f"shape mismatch for {name}: Myo={myo.shape}, MI={mi.shape}")
            cache[name] = {"myo_nii": myo_nii, "myo_raw": myo_raw, "myo": myo,
                           "mi_nii": mi_nii, "mi_raw": mi_raw, "mi": mi,
                           "changed": set()}
        return cache[name]

    records = []
    for action in spec["actions"]:
        name, operation = action["slice"], action["operation"]
        item = get(name)
        before_myo, before_mi = item["myo"].copy(), item["mi"].copy()
        evidence = ""
        if operation == "clear_both":
            item["myo"][:] = False
            item["mi"][:] = False
            item["changed"].update(("myo", "mi"))
        elif operation == "clear_mi":
            item["mi"][:] = False
            item["changed"].add("mi")
        elif operation == "keep_largest_mi":
            item["mi"] = largest(item["mi"])
            item["changed"].add("mi")
        elif operation == "keep_myo_nearest_reference":
            reference_centres = []
            for reference in action["references"]:
                reference_mask = largest(get(reference)["myo"])
                yy, xx = np.where(reference_mask)
                reference_centres.append((float(xx.mean()), float(yy.mean())))
            target = np.mean(np.asarray(reference_centres), axis=0)
            labels, comps = components(item["myo"])
            if not comps:
                raise ValueError(f"no Myo component in {name}")
            selected = min(comps, key=lambda c: np.hypot(c[2] - target[0], c[3] - target[1]))
            item["myo"] = labels == selected[0]
            item["mi"] &= item["myo"]
            item["changed"].update(("myo", "mi"))
            evidence = f"reference_centroid=({target[0]:.2f},{target[1]:.2f});selected_centroid=({selected[2]:.2f},{selected[3]:.2f})"
        elif operation == "fill_myo_adjacent_intersection":
            iterations = int(action.get("dilation_iterations", 1))
            low = ndi.binary_dilation(get(action["lower"])["myo"], iterations=iterations)
            high = ndi.binary_dilation(get(action["upper"])["myo"], iterations=iterations)
            proposed = item["myo"] | (low & high)
            added = int((proposed & ~item["myo"]).sum())
            comps, holes = topology(proposed)
            if added > int(action.get("max_added_pixels", 150)) or comps != 1 or holes != 1:
                raise ValueError(f"unsafe Myo fill for {name}: added={added}, components={comps}, holes={holes}")
            item["myo"] = proposed
            item["mi"] &= proposed
            item["changed"].update(("myo", "mi"))
            evidence = f"added={added};components={comps};holes={holes};dilation_iterations={iterations}"
        elif operation == "smooth_myo":
            closed = ndi.binary_closing(item["myo"], structure=STRUCTURE)
            proposed = ndi.binary_opening(closed, structure=STRUCTURE)
            changed = int(np.logical_xor(proposed, item["myo"]).sum())
            comps, holes = topology(proposed)
            if changed > int(action.get("max_changed_pixels", 20)) or comps != 1 or holes != 1:
                raise ValueError(f"unsafe Myo smoothing for {name}: changed={changed}, components={comps}, holes={holes}")
            item["myo"] = proposed
            item["mi"] &= proposed
            item["changed"].update(("myo", "mi"))
            evidence = f"changed={changed};components={comps};holes={holes}"
        else:
            raise ValueError(f"unsupported operation: {operation}")
        records.append({
            "slice": name, "operation": operation,
            "before_myo_pixels": int(before_myo.sum()), "after_myo_pixels": int(item["myo"].sum()),
            "before_mi_pixels": int(before_mi.sum()), "after_mi_pixels": int(item["mi"].sum()),
            "changed_myo_pixels": int(np.logical_xor(before_myo, item["myo"]).sum()),
            "changed_mi_pixels": int(np.logical_xor(before_mi, item["mi"]).sum()),
            "evidence": evidence,
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/reviewed_lge_stack_runs" / stamp
    backups = run / "backups"
    if args.mode == "apply":
        for name, item in cache.items():
            for label in sorted(item["changed"]):
                source = (myo_root if label == "myo" else mi_root) / name
                backup = backups / source.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup)
                nii = item[f"{label}_nii"]
                raw = item[f"{label}_raw"]
                out = nib.Nifti1Image(restore(item[label], raw), nii.affine, nii.header)
                out.set_data_dtype(np.uint8)
                nib.save(out, str(source))
    run.mkdir(parents=True, exist_ok=True)
    result = {"timestamp": stamp, "root": str(root), "patient": patient,
              "mode": args.mode, "spec": str(args.spec.resolve()), "records": records,
              "backup_root": str(backups), "status": "applied" if args.mode == "apply" else "candidate"}
    (run / "modifications.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
