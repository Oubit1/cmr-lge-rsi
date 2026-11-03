#!/usr/bin/env python3
"""Cross-modality myocardial position QC from the patient's majority trajectory.

The same rule is used for LGE spatial layers and CMR spatial-layer/cine-frame
units.  Trusted single-ring masks define a robust patient-specific LV centre
trajectory.  A multi-component mask is auto-correctable only when one ring is
close to several consistent trusted references, competing components are
clearly farther away, and the competing location is unsupported locally.

Single-component centroid outliers are reported only: this tool never shifts
or invents a mask.  CMR patients with persistently multi-ring anatomy are also
left unchanged because the majority single-ring evidence is insufficient.
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


STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def numeric_key(path: Path):
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return 0, float(stem)
    except ValueError:
        return 1, stem


def as_stack(raw):
    arr = np.squeeze(raw)
    if arr.ndim == 2:
        return arr[..., None], None
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1), axis


def restore(stack, raw, axis):
    out = stack[..., 0] if axis is None else np.moveaxis(stack, -1, axis)
    return out.reshape(raw.shape)


def component_stats(mask, min_pixels=30, min_ratio=.08):
    labels, count = ndi.label(mask, structure=STRUCTURE)
    raw = []
    for index in range(1, count + 1):
        region = labels == index
        area = int(region.sum())
        if not area:
            continue
        yy, xx = np.where(region)
        holes = ndi.binary_fill_holes(region) & ~region
        hole_count = int(ndi.label(holes, structure=STRUCTURE)[1])
        border = bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any())
        raw.append({"id": index, "area": area, "cx": float(xx.mean()), "cy": float(yy.mean()),
                    "nx": float(xx.mean() / mask.shape[1]), "ny": float(yy.mean() / mask.shape[0]),
                    "holes": hole_count, "border": border})
    if not raw:
        return labels, []
    largest = max(item["area"] for item in raw)
    major = [item for item in raw if item["area"] >= max(min_pixels, largest * min_ratio)]
    return labels, sorted(major, key=lambda item: item["area"], reverse=True)


def distance_px(a, b, shape):
    return float(np.hypot((a[0] - b[0]) * shape[1], (a[1] - b[1]) * shape[0]))


def describe(mask, spatial, frame, n_frames, name):
    _, comps = component_stats(mask)
    trusted = len(comps) == 1 and comps[0]["holes"] == 1 and not comps[0]["border"]
    centre = (comps[0]["nx"], comps[0]["ny"]) if trusted else None
    return {"spatial": spatial, "frame": frame, "n_frames": n_frames, "name": name,
            "shape": mask.shape, "components": comps, "trusted": trusted, "centre": centre,
            "nonempty": bool(mask.any())}


def unit_distance(a, b, modality):
    spatial = abs(a["spatial"] - b["spatial"])
    if modality == "lge":
        return float(spatial)
    if a["spatial"] == b["spatial"]:
        n = max(a["n_frames"], b["n_frames"], 1)
        temporal = min(abs(a["frame"] - b["frame"]), n - abs(a["frame"] - b["frame"]))
        return temporal * .35
    return spatial * 4.0 + abs(a["frame"] - b["frame"]) * .2


def local_support(units, index, modality, max_refs=6):
    target = units[index]
    refs = [(unit_distance(target, other, modality), other) for j, other in enumerate(units)
            if j != index and other["trusted"]]
    refs.sort(key=lambda item: item[0])
    refs = [item[1] for item in refs[:max_refs]]
    if len(refs) < 3:
        return None
    centres = np.asarray([ref["centre"] for ref in refs], dtype=float)
    expected = tuple(np.median(centres, axis=0))
    spread = max(distance_px(tuple(c), expected, target["shape"]) for c in centres)
    return expected, spread, refs


def nearby_extra_support(units, index, extra_centres, modality, radius_px=12.0):
    target = units[index]
    neighbours = []
    for j, other in enumerate(units):
        if j == index or not other["nonempty"]:
            continue
        if modality == "lge":
            close = abs(other["spatial"] - target["spatial"]) <= 1
        else:
            same_volume = other["spatial"] == target["spatial"]
            temporal = abs(other["frame"] - target["frame"])
            temporal = min(temporal, target["n_frames"] - temporal)
            adjacent_volume = abs(other["spatial"] - target["spatial"]) == 1 and other["frame"] == target["frame"]
            close = (same_volume and temporal <= 2) or adjacent_volume
        if close:
            neighbours.append(other)
    supported = 0
    for other in neighbours:
        if any(distance_px((comp["nx"], comp["ny"]), centre, target["shape"]) <= radius_px
               for comp in other["components"] for centre in extra_centres):
            supported += 1
    return supported, len(neighbours)


def load_patient(root, patient, modality):
    label_dir = (root / "labels/lge_checked/lge_Myo_labels" / patient if modality == "lge"
                 else root / "labels/cmr/cmr_Myo_mask" / patient)
    paths = sorted(label_dir.glob("*.nii.gz"), key=numeric_key)
    volumes, units = [], []
    for spatial, path in enumerate(paths):
        nii = nib.load(str(path)); raw = np.asarray(nii.get_fdata())
        stack, axis = as_stack(raw > .5)
        volume = {"path": path, "nii": nii, "raw": raw, "stack": stack.copy(), "axis": axis, "changed": False}
        volumes.append(volume)
        for frame in range(stack.shape[-1]):
            unit = describe(stack[..., frame], spatial, frame, stack.shape[-1], path.name)
            unit["volume"] = spatial
            units.append(unit)
    return volumes, units


def process_patient(task):
    root_s, patient, modality, mode, run_s, thresholds = task
    root, run = Path(root_s), Path(run_s)
    volumes, units = load_patient(root, patient, modality)
    if not units:
        return [], 0
    nonempty = [u for u in units if u["nonempty"]]
    trusted = [u for u in units if u["trusted"]]
    trusted_fraction = len(trusted) / max(len(nonempty), 1)
    records, changed = [], 0
    n_spatial = len(volumes)
    for index, unit in enumerate(units):
        if not unit["nonempty"] or unit["spatial"] < 2 or unit["spatial"] >= n_spatial - 2:
            continue
        support = local_support(units, index, modality)
        if support is None:
            continue
        expected, spread, refs = support
        comps = unit["components"]
        distances = sorted([(distance_px((comp["nx"], comp["ny"]), expected, unit["shape"]), comp)
                            for comp in comps], key=lambda item: item[0])
        base = {"modality": modality, "patient": patient, "volume": unit["name"],
                "spatial_index": unit["spatial"] + 1, "frame_index": unit["frame"] + 1,
                "trusted_fraction": round(trusted_fraction, 4), "reference_count": len(refs),
                "reference_spread_px": round(spread, 3),
                "expected_cx": round(expected[0] * unit["shape"][1], 2),
                "expected_cy": round(expected[1] * unit["shape"][0], 2)}
        if len(comps) == 1:
            if distances and distances[0][0] > thresholds["centroid_review_px"] and spread <= thresholds["max_reference_spread_px"]:
                records.append({**base, "decision": "review_centroid_outlier", "status": "unresolved",
                                "selected_component": comps[0]["id"], "selected_distance_px": round(distances[0][0], 3),
                                "second_distance_px": "", "removed_pixels": 0,
                                "evidence": "single component is off the majority trajectory; no mask invented"})
            continue
        if len(comps) < 2 or trusted_fraction < thresholds["min_trusted_fraction"] or spread > thresholds["max_reference_spread_px"]:
            continue
        selected_distance, selected = distances[0]
        second_distance = distances[1][0]
        if selected["holes"] != 1 or selected["border"]:
            continue
        if selected_distance > thresholds["max_selected_distance_px"] or second_distance - selected_distance < thresholds["min_margin_px"]:
            records.append({**base, "decision": "review_ambiguous_components", "status": "unresolved",
                            "selected_component": selected["id"], "selected_distance_px": round(selected_distance, 3),
                            "second_distance_px": round(second_distance, 3), "removed_pixels": 0,
                            "evidence": "component competition lacks sufficient trajectory margin"})
            continue
        extra = [(comp["nx"], comp["ny"]) for _, comp in distances[1:]]
        extra_supported, neighbour_count = nearby_extra_support(units, index, extra, modality)
        if extra_supported >= thresholds["max_extra_support_count"]:
            records.append({**base, "decision": "review_supported_extra_component", "status": "unresolved",
                            "selected_component": selected["id"], "selected_distance_px": round(selected_distance, 3),
                            "second_distance_px": round(second_distance, 3), "removed_pixels": 0,
                            "evidence": f"competing component supported in {extra_supported}/{neighbour_count} local units"})
            continue
        volume = volumes[unit["volume"]]
        mask = volume["stack"][..., unit["frame"]]
        labels, _ = component_stats(mask)
        proposed = labels == selected["id"]
        removed = int((mask & ~proposed).sum())
        record = {**base, "decision": "keep_component_on_majority_lv_trajectory",
                  "status": "applied" if mode == "apply" else "candidate",
                  "selected_component": selected["id"], "selected_distance_px": round(selected_distance, 3),
                  "second_distance_px": round(second_distance, 3), "removed_pixels": removed,
                  "evidence": f"extra_support={extra_supported}/{neighbour_count};selected_area={selected['area']}"}
        records.append(record)
        if mode == "apply":
            volume["stack"][..., unit["frame"]] = proposed
            volume["changed"] = True
            changed += 1
            if modality == "lge":
                mi_path = root / "labels/lge_checked/lge_MI_labels" / patient / unit["name"]
                if mi_path.exists():
                    mi_nii = nib.load(str(mi_path)); mi_raw = np.asarray(mi_nii.get_fdata()); mi = np.squeeze(mi_raw) > .5
                    clipped = mi & proposed
                    backup = run / "backups" / mi_path.relative_to(root)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mi_path, backup)
                    out = clipped.astype(np.uint8).reshape(mi_raw.shape)
                    saved = nib.Nifti1Image(out, mi_nii.affine, mi_nii.header); saved.set_data_dtype(np.uint8)
                    nib.save(saved, str(mi_path))
    if mode == "apply":
        for volume in volumes:
            if not volume["changed"]:
                continue
            path = volume["path"]
            backup = run / "backups" / path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            out_raw = restore(volume["stack"].astype(np.uint8), volume["raw"], volume["axis"])
            saved = nib.Nifti1Image(out_raw, volume["nii"].affine, volume["nii"].header); saved.set_data_dtype(np.uint8)
            nib.save(saved, str(path))
    return records, changed


def write_csv(path, rows):
    fields = ["modality", "patient", "volume", "spatial_index", "frame_index", "decision", "status",
              "trusted_fraction", "reference_count", "reference_spread_px", "expected_cx", "expected_cy",
              "selected_component", "selected_distance_px", "second_distance_px", "removed_pixels", "evidence"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--modality", choices=("lge", "cmr", "both"), default="both")
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patient")
    ap.add_argument("--min-trusted-fraction", type=float, default=.55)
    ap.add_argument("--max-reference-spread-px", type=float, default=12.0)
    ap.add_argument("--max-selected-distance-px", type=float, default=12.0)
    ap.add_argument("--min-margin-px", type=float, default=10.0)
    ap.add_argument("--centroid-review-px", type=float, default=22.0)
    ap.add_argument("--max-extra-support-count", type=int, default=2)
    args = ap.parse_args()
    root = args.root.resolve(); stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/myo_trajectory_runs" / stamp; run.mkdir(parents=True, exist_ok=True)
    thresholds = {"min_trusted_fraction": args.min_trusted_fraction,
                  "max_reference_spread_px": args.max_reference_spread_px,
                  "max_selected_distance_px": args.max_selected_distance_px,
                  "min_margin_px": args.min_margin_px,
                  "centroid_review_px": args.centroid_review_px,
                  "max_extra_support_count": args.max_extra_support_count}
    modalities = ("lge", "cmr") if args.modality == "both" else (args.modality,)
    tasks = []
    for modality in modalities:
        label_root = (root / "labels/lge_checked/lge_Myo_labels" if modality == "lge"
                      else root / "labels/cmr/cmr_Myo_mask")
        patients = sorted(p.name for p in label_root.iterdir() if p.is_dir())
        if args.patient:
            patients = [p for p in patients if p == args.patient]
        tasks.extend((str(root), patient, modality, args.mode, str(run), thresholds) for patient in patients)
    rows, changed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_patient, task) for task in tasks]
        for future in as_completed(futures):
            found, count = future.result(); rows.extend(found); changed += count
    rows.sort(key=lambda row: (row["modality"], row["patient"], row["spatial_index"], row["frame_index"]))
    write_csv(run / "trajectory_candidates.csv", rows)
    summary = {"root": str(root), "mode": args.mode, "modalities": list(modalities),
               "patients_scanned": len(tasks), "candidate_actions": sum(r["status"] in ("candidate", "applied") for r in rows),
               "review_only": sum(r["status"] == "unresolved" for r in rows), "changed_units": changed,
               "by_modality": {m: {"rows": sum(r["modality"] == m for r in rows),
                                      "candidate_actions": sum(r["modality"] == m and r["status"] in ("candidate", "applied") for r in rows),
                                      "review_only": sum(r["modality"] == m and r["status"] == "unresolved" for r in rows)} for m in modalities},
               "thresholds": thresholds, "run": str(run)}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
