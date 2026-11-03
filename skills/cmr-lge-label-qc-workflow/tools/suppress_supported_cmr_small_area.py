#!/usr/bin/env python3
"""Suppress CMR small-area-only QC false positives with strong support.

The tool never edits a mask.  It accepts a candidate without change only when
the current cine frame and its temporal neighbours form a valid single ring,
the area/centroid trajectory is smooth, most frames in the volume have the
same topology, and the paired LGE Myo is also a valid single ring.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def load_stack(path: Path):
    arr = np.squeeze(nib.load(str(path)).get_fdata()) > 0.5
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"unsupported mask shape: {arr.shape}")
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, -1)


def topology(mask):
    structure = np.ones((3, 3), dtype=np.uint8)
    components = int(ndi.label(mask, structure=structure)[1])
    holes = ndi.binary_fill_holes(mask) & ~mask
    hole_count = int(ndi.label(holes, structure=structure)[1])
    border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    yy, xx = np.where(mask)
    centroid = (float(xx.mean()), float(yy.mean())) if len(xx) else (np.nan, np.nan)
    return int(mask.sum()), components, hole_count, border, centroid


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    ap.add_argument("--min-valid-frame-fraction", type=float, default=0.90)
    ap.add_argument("--min-area-fraction", type=float, default=0.004)
    ap.add_argument("--min-local-area-ratio", type=float, default=0.75)
    ap.add_argument("--max-local-area-ratio", type=float, default=1.33)
    ap.add_argument("--max-centroid-drift", type=float, default=8.0)
    args = ap.parse_args()

    root = args.root.resolve()
    issue_dir = root / "labels/check/cmr_middle_issue_slices"
    manifest = issue_dir / "issue_manifest.csv"
    rows = read_csv(manifest)
    fields = list(rows[0].keys()) if rows else ["patient", "volume", "slice_index", "issues", "evidence", "check", "check_view"]
    cache = {}
    accepted, retained = [], []

    for row in rows:
        if row["issues"] != "cmr_myo_small_area":
            retained.append(row)
            continue
        key = (row["patient"], row["volume"])
        if key not in cache:
            cmr_path = root / "labels/cmr/cmr_Myo_mask" / key[0] / key[1]
            lge_path = root / "labels/lge_checked/lge_Myo_labels" / key[0] / key[1]
            if not cmr_path.exists() or not lge_path.exists():
                cache[key] = None
            else:
                stack = load_stack(cmr_path)
                lge = load_stack(lge_path)[..., 0]
                frame_stats = [topology(stack[..., k]) for k in range(stack.shape[-1])]
                lge_stats = topology(lge)
                cache[key] = (stack.shape, frame_stats, lge_stats)
        info = cache[key]
        if info is None:
            retained.append(row)
            continue
        shape, stats, lge_stats = info
        k = int(row["slice_index"]) - 1
        n_frames = len(stats)
        if n_frames < 3 or k < 0 or k >= n_frames:
            retained.append(row)
            continue
        p_k = (k - 1) % n_frames
        n_k = (k + 1) % n_frames
        area, comp, holes, border, center = stats[k]
        p_area, p_comp, p_holes, p_border, p_center = stats[p_k]
        n_area, n_comp, n_holes, n_border, n_center = stats[n_k]
        valid_frames = sum(c == 1 and h == 1 and not b for _, c, h, b, _ in stats)
        local_mean = (p_area + n_area) / 2.0
        ratio = area / max(local_mean, 1.0)
        area_fraction = area / float(shape[0] * shape[1])
        lge_area, lge_comp, lge_holes, lge_border, _ = lge_stats
        accepted_now = (
            comp == holes == 1 and not border
            and p_comp == p_holes == 1 and not p_border
            and n_comp == n_holes == 1 and not n_border
            and valid_frames / len(stats) >= args.min_valid_frame_fraction
            and area_fraction >= args.min_area_fraction
            and args.min_local_area_ratio <= ratio <= args.max_local_area_ratio
            and max(distance(center, p_center), distance(center, n_center)) <= args.max_centroid_drift
            and lge_area > 0 and lge_comp == lge_holes == 1 and not lge_border
        )
        if not accepted_now:
            retained.append(row)
            continue
        accepted.append({
            **row,
            "decision": "accepted_no_change",
            "rule": "small_area_single_ring_temporally_supported_with_lge",
            "decision_evidence": (
                f"area={area};area_fraction={area_fraction:.4f};local_ratio={ratio:.3f};"
                f"valid_frames={valid_frames}/{len(stats)};max_centroid_drift="
                f"{max(distance(center, p_center), distance(center, n_center)):.2f};"
                f"lge_area={lge_area};lge_components={lge_comp};lge_holes={lge_holes}"
            ),
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / "label_iteration/cmr_small_area_accept_runs" / stamp
    accepted_fields = fields + ["decision", "rule", "decision_evidence"]
    write_csv(run / "accepted.csv", accepted, accepted_fields)
    summary = {
        "root": str(root), "mode": args.mode, "input_rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_unique_slices": len({r["check"] for r in accepted}),
        "accepted_patients": len({r["patient"] for r in accepted}),
        "retained_rows": len(retained),
        "retained_unique_slices": len({r["check"] for r in retained}),
        "rule": "single ring + one blood-pool hole + temporal continuity + paired LGE single-ring support",
        "run": str(run),
    }
    if args.mode == "apply":
        write_csv(manifest, retained, fields)
        retained_checks = {r["check"] for r in retained}
        for row in accepted:
            check = Path(row["check"])
            if row["check"] not in retained_checks and check.parent == issue_dir and check.exists():
                check.unlink()
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
