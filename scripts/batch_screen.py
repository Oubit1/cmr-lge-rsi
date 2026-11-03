#!/usr/bin/env python3
"""Whole-pool batch screening with the validated rules_v2 (plane level).

This is the "auto-check every pending sample" path: for each slice in the pool,
run the validated conjunction rules; for the two classes proven safe on the
frozen evaluation set (fragmentation, temporal hole gap) build a corrected
candidate plane. Production labels are never touched - candidates and a report
are staged for batch review. The per-slice prompt agent in the editor is only
for long-tail cases this pass cannot express.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.manual_annotation import (  # noqa: E402
    _manifest,
    _paths,
    _row_identity,
    _stack,
    boundary_review_status,
)
from scripts.rules_v2 import decide  # noqa: E402

CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def fix_fragmentation(p):
    """Gate-aligned: two components may be legal LV+RV anatomy, so only
    slices with 3+ components are repairable, and the two largest must stay
    byte-identical (validate_batch_candidates enforces exactly this)."""
    labels, n = ndi.label(p, structure=CONN)
    if n < 3:
        return p, False
    sizes = np.array([int((labels == i).sum()) for i in range(1, n + 1)])
    order = np.argsort(sizes)[::-1]
    big = int(sizes[order[0]])
    out = p.copy()
    changed = False
    for rank, i in enumerate(order):
        if rank < 2:
            continue
        if int(sizes[i]) <= 0.18 * big:
            out[labels == int(i) + 1] = False
            changed = True
    return out, changed


import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_public_paths.DATASET_ROOT))
    ap.add_argument("--pool", default="cmr_middle_manifest")
    ap.add_argument("--review-status", choices=("pending", "work", "trash", "checked", "all"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runtime/batch_candidates"))
    args = ap.parse_args()
    root = Path(args.dataset)
    out_root = args.out / args.pool
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"pool": args.pool, "scanned": 0, "flagged": 0, "auto_fixed": 0,
              "by_flag": {}, "changed_pixels_total": 0, "records": []}
    rows = _manifest(root, args.pool)
    review_status = args.review_status
    if args.pool == "cmr_boundary_manifest" and review_status is None:
        review_status = "work"
    report["review_status"] = review_status or "all"
    source_cache: dict[str, np.ndarray] = {}
    for idx, row in enumerate(rows):
        if args.limit and report["scanned"] >= args.limit:
            break
        ident = _row_identity(args.pool, row)
        if (args.pool == "cmr_boundary_manifest" and review_status != "all"
                and boundary_review_status(root, ident) != review_status):
            continue
        try:
            _, label_path = _paths(root, ident, "myo")
            path = str(label_path)
            if path not in source_cache:
                import nibabel as nib
                source_cache[path], _ = _stack(np.asanyarray(nib.load(path).dataobj))
            stack = source_cache[path]
            k = max(0, int(ident["frame"]) - 1)
            if not 0 <= k < stack.shape[-1]:
                continue
            cur = stack[..., k] > .5
            prev = stack[..., k - 1] > .5 if k > 0 else None
            nxt = stack[..., k + 1] > .5 if k + 1 < stack.shape[-1] else None
        except Exception:
            continue
        frames = np.stack([f if f is not None else cur for f in (prev, cur, nxt)])
        try:
            flags = decide(frames)
        except Exception:
            flags = {}
        report["scanned"] += 1
        if not flags:
            continue
        report["flagged"] += 1
        for f in flags:
            report["by_flag"][f] = report["by_flag"].get(f, 0) + 1
        fixed = cur.copy()
        applied = []
        if "fragmentation" in flags:
            fixed, ch = fix_fragmentation(fixed)
            if ch:
                applied.append("remove_small_components")
        if "hole_missing" in flags and prev is not None and nxt is not None:
            # hole_missing = 本层血池洞丢失（腔被误实心化）；修复是"挖回"洞，
            # 且必须相邻两层在同一位置都有洞（双侧支持，与 skill 规则一致）
            hp = ndi.binary_fill_holes(prev) & ~prev
            hn = ndi.binary_fill_holes(nxt) & ~nxt
            both = hp & hn
            carve = both & fixed
            if int(carve.sum()) >= 20:
                fixed = fixed & ~both
                applied.append("restore_temporal_hole")
        changed = int((fixed != cur).sum())
        entry = {"index": idx, "patient": ident.get("patient"), "volume": ident.get("volume"),
                 "frame": ident.get("frame"), "flags": sorted(flags), "auto_fix": applied,
                 "changed_pixels": changed, "source_label": path}
        if applied and changed > 0:
            dst = out_root / f"{idx:06d}_{ident.get('patient')}_k{k}.npz"
            np.savez_compressed(dst, original=cur.astype(np.uint8), fixed=fixed.astype(np.uint8))
            entry["candidate"] = str(dst)
            report["auto_fixed"] += 1
            report["changed_pixels_total"] += changed
        report["records"].append(entry)
        if report["scanned"] % 100 == 0:
            print(f"  scanned {report['scanned']} flagged={report['flagged']} fixed={report['auto_fixed']}", flush=True)
            (args.out / f"{args.pool}_report.partial.json").write_text(
                json.dumps({k2: v for k2, v in report.items() if k2 != "records"}, ensure_ascii=False))
    report_path = args.out / f"{args.pool}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / f"{args.pool}_report.partial.json").unlink(missing_ok=True)
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
