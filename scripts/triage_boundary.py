#!/usr/bin/env python3
"""Boundary 池分诊：这个池按解剖位置筛出（心尖/心底、超出短轴范畴），
大多数切片最终应丢弃；自动化的目标不是"修 mask"，而是三分类——

  DISCARD  明确心尖/心底特征 → 自动进丢弃清单（批量确认后生效）
  SALVAGE  环形态完整、面积接近本例参考 → 捞回候选（优先人工快速过目）
  REVIEW   中间模糊带 → 仅此桶需要逐层人工复核

判据全部是确定性几何（面积相对本病例参考、血池洞、连通域），可审计可复跑。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.manual_annotation import candidate  # noqa: E402

CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def plane_features(plane: np.ndarray) -> dict:
    m = np.asarray(plane, dtype=bool)
    area = int(m.sum())
    if area == 0:
        return {"area": 0, "comps": 0, "hole": False, "hole_area": 0}
    _, comps = ndi.label(m, structure=CONN)
    holes = ndi.binary_fill_holes(m) & ~m
    hole_area = int(holes.sum())
    return {"area": area, "comps": int(comps), "hole": hole_area >= 20,
            "hole_area": hole_area}


def classify(feat: dict, ref: float) -> tuple:
    a = feat["area"]
    if ref <= 0:
        return "REVIEW", "no_reference"
    ratio = a / ref
    if a == 0 or ratio < 0.10:
        return "DISCARD", f"area {ratio:.0%} of patient reference"
    if not feat["hole"]:
        # 边界位置 + 无血池腔 = 心尖帽/心底切面或 mask 实心化，按池语义默认丢弃
        if ratio < 0.35:
            return "DISCARD", f"solid disc, area {ratio:.0%} (apex/base cap)"
        return "DISCARD", f"no cavity at boundary position, area {ratio:.0%} (cap or bloated mask)"
    if ratio >= 0.45 and feat["comps"] <= 2:
        return "SALVAGE", f"intact ring, area {ratio:.0%} of reference"
    return "REVIEW", f"ambiguous: area {ratio:.0%}, hole={feat['hole']}, comps={feat['comps']}"


import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_public_paths.DATASET_ROOT))
    ap.add_argument("--pool", default="cmr_boundary_manifest")
    ap.add_argument("--out", type=Path, default=Path("runtime/boundary_triage"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1) 枚举池行（便宜，不加载图像），按标签文件分组
    groups = defaultdict(list)
    idx = 0
    while True:
        if args.limit and idx >= args.limit:
            break
        try:
            row, ident = candidate(root, args.pool, idx)
        except Exception:
            break
        groups[(ident.get("patient"), ident.get("volume"))].append((idx, ident))
        idx += 1
    print(f"pool rows: {idx}, volumes: {len(groups)}", flush=True)

    import nibabel as nib
    from app.manual_annotation import _resolve_plane, _stack

    rows_out, buckets = [], {"DISCARD": 0, "SALVAGE": 0, "REVIEW": 0}
    done = 0
    for (patient, volume), items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        try:
            first_idx = items[0][0]
            _, label_path, _, raw, _, _, k0, _ = _resolve_plane(root, args.pool, first_idx, "myo", 0)
            stack, _ = _stack(raw)
        except Exception as exc:
            for i2, ident2 in items:
                rows_out.append({"index": i2, "patient": patient, "volume": volume,
                                 "bucket": "REVIEW", "reason": f"load_error: {exc}"[:100]})
                buckets["REVIEW"] += 1
            continue
        n_planes = stack.shape[-1] if stack.ndim == 3 else 1
        try:
            frame0 = int(items[0][1].get("frame", 0))
        except Exception:
            frame0 = 0
        offset = int(k0) - frame0
        feats = {}
        for kk in range(n_planes):
            plane = stack[..., kk] > 0.5 if stack.ndim == 3 else stack > 0.5
            feats[kk] = plane_features(plane)
        areas = sorted(f["area"] for f in feats.values() if f["area"] > 0)
        if areas:
            top = areas[-1]
            core = [a for a in areas if a >= 0.5 * top]
            ref = float(np.median(core)) if core else float(np.median(areas))
        else:
            ref = 0.0
        for row_idx, ident in items:
            try:
                k_use = (int(ident.get("frame", frame0)) + offset) % max(1, n_planes)
            except Exception:
                k_use = k0
            feat = feats.get(k_use, feats.get(k0))
            bucket, reason = classify(feat, ref)
            buckets[bucket] += 1
            rows_out.append({"index": row_idx, "patient": patient, "volume": volume,
                             "plane": k_use, "bucket": bucket, "reason": reason,
                             "area": feat["area"], "ref_area": round(ref, 1),
                             "hole": feat["hole"], "comps": feat["comps"]})
        done += 1
        if done % 25 == 0:
            print(f"  volumes {done}/{len(groups)} buckets={buckets}", flush=True)
            (args.out / "report.partial.json").write_text(json.dumps(
                {"pool": args.pool, "volumes_done": done, "volumes": len(groups),
                 "rows": len(rows_out), "buckets": buckets}, ensure_ascii=False))

    with (args.out / f"{args.pool}_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    report = {"pool": args.pool, "rows": len(rows_out), "volumes": len(groups),
              "buckets": buckets,
              "discard_rate": round(buckets["DISCARD"] / max(1, len(rows_out)), 3),
              "salvage_rate": round(buckets["SALVAGE"] / max(1, len(rows_out)), 3),
              "review_rate": round(buckets["REVIEW"] / max(1, len(rows_out)), 3),
              "by_patient_review": dict(sorted(
                  ((p, sum(1 for r in rows_out if r["patient"] == p and r["bucket"] == "REVIEW"))
                   for p in {r["patient"] for r in rows_out}), key=lambda kv: -kv[1])[:15])}
    (args.out / f"{args.pool}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
