#!/usr/bin/env python3
"""Slice-level boundary triage, keyed by the finalization bundle relative_key.

Input : dataset_finalization run 的 quarantine_candidates.csv（800 对）
Output: 每个 relative_key 一个三桶判定 + 可直接喂给
        tools/finalize_paired_dataset.py --reject-manifest 的 DISCARD 清单。
判据（切片级，跨帧聚合，参考=同患者核心栈中位面积）：
  DISCARD  med_area < 10% ref  或  cavity_frac < 0.2（几乎所有帧无血池腔=心尖帽/实心）
  SALVAGE  cavity_frac >= 0.6  且  med_area >= 45% ref（环形态完整，值得捞回）
  REVIEW   其余
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def slice_features(path: Path) -> tuple[float, float] | None:
    try:
        data = np.asanyarray(nib.load(str(path)).dataobj) > 0
    except Exception:
        return None
    if data.ndim == 2:
        data = data[None]
    areas, cavities = [], 0
    n = data.shape[-1] if data.shape[-1] <= data.shape[0] else data.shape[0]
    planes = [data[..., k] for k in range(data.shape[-1])] if data.ndim == 3 else [data[0]]
    for pl in planes:
        a = int(pl.sum())
        areas.append(a)
        if a >= 20:
            holes = ndi.binary_fill_holes(pl) & ~pl
            if int(holes.sum()) >= 20:
                cavities += 1
    if not areas:
        return None
    return float(np.median(areas)), cavities / len(areas)


def natkey(p: Path):
    m = re.match(r"(-?\d+(?:\.\d+)?)", p.name)
    return (float(m.group(1)) if m else 0.0, p.name)


import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=_public_paths.DATASET_ROOT)
    ap.add_argument("--candidates", type=Path,
                    default=Path("/srv/rsi/dataset/label_iteration/"
                                 "dataset_finalization/20261022_095340/quarantine_candidates.csv"))
    ap.add_argument("--out", type=Path, default=Path("runtime/boundary_triage"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.candidates.read_text(encoding="utf-8-sig").splitlines()))
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["relative_key"].split("/")[0]].append(r["relative_key"])
    print(f"candidates={len(rows)} patients={len(by_patient)}", flush=True)

    myo_root = args.dataset / "labels/cmr_checked/cmr_Myo_mask"
    out_rows, buckets = [], defaultdict(int)
    ref_cache: dict[str, float] = {}
    for patient, keys in sorted(by_patient.items()):
        # 患者参考：取该患者全部切片按序号排序的中段最多 12 个
        all_files = sorted((myo_root / patient).glob("*.nii.gz"), key=natkey) if (myo_root / patient).is_dir() else []
        if patient not in ref_cache:
            mid = all_files[len(all_files) // 4: len(all_files) * 3 // 4][:12] or all_files[:12]
            vals = []
            for f in mid:
                feat = slice_features(f)
                if feat and feat[0] > 0:
                    vals.append(feat[0])
            ref_cache[patient] = float(np.median(vals)) if vals else 0.0
        ref = ref_cache[patient]
        for key in keys:
            feat = slice_features(myo_root / patient / Path(key).name)
            if feat is None:
                bucket, reason, med, cav = "REVIEW", "label_load_failed", 0.0, 0.0
            else:
                med, cav = feat
                ratio = med / ref if ref > 0 else 0.0
                if ref <= 0 or med < 0.10 * ref:
                    bucket, reason = "DISCARD", f"med_area {ratio:.0%} of patient ref"
                elif cav < 0.2:
                    bucket, reason = "DISCARD", f"cavity_frac {cav:.0%} (apex/base cap or solid mask)"
                elif cav >= 0.6 and med >= 0.45 * ref:
                    bucket, reason = "SALVAGE", f"intact ring: cavity {cav:.0%}, area {ratio:.0%}"
                else:
                    bucket, reason = "REVIEW", f"ambiguous: cavity {cav:.0%}, area {ratio:.0%}"
            buckets[bucket] += 1
            out_rows.append({"relative_key": key, "patient": patient, "bucket": bucket,
                             "reason": reason, "med_area": round(med, 1),
                             "ref_area": round(ref, 1), "cavity_frac": round(cav, 2)})
        print(f"  {patient}: {len(keys)} keys, buckets={dict(buckets)}", flush=True)

    n = len(out_rows)
    with (args.out / "slice_triage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    with (args.out / "discard_reject_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_key", "status", "reason"])
        for r in out_rows:
            if r["bucket"] == "DISCARD":
                w.writerow([r["relative_key"], "ready", "triage_discard:" + r["reason"].split(" ")[0]])
    report = {"keys": n, "patients": len(by_patient), "buckets": dict(buckets),
              "discard_rate": round(buckets["DISCARD"] / max(1, n), 3),
              "salvage_rate": round(buckets["SALVAGE"] / max(1, n), 3),
              "review_rate": round(buckets["REVIEW"] / max(1, n), 3),
              "unit": "spatial slice (relative_key)",
              "candidates_source": str(args.candidates)}
    (args.out / "slice_triage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
