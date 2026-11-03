#!/usr/bin/env python3
"""晋升执行器：算法判定可到 checked 的条目批量晋升。

默认 --dry-run（只输出计划）；加 --execute 才动文件。
 ① direct 118 : PNG 待处理 -> labels/check/checked/<mod>/<patient>/<volume>.png
 ② fix     26 : gate 批准候选（npz fixed 平面）写入
                labels/cmr_checked/cmr_Myo_mask/<patient>/<volume>.nii.gz 的 k 平面
                写前备份 + sha256 + 审计日志；改动 0 或形状不符则跳过。
全程幂等可审计：runtime/promotion/promotion_log.jsonl 每步一行。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.manual_annotation import _restore, _stack  # noqa: E402

import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
DATASET = _public_paths.DATASET_ROOT
CHECK = DATASET / "labels/check"
LOG = ROOT / "runtime/promotion/promotion_log.jsonl"
BACKUP_ROOT = DATASET / "label_iteration/promotion_backups"
TOTAL = {"direct": 0, "fix": 0, "skip": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _triage_buckets() -> dict:
    out = {}
    p = ROOT / "runtime/boundary_triage/slice_triage.csv"
    if p.exists():
        for r in csv.DictReader(p.read_text(encoding="utf-8").splitlines()):
            out[r["relative_key"]] = r["bucket"]
    return out


def _gate_passed() -> dict:
    """relative_key -> list[fix]（来自 candidate_validation.json + 池 manifest 行映射）"""
    out: dict = {}
    val = json.loads((ROOT / "runtime/batch_candidates/candidate_validation.json").read_text(encoding="utf-8"))
    for name in ("cmr_boundary_manifest", "cmr_middle_manifest"):
        mp = (CHECK / "manifest_backup" / f"{name}.csv")
        if not mp.exists():
            continue
        rows = list(csv.DictReader(mp.read_text(encoding="utf-8-sig").splitlines()))
        for i, r in enumerate(rows):
            key = f"{r['patient']}/{r['volume']}"
            out.setdefault(key, [])
        for rec in val.get("records", []):
            if rec.get("pool") != name or not rec.get("passed"):
                continue
            idx = rec.get("index")
            if idx is not None and 0 <= idx < len(rows):
                key = f"{rows[idx]['patient']}/{rows[idx]['volume']}"
                out.setdefault(key, []).append({"pool": name, "index": idx, "ops": rec.get("operation", [])})
    return out


def _find_npz(pool: str, index: int, patient: str) -> Path | None:
    hits = list((ROOT / "runtime/batch_candidates" / pool).glob(f"{int(index):06d}_{patient}_k*.npz"))
    return hits[0] if hits else None


def run(args) -> None:
    TOTAL.update(direct=0, fix=0, skip=0)
    buckets = _triage_buckets()
    gate = _gate_passed()
    logf = open(LOG, "a", encoding="utf-8") if args.execute else None
    do = args.execute

    for state in ("待处理", "待筛"):
        base = CHECK / state
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.png")):
            rel = f.relative_to(base).parts
            if len(rel) != 3:
                continue
            mod, patient, png = rel
            volume = png[:-4] + ".nii.gz"
            key = f"{patient}/{volume}"
            dst = CHECK / "checked" / mod / patient / png
            fixes = gate.get(key) or []
            bucket = buckets.get(key, "?")
            if bucket == "SALVAGE" and fixes:
                TOTAL["fix"] += 1
                if not do:
                    print(f"[fix   ] {key} frames={len(fixes)} ops={[x['ops'] for x in fixes]} bucket={bucket}")
                    continue
                if dst.exists():
                    raise FileExistsError(dst)
                vol = DATASET / "labels/cmr_checked/cmr_Myo_mask" / patient / volume
                if not vol.exists():
                    TOTAL["skip"] += 1
                    logf.write(json.dumps({"ts": str(datetime.now()), "action": "fix_skip", "key": key,
                                           "reason": "no_volume_in_checked"}, ensure_ascii=False) + "\n")
                    continue
                img = nib.load(str(vol))
                raw = np.asanyarray(img.dataobj)
                stack, axis = _stack(raw)
                changes = {}
                reason = None
                for fx in fixes:
                    npz_path = _find_npz(fx["pool"], fx["index"], patient)
                    if npz_path is None:
                        reason = "no_candidate_npz"
                        break
                    k = int(npz_path.name.rsplit("_k", 1)[1].split(".")[0])
                    with np.load(npz_path, allow_pickle=False) as data:
                        original = data["original"].astype(bool)
                        fixed = data["fixed"].astype(bool)
                    if (k < 0 or k >= stack.shape[-1] or original.shape != stack[..., k].shape
                            or fixed.shape != original.shape):
                        reason = "k_or_shape_mismatch"
                        break
                    if not np.array_equal(stack[..., k] > .5, original):
                        reason = "stale_source_label"
                        break
                    if k in changes and not np.array_equal(changes[k], fixed):
                        reason = "conflicting_frame_candidates"
                        break
                    changes[k] = fixed
                if reason:
                    TOTAL["skip"] += 1
                    logf.write(json.dumps({"ts": str(datetime.now()), "action": "fix_skip", "key": key,
                                           "reason": reason}, ensure_ascii=False) + "\n")
                    continue
                changed = sum(int(np.count_nonzero((stack[..., k] > .5) != fixed))
                              for k, fixed in changes.items())
                if changed == 0:
                    TOTAL["skip"] += 1
                    logf.write(json.dumps({"ts": str(datetime.now()), "action": "fix_skip", "key": key,
                                           "reason": "no_change"}, ensure_ascii=False) + "\n")
                    continue
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bk_dir = BACKUP_ROOT / ts / patient
                bk_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vol, bk_dir / volume)
                before = _sha(vol)
                for k, fixed in changes.items():
                    stack[..., k] = fixed.astype(stack.dtype) * np.max(raw)
                temporary = vol.with_name(vol.name + ".tmp.nii.gz")
                nib.save(nib.Nifti1Image(_restore(stack, raw, axis), img.affine, img.header), str(temporary))
                os.replace(temporary, vol)
                after = _sha(vol)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
                logf.write(json.dumps({"ts": str(datetime.now()), "action": "fix_applied", "key": key,
                                       "frames": sorted(changes), "changed_pixels": changed,
                                       "ops": [x["ops"] for x in fixes],
                                       "backup": str(bk_dir / volume), "sha_before": before,
                                       "sha_after": after, "png_src": str(f), "png_dst": str(dst)},
                                      ensure_ascii=False) + "\n")
            elif bucket == "SALVAGE":
                TOTAL["direct"] += 1
                if not do:
                    print(f"[direct] {key} bucket={bucket}")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise FileExistsError(dst)
                h = _sha(f)
                shutil.move(str(f), str(dst))
                logf.write(json.dumps({"ts": str(datetime.now()), "action": "direct_promoted", "key": key,
                                       "src": str(f), "dst": str(dst), "sha256": h},
                                      ensure_ascii=False) + "\n")
            # REVIEW/DISCARD/其他 -> 不动
    if logf:
        logf.close()
    print("--- 汇总 ---")
    print(json.dumps(TOTAL, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真正执行（默认 dry-run）")
    run(ap.parse_args())
