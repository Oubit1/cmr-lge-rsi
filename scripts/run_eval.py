#!/usr/bin/env python3
"""Evaluate a QC decision rule against the frozen evaluation set.

Detector `reference` mirrors the deterministic signals the skill already uses
(component count, blood-pool hole, border contact, temporal area jump). It exists
so the evaluation set itself is validated before any agent-written shadow tool is
scored against it: a rule that cannot separate injected defects from clean GT
would make every later comparison meaningless.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

CLASS_SIGNAL = {"fragmentation": "components", "hole_fill": "no_hole",
                "border_clip": "border", "layer_jump": "area_jump"}


def components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    n = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


def hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    reach = np.zeros_like(bg, bool)
    stack = [(y, x) for y in range(h) for x in (0, w - 1) if bg[y, x]]
    stack += [(y, x) for x in range(w) for y in (0, h - 1) if bg[y, x]]
    for y, x in stack:
        reach[y, x] = True
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                stack.append((ny, nx))
    holes = bg & ~reach
    return components(holes)


def signals(label: Path) -> dict:
    a = np.asanyarray(nib.load(str(label)).dataobj) > 0
    frames = a if a.ndim == 3 else a[None]
    areas = frames.reshape(len(frames), -1).sum(axis=1).astype(float)
    mid = frames[len(frames) // 2]
    ys, xs = np.nonzero(mid)
    if len(ys):
        dist = min(ys.min(), xs.min(), mid.shape[0] - 1 - ys.max(), mid.shape[1] - 1 - xs.max())
    else:
        dist = 999
    border = dist <= 3
    cv = float(areas.std() / areas.mean()) if areas.mean() else 0.0
    others = np.delete(areas, len(areas) // 2) if len(areas) > 2 else areas
    med = float(np.median(others)) if len(others) else 0.0
    mid = frames[len(frames) // 2]
    mid_area = float(areas[len(areas) // 2])
    return {
        "components": components(mid),
        "no_hole": hole_count(mid) == 0 and areas.mean() > 200,
        "border": border,
        # ED/ES area change is physiology; only a single-frame outlier is anomalous
        "area_jump": bool(med > 0 and mid_area > 1.3 * med),
        "area": float(areas.mean()),
        "cv": cv,
    }


def flagged(s: dict) -> list[str]:
    out = []
    if s["components"] > 1:
        out.append("components")
    if s["no_hole"]:
        out.append("no_hole")
    if s["border"]:
        out.append("border")
    if s["area_jump"]:
        out.append("area_jump")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("runtime/eval_set/v1"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    syn = [json.loads(l) for l in (args.eval_dir / "synthetic/manifest.jsonl").read_text().splitlines() if l]
    gold = [json.loads(l) for l in (args.eval_dir / "manifest.jsonl").read_text().splitlines() if l]
    gold_cmr = [r for r in gold if r["modality"] == "cmr" and r["tier"] in ("gold", "quasi")]
    if args.limit:
        syn, gold_cmr = syn[:args.limit], gold_cmr[:args.limit]

    per_class = defaultdict(lambda: {"n": 0, "hit": 0})
    fp = defaultdict(int)
    for rec in syn:
        s = signals(Path(rec["synthetic_label"]))
        f = flagged(s)
        cls = rec["class"]
        per_class[cls]["n"] += 1
        if cls == "true_small":
            if f:
                fp["true_small"] += 1
        elif CLASS_SIGNAL[cls] in f:
            per_class[cls]["hit"] += 1

    clean_flag = 0
    breakdown = defaultdict(int)
    for rec in gold_cmr:
        f = flagged(signals(Path(rec["label"])))
        if f:
            clean_flag += 1
            for name in f:
                breakdown[name] += 1

    report = {
        "records": {"synthetic": len(syn), "clean_gold_cmr": len(gold_cmr)},
        "detection_rate": {c: round(v["hit"] / v["n"], 3) if v["n"] else None
                           for c, v in per_class.items() if c != "true_small"},
        "false_repair_true_small": round(fp["true_small"] / max(1, per_class["true_small"]["n"]), 3),
        "clean_gold_flag_rate": round(clean_flag / max(1, len(gold_cmr)), 3),
        "clean_flag_breakdown": dict(sorted(breakdown.items(), key=lambda kv: -kv[1])),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
