#!/usr/bin/env python3
"""Conjunction QC rules derived from the v1 signal sweep.

The sweep showed that every single signal also fires on healthy anatomy:
no blood-pool hole (55% of clean), touching the border (65%), two components
(every LV+RV slice). These rules only fire when two independent conditions agree,
and they use the temporal neighbours to separate a real gap from an apical frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

try:                       # scipy is optional but much faster
    from scipy import ndimage as ndi
except Exception:          # pragma: no cover
    ndi = None


def _components(mask: np.ndarray) -> int:
    if ndi is not None:
        return int(ndi.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))[1])
    seen = np.zeros_like(mask, bool)
    n = 0
    h, w = mask.shape
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


def _holes(mask: np.ndarray) -> int:
    if ndi is not None:
        filled = ndi.binary_fill_holes(mask)
        holes = filled & ~mask
        return int(ndi.label(holes)[1]) if holes.any() else 0
    return 0


def _border_contact(mask: np.ndarray) -> int:
    """How many label pixels sit on the image border."""
    if not mask.any():
        return 0
    return int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())


def _border_distance(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0
    return int(min(ys.min(), xs.min(), mask.shape[0] - 1 - ys.max(), mask.shape[1] - 1 - xs.max()))


def _framestats(mask: np.ndarray) -> dict:
    area = int(mask.sum())
    comps = _components(mask)
    if comps > 1 and ndi is not None:
        labels, n = ndi.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        sizes = sorted((int((labels == i).sum()) for i in range(1, n + 1)), reverse=True)
    else:
        sizes = [area] if area else []
    return {"area": area, "components": comps, "holes": _holes(mask),
            "border": _border_distance(mask), "contact": _border_contact(mask),
            "sizes": sizes}


def decide(mask: np.ndarray) -> dict:
    frames = mask if mask.ndim == 3 else mask[None]
    n = len(frames)
    mid = n // 2
    idx = [i for i in (mid - 1, mid, mid + 1) if 0 <= i < n]
    stats = {i: _framestats(frames[i]) for i in idx}
    s = stats[mid]
    prev = stats.get(mid - 1)
    nxt = stats.get(mid + 1)
    others = [stats[i]["area"] for i in idx if i != mid] or [s["area"]]
    med = float(np.median(others)) if others else 0.0

    flags = {}
    # 1) fragmentation: one dominant ring plus a tiny detached piece, ring still has a hole
    neighbour_comps = [st["components"] for st in (prev, nxt) if st is not None]
    baseline_comps = float(np.median(neighbour_comps)) if neighbour_comps else 0.0
    # a new, tiny component that the neighbouring frames do not have
    if s["components"] > baseline_comps and len(s["sizes"]) >= 2:
        extra = sum(s["sizes"][1:])
        if extra <= 0.18 * s["sizes"][0]:
            flags["fragmentation"] = extra / max(1, s["sizes"][0])
    # 2) temporal hole gap: this frame lost the hole but a neighbour still has it
    if s["holes"] == 0 and med > 0 and s["area"] > 0.5 * med:
        if (prev and prev["holes"] > 0) or (nxt and nxt["holes"] > 0):
            flags["hole_missing"] = 1
    # 3) border clipping: touches the border AND lost area versus neighbours
    neighbour_contact = [st["contact"] for st in (prev, nxt) if st is not None]
    contact_baseline = max(neighbour_contact) if neighbour_contact else 0
    # new contact with the field-of-view border that the neighbouring slices lack
    if s["contact"] > contact_baseline + 5:
        flags["border_clip"] = s["contact"] - contact_baseline
    # 4) single-frame area jump, but not explained by border clipping
    if "border_clip" not in flags and med > 0 and s["area"] > 1.15 * med:
        flags["area_jump"] = s["area"] / med
    return flags


CLASS_FLAG = {"fragmentation": "fragmentation", "hole_fill": "hole_missing",
              "border_clip": "border_clip", "layer_jump": "area_jump"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("runtime/eval_set/v1"))
    args = ap.parse_args()

    syn = [json.loads(l) for l in (args.eval_dir / "synthetic/manifest.jsonl").read_text().splitlines() if l]
    gold = [json.loads(l) for l in (args.eval_dir / "manifest.jsonl").read_text().splitlines() if l]
    clean = [r for r in gold if r["modality"] == "cmr" and r["tier"] in ("gold", "quasi")]

    hit = {c: 0 for c in CLASS_FLAG}
    total = {c: 0 for c in CLASS_FLAG}
    small_fp = small_n = 0
    for rec in syn:
        flags = decide(np.asanyarray(nib.load(rec["synthetic_label"]).dataobj) > 0)
        cls = rec["class"]
        if cls == "true_small":
            small_n += 1
            small_fp += 1 if flags else 0
        else:
            total[cls] += 1
            hit[cls] += 1 if CLASS_FLAG[cls] in flags else 0

    clean_fp = 0
    examples = {}
    for rec in clean:
        flags = decide(np.asanyarray(nib.load(rec["label"]).dataobj) > 0)
        if flags:
            clean_fp += 1
            for k in flags:
                examples[k] = examples.get(k, 0) + 1

    report = {
        "clean_records": len(clean),
        "clean_false_positive_rate": round(clean_fp / max(1, len(clean)), 3),
        "clean_false_positive_by_rule": examples,
        "detection_rate": {c: round(hit[c] / max(1, total[c]), 3) for c in CLASS_FLAG},
        "true_small_false_repair": round(small_fp / max(1, small_n), 3),
    }
    (args.eval_dir / "rules_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
