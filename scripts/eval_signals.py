#!/usr/bin/env python3
"""Which QC signal actually separates defects from clean GT?

For every candidate signal we sweep the threshold that keeps the false-positive
rate on 1,350 human-verified clean slices at or below --max-fpr, then report the
detection rate it achieves on each injected defect class. This turns "the rule
feels reasonable" into a number a gate can enforce.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np

B1 = Path("runtime/maintenance/run_000009/B1-CMR-SMALL-AREA-SHADOW-01_20261021_151940"
          "/orchestrator/experiments/cmr_myo_small_area_shadow.py")


def load_b1():
    if not B1.exists():
        return None
    import sys
    spec = importlib.util.spec_from_file_location("b1_shadow", B1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b1_shadow"] = mod   # dataclasses need the module registered first
    spec.loader.exec_module(mod)
    return mod


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


def border_mm(mask: np.ndarray, spacing: float = 1.0) -> float:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0.0
    return float(min(ys.min(), xs.min(), mask.shape[0] - 1 - ys.max(),
                     mask.shape[1] - 1 - xs.max())) * spacing


def signals(label: Path, b1) -> dict:
    a = np.asanyarray(nib.load(str(label)).dataobj) > 0
    frames = a if a.ndim == 3 else a[None]
    areas = frames.reshape(len(frames), -1).sum(axis=1).astype(float)
    mid = frames[len(frames) // 2]
    others = np.delete(areas, len(areas) // 2) if len(areas) > 2 else areas
    med = float(np.median(others)) if len(others) else 0.0
    mid_area = float(areas[len(areas) // 2])
    if b1 is not None:
        try:
            holes = int(b1._compute_hole_count(mid))
            border = float(b1._compute_border_distance(mid))
        except Exception:
            holes, border = 0, border_mm(mid)
    else:
        holes, border = 0, border_mm(mid)
    return {
        "components": components(mid),
        "holes_zero": 1.0 if holes == 0 else 0.0,
        "border_near": 1.0 if border <= 3 else 0.0,
        "border_distance": -border,          # smaller distance = more anomalous
        "temporal_cv": float(areas.std() / areas.mean()) if areas.mean() else 0.0,
        "mid_ratio": (mid_area / med) if med else 0.0,
    }


def sweep(values, labels, max_fpr):
    """Pick the threshold with the best detection rate under the FPR budget."""
    clean = [v for v, y in zip(values, labels) if y == 0]
    defect = [v for v, y in zip(values, labels) if y == 1]
    if not clean or not defect:
        return None
    if set(values) <= {0.0, 1.0}:            # binary signal: its own threshold is 1
        tpr = sum(1 for v in defect if v >= 1.0) / len(defect)
        fpr = sum(1 for v in clean if v >= 1.0) / len(clean)
        return (1.0, tpr, fpr)
    candidates = sorted(set(values))
    best = None
    for t in candidates:
        fpr = sum(1 for v in clean if v >= t) / len(clean)
        if fpr > max_fpr:
            continue
        tpr = sum(1 for v in defect if v >= t) / len(defect)
        if best is None or tpr > best[1]:
            best = (t, tpr, fpr)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("runtime/eval_set/v1"))
    ap.add_argument("--max-fpr", type=float, default=0.10)
    args = ap.parse_args()

    b1 = load_b1()
    syn = [json.loads(l) for l in (args.eval_dir / "synthetic/manifest.jsonl").read_text().splitlines() if l]
    gold = [json.loads(l) for l in (args.eval_dir / "manifest.jsonl").read_text().splitlines() if l]
    clean = [r for r in gold if r["modality"] == "cmr" and r["tier"] in ("gold", "quasi")]

    clean_sig = [signals(Path(r["label"]), b1) for r in clean]
    by_class = {}
    for cls in ("fragmentation", "hole_fill", "border_clip", "layer_jump", "true_small"):
        recs = [r for r in syn if r["class"] == cls]
        by_class[cls] = [signals(Path(r["synthetic_label"]), b1) for r in recs]

    names = list(clean_sig[0].keys())
    table = {}
    for name in names:
        clean_vals = [s[name] for s in clean_sig]
        row = {}
        for cls, sigs in by_class.items():
            vals = clean_vals + [s[name] for s in sigs]
            labels = [0] * len(clean_vals) + [1] * len(sigs)
            best = sweep(vals, labels, args.max_fpr)
            row[cls] = round(best[1], 3) if best else None
            row["threshold_" + cls] = round(best[0], 3) if best else None
        row["clean_flag_rate"] = round(
            sum(1 for v in clean_vals if best and v >= best[0]) / len(clean_vals), 3) if best else None
        table[name] = row

    report = {"clean_records": len(clean), "max_fpr": args.max_fpr,
              "b1_available": b1 is not None, "signals": table}
    out = args.eval_dir / "signal_sweep.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"clean={len(clean)}  b1={b1 is not None}  -> {out}")
    print(f"{'signal':<16}{'thr':>8}{'cleanFP':>9}{'frag':>7}{'hole':>7}{'border':>8}{'jump':>7}{'smallFP':>9}")
    for name, row in table.items():
        thr = row.get("threshold_layer_jump")
        print(f"{name:<16}{(thr if thr is not None else float('nan')):>8.2f}"
              f"{(row['clean_flag_rate'] if row['clean_flag_rate'] is not None else float('nan')):>9.3f}"
              f"{(row['fragmentation'] or 0):>7.3f}{(row['hole_fill'] or 0):>7.3f}"
              f"{(row['border_clip'] or 0):>8.3f}{(row['layer_jump'] or 0):>7.3f}"
              f"{(row['true_small'] or 0):>9.3f}")


if __name__ == "__main__":
    main()
