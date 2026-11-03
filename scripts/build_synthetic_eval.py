#!/usr/bin/env python3
"""Inject known defects into clean GT so recall and over-repair have a true answer.

Classes (100 each):
  fragmentation : a small satellite blob detached from the myocardium
  hole_fill     : the blood-pool hole inside the ring is filled
  border_clip   : label truncated at the image border
  layer_jump    : one frame is dilated, area jumps ~30%
  true_small    : legitimate small apical ring, must NOT be repaired (control)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np

CLASSES = ("fragmentation", "hole_fill", "border_clip", "layer_jump", "true_small")


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the biggest 4-connected component (iterative, no scipy)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    best, best_n = None, -1
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, comp = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > best_n:
                best_n, best = len(comp), comp
    out = np.zeros_like(mask)
    if best:
        for y, x in best:
            out[y, x] = True
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood the background from the border; anything unreached is a hole."""
    h, w = mask.shape
    bg = ~mask
    reach = np.zeros_like(bg, bool)
    stack = []
    for y in range(h):
        for x in (0, w - 1):
            if bg[y, x]:
                stack.append((y, x)); reach[y, x] = True
    for x in range(w):
        for y in (0, h - 1):
            if bg[y, x]:
                stack.append((y, x)); reach[y, x] = True
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                stack.append((ny, nx))
    return mask | (bg & ~reach)


def dilate(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    out = mask.copy()
    for _ in range(iters):
        up = np.roll(out, 1, axis=0); up[-1, :] = False
        dn = np.roll(out, -1, axis=0); dn[0, :] = False
        lf = np.roll(out, 1, axis=1); lf[:, -1] = False
        rt = np.roll(out, -1, axis=1); rt[:, 0] = False
        out = out | up | dn | lf | rt
    return out


def shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def defect(mask2d: np.ndarray, cls: str, rng: random.Random, is_mid: bool = True) -> np.ndarray:
    m = mask2d.copy()
    if cls == "true_small":
        return m
    if cls == "hole_fill":
        # the real defect is a 1-3 frame gap; filling every frame would be
        # indistinguishable from a legitimately solid apical slice
        return fill_holes(m) if is_mid else m
    if cls == "border_clip":
        if not is_mid:
            return m          # neighbours stay intact so the area drop is measurable
        ys, xs = np.nonzero(m)
        if not len(ys):
            return m
        # the ring runs out of the field of view: move it until 4 px stick out on
        # two sides, then let the crop remove what falls outside
        return shift(m, -(int(ys.min()) + 4), -(int(xs.min()) + 4))
    if cls == "layer_jump":
        return dilate(m, 2) if is_mid else m
    if cls == "fragmentation":
        if not is_mid:
            return m          # only the middle frame carries the extra piece
        main = largest_component(m)
        ys, xs = np.nonzero(main)
        if len(ys) < 200:
            return m
        i = rng.randrange(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        r = 4
        blob = np.zeros_like(m)
        blob[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = True
        blob &= main
        if blob.sum() < 20:
            return m
        disc = blob[max(0, cy - r):cy + r, max(0, cx - r):cx + r].copy()
        m = m & ~blob
        h, w = m.shape
        # image corners are almost always free of myocardium
        for ty, tx in ((r + 2, r + 2), (r + 2, w - r - 2), (h - r - 2, r + 2), (h - r - 2, w - r - 2)):
            if m[ty - r:ty + r, tx - r:tx + r].any():
                continue
            m[ty - r:ty + r, tx - r:tx + r] |= disc
            return m
        return m
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("runtime/eval_set/v1/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runtime/eval_set/v1/synthetic"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [r for r in rows if r["modality"] == "cmr" and r["tier"] == "gold"
            and not r.get("border_touch") and r.get("area", 0) > 800]
    pool.sort(key=lambda r: r["key"])
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, made = [], {c: 0 for c in CLASSES}
    for rec in pool:
        if all(v >= args.per_class for v in made.values()):
            break
        cls = min(CLASSES, key=lambda c: made[c])
        if made[cls] >= args.per_class:
            continue
        img = nib.load(rec["label"])
        data = np.asanyarray(img.dataobj) > 0
        squeeze = data.ndim == 3 and data.shape[0] <= 40
        frames = data if squeeze else data[None]
        mid = len(frames) // 2
        gap = {max(0, mid - 1), mid} if cls == "hole_fill" else {mid}
        out = np.stack([defect(fr, cls, rng, i in gap) for i, fr in enumerate(frames)]).astype(np.uint8)
        if not squeeze:
            out = out[0]
        dst = args.out / cls / f"{Path(rec['label']).stem}.nii.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(out.reshape(data.shape), img.affine, img.header), str(dst))
        manifest.append({
            "class": cls, "expected_action": "keep" if cls == "true_small" else "repair",
            "image": rec["image"], "synthetic_label": str(dst), "clean_label": rec["label"],
            "dataset": rec["dataset"], "split": rec["split"], "stratum": rec["stratum"],
            "area": rec.get("area"), "frames": rec.get("frames"),
        })
        made[cls] += 1

    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(manifest), "by_class": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
