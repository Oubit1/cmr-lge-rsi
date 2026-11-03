#!/usr/bin/env python3
"""Build the frozen QC evaluation set from ORIGINAL (unregistered) ground truth.

Sources and their exact GT roots (given by the project owner):

  site_c/processed_dataset/{train,test}
      images/cmr + labels/cmr
      images/lge + labels/lge_original/{lge_Myo,lge_MI}
  site_b/processed_dataset/{train,val,test}
      images/cmr + labels/cmr
      images/lge + labels/lge_original/{lge_Myo,lge_MI}
  site_d/original
      images/cmr + labels/cmr_checked/cmr_Myo_mask
      images/lge + labels/lge_original/{lge_Myo_labels,lge_MI_labels}

Different centers store the same thing under slightly different folder names and
either flat or per-patient layouts, so pairing is done by path suffix instead of
by naming convention. Registered/warped variants are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

SOURCES = {
    "shengyi": [
        Path("/srv/rsi/site_c/processed_dataset/train"),
        Path("/srv/rsi/site_c/processed_dataset/test"),
    ],
    "site_b_2016": [
        Path("/srv/rsi/site_b/processed_dataset/train"),
        Path("/srv/rsi/site_b/processed_dataset/val"),
        Path("/srv/rsi/site_b/processed_dataset/test"),
    ],
    "site_d": [
        Path("/srv/rsi/site_d/original"),
    ],
}

# automated + manual QC, not yet signed off by a doctor
QUASI_SOURCES = {
    "site_b_v1": [
        Path("/srv/rsi/site_b/original/Processed_dataset_V1"),
    ],
}

# public multi-centre data, used only as an out-of-domain false-positive guard
EXTERNAL_SOURCES = {
    "multicmr": [
        Path("/srv/rsi/26CMR-MULTI/processed_dataset/test"),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_key(path: Path, salt: str) -> str:
    return hashlib.sha256((salt + str(path)).encode()).hexdigest()


def resolve_label_root(path: Path, contains: str) -> Path | None:
    """Return the directory that holds the per-case files (skips wrapper dirs)."""
    if not path.is_dir():
        return None
    direct = [p for p in path.iterdir() if p.is_file() and p.name.endswith(".nii.gz")]
    if direct:
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and contains.lower() in child.name.lower():
            return child
    for child in sorted(path.iterdir()):
        if child.is_dir() and any(p.is_file() and p.name.endswith(".nii.gz")
                                  for p in child.iterdir()):
            return child
    return None


def pair_by_suffix(img_root: Path, lab_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    if not img_root.is_dir() or lab_root is None:
        return pairs
    for lab in sorted(lab_root.rglob("*.nii.gz")):
        img = img_root / lab.relative_to(lab_root)
        if img.is_file():
            pairs.append((img, lab))
    return pairs


def cmr_features(label: Path) -> dict:
    lab = np.asanyarray(nib.load(str(label)).dataobj) > 0
    if lab.ndim == 2:
        lab = lab[None]
    areas = lab.reshape(lab.shape[0], -1).sum(axis=1).astype(float)
    mean = float(areas.mean())
    mid = lab[len(lab) // 2]
    edge = np.concatenate([mid[0], mid[-1], mid[:, 0], mid[:, -1]])
    return {"area": mean, "temporal_cv": float(areas.std() / mean) if mean else 0.0,
            "frames": int(lab.shape[0]), "border_touch": bool(edge.any())}


def lge_features(myo: Path, mi: Path) -> dict:
    m = np.asanyarray(nib.load(str(myo)).dataobj) > 0
    i = np.asanyarray(nib.load(str(mi)).dataobj) > 0
    m2 = m.reshape(-1, m.shape[-2], m.shape[-1])[0]
    i2 = i.reshape(-1, i.shape[-2], i.shape[-1])[0]
    myo_area, mi_area = int(m2.sum()), int(i2.sum())
    edge = np.concatenate([m2[0], m2[-1], m2[:, 0], m2[:, -1]])
    return {"myo_area": myo_area, "mi_area": mi_area,
            "mi_ratio": (mi_area / myo_area) if myo_area else 0.0,
            "has_mi": mi_area > 0, "border_touch": bool(edge.any())}


def stratified_take(records: list[dict], n: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r["stratum"]].append(r)
    for key in buckets:
        buckets[key].sort(key=lambda r: r["key"])
    picked, keys, i = [], sorted(buckets), 0
    while len(picked) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop())
        i += 1
    return picked


def build(out: Path, per_modality: int, pool: int, salt: str, tiers: set[str]) -> dict:
    manifest, summary = [], {"salt": salt, "sources": {}}
    selected = [("gold", SOURCES)] if "gold" in tiers else []
    if "quasi" in tiers:
        selected.append(("quasi", QUASI_SOURCES))
    if "external" in tiers:
        selected.append(("external", EXTERNAL_SOURCES))
    for tier, sources in selected:
      for name, split_dirs in sources.items():
          counts = {"cmr": 0, "lge": 0}
          for split_dir in split_dirs:
              if not split_dir.is_dir():
                  print(f"  ! missing {split_dir}", flush=True)
                  continue
              split = split_dir.name
              cmr_img = split_dir / "images/cmr"
              cmr_lab = resolve_label_root(split_dir / "labels/cmr", "myo")
              lge_checked = split_dir / "labels/lge_checked"
              lge_img = split_dir / "images/lge"
              lge_gt = split_dir / "labels/lge_original"
              gt_root = lge_checked if lge_checked.is_dir() else lge_gt
              myo_root = resolve_label_root(gt_root / "lge_Myo", "myo") or resolve_label_root(gt_root, "myo")
              mi_root = resolve_label_root(gt_root / "lge_MI", "mi") or resolve_label_root(gt_root, "mi")

              for modality, pairs in (
                  ("cmr", [(i, (l,)) for i, l in pair_by_suffix(cmr_img, cmr_lab)]),
                  ("lge", [(i, (a, b)) for (i, a), (_, b) in zip(
                      pair_by_suffix(lge_img, myo_root), pair_by_suffix(lge_img, mi_root))]),
              ):
                  if not pairs:
                      print(f"  ! {name}/{split}/{modality}: no pairs "
                            f"(img={cmr_img if modality=='cmr' else lge_img})", flush=True)
                      continue
                  pairs = sorted(pairs, key=lambda t: pick_key(t[0], salt))[:pool]
                  records = []
                  for img, labels in pairs:
                      try:
                          feat = cmr_features(labels[0]) if modality == "cmr" else lge_features(*labels)
                      except Exception:
                          continue
                      if modality == "cmr":
                          quart = "q%d" % min(3, int(feat["area"] // 500) % 4)
                          stratum = f"cmr|{quart}|border{int(feat['border_touch'])}"
                          rec = {"modality": "cmr", "image": str(img), "label": str(labels[0]), **feat}
                      else:
                          quart = "q%d" % min(3, int(feat["myo_area"] // 500) % 4)
                          stratum = (f"lge|mi{int(feat['has_mi'])}|{quart}"
                                     f"|border{int(feat['border_touch'])}")
                          rec = {"modality": "lge", "image": str(img),
                                 "myo": str(labels[0]), "mi": str(labels[1]), **feat}
                      rec.update({"tier": tier, "dataset": name, "split": split,
                                  "stratum": stratum, "key": pick_key(img, salt)})
                      records.append(rec)
                  picked = stratified_take(records, per_modality)
                  for rec in picked:
                      rec["sha256_image"] = sha256(Path(rec["image"]))
                  manifest.extend(picked)
                  counts[modality] += len(picked)
                  print(f"  {name}/{split}/{modality}: {len(picked)} of {len(records)}", flush=True)
          summary["sources"][name] = counts

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8")
    summary["total"] = len(manifest)
    summary["by_modality"] = {m: sum(1 for r in manifest if r["modality"] == m) for m in ("cmr", "lge")}
    summary["strata"] = len({r["stratum"] for r in manifest})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runtime/eval_set/v1"))
    ap.add_argument("--per-modality", type=int, default=200)
    ap.add_argument("--pool", type=int, default=700)
    ap.add_argument("--salt", default="cmr-lge-eval-v1")
    ap.add_argument("--tiers", default="gold,quasi")
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.per_modality, args.pool, args.salt,
                           {x.strip() for x in args.tiers.split(",") if x.strip()}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
