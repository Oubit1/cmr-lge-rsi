#!/usr/bin/env python3
"""Move CMR boundary check PNGs into four simple review-state folders.

Only PNG review artifacts move. NIfTI images and labels are never moved.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


STATUS_FOLDER = {
    "DISCARD": "垃圾桶",
    "SALVAGE": "待处理",
    "REVIEW": "待筛",
}


def png_name(volume: str) -> str:
    return (volume[:-7] if volume.endswith(".nii.gz") else Path(volume).stem) + ".png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.triage.open(encoding="utf-8-sig", newline="")))
    check = args.dataset / "labels/check"
    manifest = args.manifest or check / "manifest_backup/cmr_boundary_manifest.csv"
    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    cmr_keys = {
        f'{row.get("patient", "")}/{row.get("volume") or row.get("slice") or ""}'
        for row in manifest_rows
    }
    counts: Counter[str] = Counter()
    missing = []
    conflicts = []
    scoped_rows = [row for row in rows if row["relative_key"] in cmr_keys]
    scoped_keys = {row["relative_key"] for row in scoped_rows}
    pruned_links = 0
    if args.apply:
        for folder in ("待筛", "待处理", "垃圾桶", "checked"):
            base = check / folder / "cmr"
            for path in base.rglob("*.png") if base.is_dir() else ():
                relative_key = f"{path.parent.name}/{path.stem}.nii.gz"
                if relative_key not in scoped_keys and path.is_symlink():
                    path.unlink()
                    pruned_links += 1
    for row in scoped_rows:
        patient, volume = row["relative_key"].split("/", 1)
        source = check / "cmr/cmr_Myo_mask" / patient / png_name(volume)
        folder = STATUS_FOLDER[row["bucket"]]
        target = check / folder / "cmr" / patient / png_name(volume)
        existing = [check / name / "cmr" / patient / png_name(volume)
                    for name in ("待筛", "待处理", "垃圾桶", "checked")]
        existing = [path for path in existing if path.exists()]
        if existing:
            if target not in existing:
                conflicts.append({"relative_key": row["relative_key"],
                                  "existing": str(existing[0]), "suggested": str(target)})
            current = existing[0]
            if args.apply and current.is_symlink():
                current.unlink()
                current.parent.mkdir(parents=True, exist_ok=True)
                current_source = source
                if not current_source.is_file():
                    missing.append(str(current_source))
                    continue
                current_source.replace(current)
            counts[current.parents[2].name] += 1
            continue
        if not source.is_file():
            missing.append(str(source))
            continue
        counts[folder] += 1
        if args.apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "triage_rows": len(rows), "cmr_manifest_keys": len(cmr_keys),
        "scoped_rows": len(scoped_rows), "counts": dict(counts),
        "missing_pngs": len(missing), "state_conflicts": len(conflicts),
        "pruned_non_cmr_links": pruned_links,
        "note": "check PNGs are moved once into review-state folders; production NIfTI files are unchanged",
    }
    if args.apply:
        (check / "boundary_review_initialization.json").write_text(
            json.dumps({**summary, "missing": missing, "conflicts": conflicts},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
