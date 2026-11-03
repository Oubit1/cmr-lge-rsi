#!/usr/bin/env python3
"""Separate boundary-layer QC candidates from middle-layer candidates.

This is a non-destructive triage step.  It does not edit images or masks and
does not delete the original issue directory.  It creates symlinked views and
CSV manifests so apex/base candidates can be excluded from the next repair
round while middle-layer candidates remain available for review/repair.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def materialize_view(row, destination: Path):
    if not row.get("check"):
        return ""
    source = Path(row["check"]).resolve()
    if not source.exists():
        return ""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if target.exists() or target.is_symlink():
        target.unlink()
    # Hard links preserve the PNG after the legacy issue directory is removed
    # without doubling storage. Fall back to a real copy across filesystems.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return str(target)


def numeric_stem(path: Path):
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem)


def volume_ranks(root: Path, modality: str):
    ranks = {}
    image_root = root / "images" / modality
    for patient_dir in image_root.iterdir():
        if not patient_dir.is_dir():
            continue
        files = sorted(patient_dir.glob("*.nii.gz"), key=numeric_stem)
        total = len(files)
        for rank, path in enumerate(files, 1):
            ranks[(patient_dir.name, path.name)] = (rank, total)
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cmr-end-volumes", type=int, default=2,
                    help="Patient-level spatial CMR positions isolated at each stack end")
    ap.add_argument("--lge-end-layers", type=int, default=2)
    ap.add_argument("--remove-source-cmr", action="store_true",
                    help="remove labels/check/cmr_issue_slices after independent views are verified")
    args = ap.parse_args()
    root = args.root.resolve()
    result = {"root": str(root), "rules": {"cmr_end_volumes": args.cmr_end_volumes,
             "lge_end_layers": args.lge_end_layers}, "modalities": {}}

    cmr_ranks = volume_ranks(root, "cmr")

    for modality in ("cmr", "lge"):
        issue_dir = root / "labels/check" / f"{modality}_issue_slices"
        manifest = issue_dir / "issue_manifest.csv"
        if not manifest.exists():
            continue
        data = rows(manifest)
        if not data:
            continue
        fields = list(data[0].keys())
        boundary, middle = [], []
        # LGE layer_index is a patient-relative stack index.  The total stack
        # size is read from images/lge, not inferred from issue rows.
        lge_counts = {}
        if modality == "lge":
            for p in (root / "images/lge").iterdir():
                if p.is_dir():
                    lge_counts[p.name] = len(list(p.glob("*.nii.gz")))
        for row in data:
            if modality == "cmr":
                # slice_index is a temporal cine frame, not a spatial layer.
                # All frames from an outer spatial volume share its boundary
                # classification (including long-axis/three-chamber views).
                rank, total = cmr_ranks[(row["patient"], row["volume"])]
                is_boundary = rank <= args.cmr_end_volumes or rank > total - args.cmr_end_volumes
            else:
                idx = int(row["layer_index"])
                total = lge_counts.get(row["patient"], idx)
                is_boundary = idx <= args.lge_end_layers or idx > total - args.lge_end_layers
            out = boundary if is_boundary else middle
            out.append(dict(row))

        bdir = root / "labels/check" / f"{modality}_boundary_issue_slices"
        mdir = root / "labels/check" / f"{modality}_middle_issue_slices"
        for d in (bdir, mdir):
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.png"):
                old.unlink()
        for group, d in ((boundary, bdir), (middle, mdir)):
            for row in group:
                new_check = materialize_view(row, d)
                row["check"] = new_check
                row["check_view"] = new_check
        write_csv(bdir / "issue_manifest.csv", boundary, fields + ["check_view"])
        write_csv(mdir / "issue_manifest.csv", middle, fields + ["check_view"])
        result["modalities"][modality] = {
            "issue_rows": len(data),
            "issue_unique_checks": len({r["check"] for r in data}),
            "boundary_rows": len(boundary),
            "boundary_unique_checks": len({r["check"] for r in boundary}),
            "middle_rows": len(middle),
            "middle_unique_checks": len({r["check"] for r in middle}),
            "boundary_patients": len({r["patient"] for r in boundary}),
            "middle_patients": len({r["patient"] for r in middle}),
            "boundary_dir": str(bdir),
            "middle_dir": str(mdir),
        }
    out = root / "label_iteration" / "issue_triage" / "boundary_vs_middle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.remove_source_cmr:
        source = root / "labels/check/cmr_issue_slices"
        if source.exists():
            shutil.rmtree(source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
