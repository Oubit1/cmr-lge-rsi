#!/usr/bin/env python3
"""Create the skill's standard image layout from NIfTI inputs.

This tool intentionally refuses DICOM. DICOM series selection is a clinical
modality decision and must be resolved before model inference.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def is_nifti(path: Path) -> bool:
    return path.is_file() and path.name.endswith((".nii", ".nii.gz"))


def copy_modality(source: Path, destination: Path, modality: str) -> list[dict]:
    if not source.exists(): raise FileNotFoundError(source)
    files = [source] if is_nifti(source) else sorted(p for p in source.rglob("*") if is_nifti(p))
    if not files:
        dicom = list(source.rglob("*.dcm")) if source.is_dir() else []
        if dicom:
            raise RuntimeError(f"{modality}: found DICOM but no NIfTI. Select/convert the series before this stage: {source}")
        raise RuntimeError(f"{modality}: no NIfTI files found: {source}")
    records = []
    for path in files:
        if source.is_file():
            patient = path.parent.name
            relative = Path(patient) / path.name
        else:
            relative = path.relative_to(source)
            patient = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
            if len(relative.parts) == 1:
                relative = Path(patient) / path.name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({"modality": modality, "patient": patient, "source": str(path), "target": str(target), "relative": str(relative)})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Output Processed_dataset root")
    ap.add_argument("--cmr-input", type=Path)
    ap.add_argument("--lge-input", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(); root = args.root.resolve()
    if not args.cmr_input and not args.lge_input: ap.error("at least one of --cmr-input/--lge-input is required")
    if args.overwrite and root.exists():
        raise RuntimeError("Refusing recursive overwrite. Use a new output root or remove only a verified empty intake directory.")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    if args.cmr_input: records += copy_modality(args.cmr_input.resolve(), root / "images/cmr", "cmr")
    if args.lge_input: records += copy_modality(args.lge_input.resolve(), root / "images/lge", "lge")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / "label_iteration/intake_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"timestamp": stamp, "root": str(root), "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "records": len(records), "manifest": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
