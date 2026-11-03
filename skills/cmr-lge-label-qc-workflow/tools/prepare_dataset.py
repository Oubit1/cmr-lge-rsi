#!/usr/bin/env python3
"""Create a Processed_dataset layout from NIfTI or DICOM input.

For DICOM, series are classified from SeriesDescription/ProtocolName using
conservative cine/LGE keywords. Ambiguous series are copied to an intake issue
manifest and stop before model inference rather than being mislabeled.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom import dcmread


NIFTI = (".nii", ".nii.gz")
CINE = ("cine", "fiesta", "truefisp", "bffe", "bssfp", "ssfp", "movie")
LGE = ("lge", "late", "delayed", "psir", "psmde", "de-high", "de high", "inversion recovery")


def classify(text: str) -> str | None:
    value = re.sub(r"[_/\\-]+", " ", text.lower())
    has_lge = any(k in value for k in LGE)
    has_cine = any(k in value for k in CINE)
    if has_lge and not has_cine:
        return "lge"
    if has_cine and not has_lge:
        return "cmr"
    return None


def copy_nifti(input_root: Path, output: Path) -> list[dict]:
    records = []
    for path in sorted(p for p in input_root.rglob("*") if p.is_file() and p.name.endswith(NIFTI)):
        rel = path.relative_to(input_root)
        patient = rel.parts[0] if len(rel.parts) > 1 else path.parent.name
        modality = classify(" ".join(rel.parts))
        if modality is None and ("cmr" in str(rel).lower() or "cine" in str(rel).lower()):
            modality = "cmr"
        if modality is None and ("lge" in str(rel).lower() or "late" in str(rel).lower()):
            modality = "lge"
        if modality is None:
            continue
        target = output / "images" / modality / patient / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({"source": str(path), "target": str(target), "patient": patient, "modality": modality})
    return records


def read_series(path: Path) -> tuple[str, list[Path], str]:
    first = dcmread(str(path), stop_before_pixels=True, force=True)
    uid = str(getattr(first, "SeriesInstanceUID", ""))
    patient = str(getattr(first, "PatientID", path.parent.name))
    text = " ".join(str(getattr(first, key, "")) for key in ("SeriesDescription", "ProtocolName", "SequenceName", "Modality"))
    return uid, [], patient + " " + text


def convert_dicom(input_root: Path, output: Path) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    metadata: dict[tuple[str, str], str] = {}
    for path in sorted(p for p in input_root.rglob("*") if p.is_file()):
        try:
            ds = dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            continue
        patient = str(getattr(ds, "PatientID", path.parent.name))
        key = (patient, uid)
        groups[key].append(path)
        metadata[key] = " ".join(str(getattr(ds, k, "")) for k in ("SeriesDescription", "ProtocolName", "SequenceName", "Modality"))
    records, issues = [], []
    for (patient, uid), files in groups.items():
        modality = classify(metadata[(patient, uid)])
        if modality is None:
            issues.append({"patient": patient, "series_uid": uid, "files": len(files), "description": metadata[(patient, uid)], "issue": "ambiguous_series"})
            continue
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(p) for p in sorted(files)])
        try:
            image = reader.Execute()
        except Exception as exc:
            issues.append({"patient": patient, "series_uid": uid, "files": len(files), "description": metadata[(patient, uid)], "issue": f"dicom_conversion_failed: {exc}"})
            continue
        name = f"series_{abs(hash(uid)) % 100000000:08d}.nii.gz"
        target = output / "images" / modality / patient / name
        target.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(target))
        records.append({"patient": patient, "series_uid": uid, "modality": modality, "files": len(files), "target": str(target), "description": metadata[(patient, uid)]})
    return records, issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(p.is_file() and p.name.endswith(NIFTI) for p in args.input.rglob("*")):
        records = copy_nifti(args.input, args.output)
        issues = []
    else:
        records, issues = convert_dicom(args.input, args.output)
    manifest = args.output / "label_iteration/intake_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now().isoformat(), "input": str(args.input.resolve()), "output": str(args.output.resolve()), "records": records, "issues": issues}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "issues": len(issues), "manifest": str(manifest)}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit("intake contains ambiguous or failed series; review intake_manifest.json before inference")


if __name__ == "__main__":
    main()
