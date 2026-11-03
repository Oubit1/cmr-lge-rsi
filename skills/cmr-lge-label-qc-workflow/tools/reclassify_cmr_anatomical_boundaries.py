#!/usr/bin/env python3
"""Reclassify anatomically basal/apical CMR volumes from middle to boundary QC.

The existing boundary splitter uses only spatial rank.  That misses basal
multi-chamber/outflow-tract volumes when they are more than two files from the
numeric stack end.  This tool derives a patient-level trusted SAX core from
the fraction of cine frames with physiological ring topology.  Only a
contiguous prefix/suffix outside that core can be reclassified.  Internal
topology failures always remain in the middle repair pool.

This tool never edits an image or a label.  ``dry-run`` writes evidence only;
``apply`` updates the two check manifests and moves their rendered PNG views.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def numeric_stem(path: Path):
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def major_component_count(mask: np.ndarray, minimum: int = 20) -> int:
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    if not count:
        return 0
    areas = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(areas >= minimum))


def physiological_ring(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return False
    components = major_component_count(mask)
    holes = ndi.binary_fill_holes(mask) & ~mask
    _, hole_count = ndi.label(holes, structure=np.ones((3, 3)))
    return 1 <= components <= 2 and 1 <= hole_count <= 2


def volume_metrics(mask_path: Path) -> dict[str, float | int]:
    data = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    if data.ndim == 2:
        data = data[..., None]
    elif data.ndim != 3:
        return {"frames": 0, "ring_frames": 0, "ring_fraction": 0.0,
                "median_area": 0.0}
    ring = [physiological_ring(data[..., index]) for index in range(data.shape[-1])]
    areas = np.sum(data, axis=(0, 1))
    return {
        "frames": int(data.shape[-1]),
        "ring_frames": int(sum(ring)),
        "ring_fraction": float(np.mean(ring)),
        "median_area": float(np.median(areas)),
    }


def fill_single_internal_gaps(values: list[bool]) -> list[bool]:
    out = list(values)
    for index in range(1, len(values) - 1):
        if not values[index] and values[index - 1] and values[index + 1]:
            out[index] = True
    return out


def longest_true_run(values: list[bool]) -> tuple[int, int] | None:
    best = None
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            run = (start, index - 1)
            if best is None or run[1] - run[0] > best[1] - best[0]:
                best = run
            start = None
    return best


def true_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def patient_classification(root: Path, patient: str, threshold: float,
                           minimum_core: int) -> dict:
    image_dir = root / "images/cmr" / patient
    mask_dir = root / "labels/cmr/cmr_Myo_mask" / patient
    volumes = sorted(image_dir.glob("*.nii.gz"), key=numeric_stem)
    metrics = []
    for volume in volumes:
        mask = mask_dir / volume.name
        item = volume_metrics(mask) if mask.exists() else {
            "frames": 0, "ring_frames": 0, "ring_fraction": 0.0,
            "median_area": 0.0,
        }
        item["volume"] = volume.name
        metrics.append(item)
    trusted = fill_single_internal_gaps([
        item["ring_fraction"] >= threshold for item in metrics
    ])
    segments = true_runs(trusted)
    core = longest_true_run(trusted)
    # Multiple trusted segments can indicate mixed acquisitions or coverage
    # gaps.  A single longest core must not silently absorb that ambiguity.
    reliable = (len(segments) == 1 and core is not None and
                core[1] - core[0] + 1 >= minimum_core)
    anatomical_boundary = set()
    if reliable:
        anatomical_boundary.update(item["volume"] for item in metrics[:core[0]])
        anatomical_boundary.update(item["volume"] for item in metrics[core[1] + 1:])
    return {
        "patient": patient,
        "reliable_core": reliable,
        "core_start_rank": core[0] + 1 if reliable else None,
        "core_end_rank": core[1] + 1 if reliable else None,
        "core_length": core[1] - core[0] + 1 if reliable else 0,
        "trusted_core_segments": len(segments),
        "fallback_reason": ("multiple_trusted_segments" if len(segments) > 1
                            else "core_too_short_or_absent" if not reliable else ""),
        "volume_count": len(volumes),
        "anatomical_boundary_volumes": sorted(anatomical_boundary, key=lambda x: numeric_stem(Path(x))),
        "volumes": metrics,
    }


def materialize(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "apply", "rollback"), default="dry-run")
    parser.add_argument("--rollback-manifest", type=Path,
                        help="candidate CSV produced before apply; required for rollback")
    parser.add_argument("--ring-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-core-volumes", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    check_root = root / "labels/check"
    middle_dir = check_root / "cmr_middle_issue_slices"
    boundary_dir = check_root / "cmr_boundary_issue_slices"
    middle_manifest = middle_dir / "issue_manifest.csv"
    boundary_manifest = boundary_dir / "issue_manifest.csv"
    middle = read_csv(middle_manifest)
    boundary = read_csv(boundary_manifest)
    # A check can have multiple issue rows.  All rows must point to the one
    # materialized boundary PNG even if a previous row already moved it.
    if args.mode == "apply":
        for row in boundary:
            source_value = row.get("check_view") or row.get("check")
            if not source_value:
                continue
            local = boundary_dir / Path(source_value).name
            if local.exists():
                row["check"] = str(local)
                row["check_view"] = str(local)

    if args.mode == "rollback":
        if not args.rollback_manifest or not args.rollback_manifest.exists():
            raise SystemExit("--rollback-manifest is required for rollback")
        candidates = read_csv(args.rollback_manifest)
        def identity(row):
            return tuple(row.get(key, "") for key in
                         ("patient", "volume", "slice_index", "issues", "evidence"))
        candidate_keys = {identity(row) for row in candidates}
        retained_boundary = [row for row in boundary if identity(row) not in candidate_keys]
        existing_middle = {identity(row) for row in middle}
        restored = []
        for row in candidates:
            clean = {key: value for key, value in row.items()
                     if key not in {"triage_reason", "ring_fraction", "sax_core_ranks"}}
            source_value = clean.get("check_view") or clean.get("check")
            basename = Path(source_value).name if source_value else ""
            source = boundary_dir / basename if basename else None
            target = middle_dir / basename if basename else None
            if source and source.exists() and target:
                materialize(source, target)
            if target and target.exists():
                clean["check"] = str(target)
                clean["check_view"] = str(target)
            if identity(clean) not in existing_middle:
                restored.append(clean)
                existing_middle.add(identity(clean))
        referenced_boundary = {
            Path(row.get("check_view") or row.get("check")).name
            for row in retained_boundary if row.get("check_view") or row.get("check")
        }
        for row in candidates:
            value = row.get("check_view") or row.get("check")
            if value:
                path = boundary_dir / Path(value).name
                if path.exists() and path.name not in referenced_boundary:
                    path.unlink()
        excluded = {"triage_reason", "ring_fraction", "sax_core_ranks"}
        all_rows = middle + restored + retained_boundary
        fields = list(dict.fromkeys(
            key for row in all_rows for key in row.keys() if key not in excluded
        ))
        write_csv(middle_manifest, middle + restored, fields)
        write_csv(boundary_manifest, retained_boundary, fields)
        result = {
            "mode": "rollback", "restored_rows": len(restored),
            "middle_rows": len(middle) + len(restored),
            "boundary_rows": len(retained_boundary),
            "labels_modified": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    patients = sorted({row["patient"] for row in middle})
    classifications = {
        patient: patient_classification(root, patient, args.ring_fraction,
                                        args.minimum_core_volumes)
        for patient in patients
    }
    move_rows, keep_rows = [], []
    for row in middle:
        info = classifications[row["patient"]]
        if info["reliable_core"] and row["volume"] in set(info["anatomical_boundary_volumes"]):
            enriched = dict(row)
            metric = next(item for item in info["volumes"] if item["volume"] == row["volume"])
            enriched.update({
                "triage_reason": "anatomical_boundary_outside_trusted_sax_core",
                "ring_fraction": f'{metric["ring_fraction"]:.4f}',
                "sax_core_ranks": f'{info["core_start_rank"]}-{info["core_end_rank"]}',
            })
            move_rows.append(enriched)
        else:
            keep_rows.append(dict(row))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = root / "label_iteration/issue_triage" / f"cmr_anatomical_boundary_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "root": str(root), "mode": args.mode,
        "rule": {
            "ring_fraction": args.ring_fraction,
            "minimum_core_volumes": args.minimum_core_volumes,
            "scope": "contiguous prefix/suffix outside longest trusted SAX core",
            "internal_failures_remain_middle": True,
            "labels_modified": False,
        },
        "before": {"middle_rows": len(middle), "boundary_rows": len(boundary)},
        "candidate_move_rows": len(move_rows),
        "candidate_move_unique_checks": len({row.get("check", "") for row in move_rows}),
        "candidate_move_patients": len({row["patient"] for row in move_rows}),
        "after": {"middle_rows": len(keep_rows), "boundary_rows": len(boundary) + len(move_rows)},
        "patients": list(classifications.values()),
    }
    fields = list(dict.fromkeys(
        [key for row in middle + boundary + move_rows for key in row.keys()]
    ))
    write_csv(report_dir / "reclassification_candidates.csv", move_rows, fields)
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.mode == "dry-run":
        shadow_boundary = report_dir / "shadow_boundary_candidates"
        shadow_middle = report_dir / "shadow_middle_retained"
        shadow_boundary_rows, shadow_middle_rows = [], []
        for source_rows, destination, output_rows in (
            (move_rows, shadow_boundary, shadow_boundary_rows),
            (keep_rows, shadow_middle, shadow_middle_rows),
        ):
            for row in source_rows:
                copied = dict(row)
                source_value = row.get("check_view") or row.get("check")
                source = Path(source_value) if source_value else None
                target = destination / source.name if source else None
                if source and source.exists() and target:
                    materialize(source, target)
                    copied["check"] = str(target)
                    copied["check_view"] = str(target)
                output_rows.append(copied)
        write_csv(shadow_boundary / "issue_manifest.csv", shadow_boundary_rows, fields)
        write_csv(shadow_middle / "issue_manifest.csv", shadow_middle_rows, fields)

    if args.mode == "apply":
        for row in move_rows:
            source_value = row.get("check_view") or row.get("check")
            source = Path(source_value) if source_value else None
            target = boundary_dir / source.name if source else None
            if source and source.exists() and target:
                materialize(source, target)
                if source.resolve() != target.resolve():
                    source.unlink()
            if target and target.exists():
                row["check"] = str(target)
                row["check_view"] = str(target)
        write_csv(middle_manifest, keep_rows, fields)
        write_csv(boundary_manifest, boundary + move_rows, fields)

    print(json.dumps({**{k: v for k, v in report.items() if k != "patients"},
                      "report_dir": str(report_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
