#!/usr/bin/env python3
"""Audit and finalize a strictly paired CMR/LGE dataset.

An unusable spatial level is handled as one core five-file bundle: CMR image,
LGE image, CMR Myo, LGE Myo and LGE MI. Any duplicate LGE label namespace is
audited and quarantined with that core bundle so stale copies cannot survive.
Quarantine mode atomically moves complete bundles into a recoverable
``quarantine/unusable_sax/<run>`` tree and prunes their stale check rows.
Nothing is permanently deleted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib


LGE_LAYOUTS = (
    ("original", "labels/lge_original/lge_Myo_labels", "labels/lge_original/lge_MI_labels"),
    ("checked", "labels/lge_checked/lge_Myo_labels", "labels/lge_checked/lge_MI_labels"),
)
BOUNDARY_MANIFESTS = (
    "labels/check/cmr_boundary_issue_slices/issue_manifest.csv",
    "labels/check/lge_boundary_issue_slices/issue_manifest.csv",
)


def nii_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*.nii*") if path.is_file()
    }


def choose_layout(root: Path, preferred: str) -> tuple[dict[str, Path], str]:
    cmr_myo = root / "labels/cmr_checked/cmr_Myo_mask"
    if not cmr_myo.is_dir():
        raise FileNotFoundError(cmr_myo)
    candidates = []
    for name, myo_rel, mi_rel in LGE_LAYOUTS:
        myo, mi = root / myo_rel, root / mi_rel
        if myo.is_dir() and mi.is_dir():
            candidates.append((name, myo, mi))
    if not candidates:
        raise RuntimeError(
            "expected a complete LGE label layout (lge_original or lge_checked)"
        )
    names = {item[0] for item in candidates}
    selected = preferred if preferred in names else ("original" if "original" in names else candidates[0][0])
    primary = next(item for item in candidates if item[0] == selected)
    layout = {
        "cmr_image": root / "images/cmr",
        "lge_image": root / "images/lge",
        "cmr_myo": cmr_myo,
        "lge_myo": primary[1],
        "lge_mi": primary[2],
    }
    # Duplicate label namespaces are not silently ignored. They are audited and
    # quarantined with the core five-file bundle so no stale counterpart remains.
    for name, myo, mi in candidates:
        if name != selected:
            layout[f"lge_myo_aux_{name}"] = myo
            layout[f"lge_mi_aux_{name}"] = mi
    return layout, selected


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str | None:
    patient = (row.get("patient") or "").strip()
    volume = (row.get("volume") or row.get("slice") or "").strip()
    if not patient or not volume:
        return None
    return f"{patient}/{Path(volume).name}"


def collect_candidates(root: Path, include_boundary: bool,
                       reject_manifests: list[Path]) -> tuple[dict[str, set[str]], list[str]]:
    reasons: dict[str, set[str]] = defaultdict(set)
    sources: list[str] = []
    paths = [root / item for item in BOUNDARY_MANIFESTS] if include_boundary else []
    paths.extend(path.resolve() for path in reject_manifests)
    for path in paths:
        rows = read_csv(path)
        if not path.is_file():
            continue
        sources.append(str(path))
        default_reason = "boundary_issue_outside_trusted_sax" if "boundary_issue" in path.as_posix() else "manually_rejected_unusable"
        for row in rows:
            key = row_key(row)
            if not key:
                continue
            reason = (row.get("quarantine_reason") or row.get("reason") or
                      row.get("triage_reason") or default_reason)
            reasons[key].add(reason)
    return reasons, sources


def normalized_shape(path: Path) -> tuple[int, ...]:
    shape = tuple(int(x) for x in nib.load(str(path)).shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def pairing_audit(layout: dict[str, Path], *, check_shapes: bool) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]], dict[str, Any]]:
    files = {name: nii_files(path) for name, path in layout.items()}
    all_keys = sorted(set().union(*(set(group) for group in files.values())))
    rows: list[dict[str, Any]] = []
    complete: dict[str, dict[str, Path]] = {}
    shape_errors = 0
    for key in all_keys:
        missing = [name for name, group in files.items() if key not in group]
        errors: list[str] = []
        if not missing and check_shapes:
            try:
                cmr_shapes = {normalized_shape(files[name][key]) for name in ("cmr_image", "cmr_myo")}
                lge_names = [name for name in files if name == "lge_image" or name.startswith("lge_myo") or name.startswith("lge_mi")]
                lge_shapes = {normalized_shape(files[name][key]) for name in lge_names}
                if len(cmr_shapes) != 1:
                    errors.append("cmr_image_label_shape_mismatch")
                if len(lge_shapes) != 1:
                    errors.append("lge_image_label_shape_mismatch")
            except Exception as exc:  # corrupted NIfTI is a pairing failure
                errors.append(f"nifti_header_error:{type(exc).__name__}")
        if errors:
            shape_errors += 1
        status = "complete" if not missing and not errors else "invalid"
        rows.append({"relative_key": key, "status": status,
                     "missing": ";".join(missing), "errors": ";".join(errors)})
        if status == "complete":
            complete[key] = {name: group[key] for name, group in files.items()}
    counts = {name: len(group) for name, group in files.items()}
    summary = {
        "counts": counts,
        "union_keys": len(all_keys),
        "complete_pairs": len(complete),
        "incomplete_or_invalid": len(all_keys) - len(complete),
        "shape_errors": shape_errors,
        "exact_key_sets": len({frozenset(group) for group in files.values()}) == 1,
    }
    return complete, rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def prune_check_manifests(root: Path, run_quarantine: Path, keys: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    check_root = root / "labels/check"
    if not check_root.is_dir():
        return result
    for manifest in sorted(check_root.glob("*/issue_manifest.csv")):
        data = read_csv(manifest)
        if not data:
            continue
        removed = [row for row in data if row_key(row) in keys]
        if not removed:
            continue
        retained = [row for row in data if row_key(row) not in keys]
        evidence = run_quarantine / "check_evidence" / manifest.parent.name
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, evidence / "issue_manifest.before.csv")
        fields = list(data[0])
        write_csv(evidence / "removed_issue_manifest.csv", removed, fields)
        write_csv(manifest, retained, fields)
        moved_pngs = 0
        for row in removed:
            for field in ("check", "check_view"):
                value = row.get(field)
                if not value:
                    continue
                source = Path(value)
                if source.is_file() and source.parent.resolve() == manifest.parent.resolve():
                    target = evidence / source.name
                    if not target.exists():
                        os.replace(source, target)
                        moved_pngs += 1
        result[manifest.parent.name] = len(removed)
        result[manifest.parent.name + "_pngs"] = moved_pngs
    return result


def quarantine_bundles(root: Path, layout: dict[str, Path], complete: dict[str, dict[str, Path]],
                       reasons: dict[str, set[str]], run_quarantine: Path) -> list[dict[str, Any]]:
    keys = sorted(reasons)
    blocked = [key for key in keys if key not in complete]
    if blocked:
        raise RuntimeError(
            f"refusing partial quarantine: {len(blocked)} candidate keys do not have a valid five-file bundle; "
            "inspect quarantine_candidates.csv"
        )
    plan: list[tuple[str, str, Path, Path]] = []
    for key in keys:
        for name, source in complete[key].items():
            relative_root = layout[name].relative_to(root)
            destination = run_quarantine / relative_root / key
            if destination.exists():
                raise FileExistsError(destination)
            plan.append((key, name, source, destination))
    moved: list[tuple[Path, Path]] = []
    try:
        for _, _, source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        raise
    return [
        {"relative_key": key, "reason": ";".join(sorted(reasons[key])),
         "bundle_files": len(layout), "quarantine_root": str(run_quarantine)}
        for key in keys
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "quarantine"), default="audit")
    parser.add_argument("--reject-manifest", type=Path, action="append", default=[],
                        help="additional CSV; every patient/volume or patient/slice row is quarantined")
    parser.add_argument("--no-boundary", action="store_true",
                        help="do not include current CMR/LGE boundary issue manifests")
    parser.add_argument("--skip-shape-check", action="store_true")
    parser.add_argument("--quarantine-root", type=Path,
                        help="default: ROOT/quarantine/unusable_sax")
    parser.add_argument("--lge-label-set", choices=("original", "checked"), default="checked",
                        help="authoritative LGE label namespace; other complete namespaces remain audited auxiliaries")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    layout, selected_lge_labels = choose_layout(root, args.lge_label_set)
    reasons, sources = collect_candidates(root, not args.no_boundary, args.reject_manifest)
    complete, pair_rows, before = pairing_audit(layout, check_shapes=not args.skip_shape_check)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/dataset_finalization" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_rows = [
        {"relative_key": key, "status": "ready" if key in complete else "blocked_incomplete_bundle",
         "reason": ";".join(sorted(value))}
        for key, value in sorted(reasons.items())
    ]
    write_csv(run_dir / "pair_audit_before.csv", pair_rows,
              ["relative_key", "status", "missing", "errors"])
    write_csv(run_dir / "quarantine_candidates.csv", candidate_rows,
              ["relative_key", "status", "reason"])

    quarantined: list[dict[str, Any]] = []
    pruned: dict[str, int] = {}
    quarantine_run = None
    after = before
    if args.mode == "quarantine":
        if before["incomplete_or_invalid"]:
            raise RuntimeError(
                "dataset is not five-way complete before finalization; inspect pair_audit_before.csv"
            )
        quarantine_base = (args.quarantine_root or root / "quarantine/unusable_sax").resolve()
        quarantine_run = quarantine_base / stamp
        quarantine_run.mkdir(parents=True, exist_ok=False)
        quarantined = quarantine_bundles(root, layout, complete, reasons, quarantine_run)
        write_csv(quarantine_run / "quarantine_manifest.csv", quarantined,
                  ["relative_key", "reason", "bundle_files", "quarantine_root"])
        pruned = prune_check_manifests(root, quarantine_run, set(reasons))
        _, after_rows, after = pairing_audit(layout, check_shapes=not args.skip_shape_check)
        write_csv(run_dir / "pair_audit_after.csv", after_rows,
                  ["relative_key", "status", "missing", "errors"])
        if not after["exact_key_sets"] or after["incomplete_or_invalid"]:
            raise RuntimeError("post-quarantine pair invariant failed; do not use finalized dataset")

    summary = {
        "root": str(root), "mode": args.mode, "run": str(run_dir),
        "candidate_sources": sources, "candidate_pairs": len(reasons),
        "quarantined_pairs": len(quarantined),
        "quarantined_files": len(quarantined) * len(layout),
        "bundle_members": sorted(layout),
        "authoritative_lge_labels": selected_lge_labels,
        "quarantine_run": str(quarantine_run or ""),
        "check_rows_pruned": pruned,
        "before": before, "after": after,
        "final_invariant": {
            "exact_all_bundle_relative_keys": bool(after["exact_key_sets"]),
            "all_bundles_complete_and_shape_valid": after["incomplete_or_invalid"] == 0,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
