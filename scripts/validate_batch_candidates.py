#!/usr/bin/env python3
"""Deterministically validate staged CMR mask repairs without writing labels."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.manual_annotation import _manifest, _paths, _row_identity, _stack  # noqa: E402

CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def identity_matches(row: dict, current_ident: dict) -> bool:
    """Require a staged report row to still name the manifest row at its index."""
    try:
        reported = (
            str(row.get("patient") or ""),
            str(row.get("volume") or ""),
            int(row.get("frame") or 0),
        )
    except (TypeError, ValueError):
        return False
    return reported == (
        str(current_ident.get("patient") or ""),
        str(current_ident.get("volume") or ""),
        int(current_ident.get("frame") or 0),
    )


def candidate_context_reasons(row: dict, current_ident: dict,
                              dataset: Path, current_source: Path,
                              candidate_path: Path) -> tuple[list[str], int]:
    """Validate report identity, canonical source path and encoded frame."""
    reasons = []
    if not identity_matches(row, current_ident):
        reasons.append("stale_manifest_index")
    reported_source = Path(str(row.get("source_label") or ""))
    if not reported_source.is_absolute():
        reported_source = dataset / reported_source
    if reported_source.resolve() != current_source.resolve():
        reasons.append("stale_source_path")
    match = re.search(r"_k(\d+)\.npz$", candidate_path.name)
    if not match:
        raise ValueError("candidate frame is missing")
    frame = int(match.group(1))
    if frame != int(current_ident["frame"]) - 1:
        reasons.append("stale_candidate_frame")
    return reasons, frame


def components(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    labels, count = ndi.label(mask, structure=CONN)
    sizes = [int((labels == index).sum()) for index in range(1, count + 1)]
    return labels, count, sizes


def validate_pair(original: np.ndarray, fixed: np.ndarray,
                  operations: list[str]) -> tuple[bool, list[str], dict]:
    original = np.asarray(original, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)
    reasons = []
    changed = int(np.logical_xor(original, fixed).sum())
    if original.shape != fixed.shape:
        reasons.append("shape_mismatch")
        return False, reasons, {"changed_pixels": changed}
    if not original.any() or not fixed.any() or changed == 0:
        reasons.append("empty_or_unchanged")
    if np.logical_and(fixed, ~original).any():
        reasons.append("unexpected_added_pixels")
    labels0, count0, sizes0 = components(original)
    _, count1, sizes1 = components(fixed)
    area0, area1 = int(original.sum()), int(fixed.sum())
    if "remove_small_components" in operations:
        if count1 >= count0:
            reasons.append("component_count_not_reduced")
        # Two dominant components may represent valid LV/RV anatomy.  A
        # fragmentation repair may remove only rank >=3 components and must
        # preserve both largest source components exactly.
        if count0 <= 2:
            reasons.append("two_component_anatomy_is_legal")
        if changed > max(8, int(round(area0 * .35))):
            reasons.append("excessive_component_removal")
        for source_component in np.argsort(sizes0)[-2:]:
            label_index = int(source_component) + 1
            retained = (np.logical_and(fixed, labels0 == label_index).sum()
                        / max(sizes0[source_component], 1))
            if retained < .99:
                reasons.append("dominant_component_modified")
                break
    if "restore_temporal_hole" in operations:
        hull0 = ndi.binary_fill_holes(original)
        hull1 = ndi.binary_fill_holes(fixed)
        holes0 = int((hull0 & ~original).sum())
        holes1 = int((hull1 & ~fixed).sum())
        if not np.array_equal(hull0, hull1):
            reasons.append("outer_silhouette_changed")
        if holes1 <= holes0:
            reasons.append("hole_not_restored")
        if count1 > count0:
            reasons.append("repair_split_myocardium")
        if area1 < .5 * area0:
            reasons.append("excessive_hole_carve")
    if not operations or any(op not in {"remove_small_components", "restore_temporal_hole"}
                             for op in operations):
        reasons.append("unsupported_operation")
    return not reasons, reasons, {
        "changed_pixels": changed, "area_before": area0, "area_after": area1,
        "components_before": count0, "components_after": count1,
    }


def validate_reports(dataset: Path, candidate_root: Path) -> dict:
    records = []
    source_cache: dict[str, np.ndarray] = {}
    for report_path in sorted(candidate_root.glob("*_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pool = str(report.get("pool") or report_path.name.removesuffix("_report.json"))
        manifest_rows = _manifest(dataset, pool)
        for row in report.get("records", []):
            candidate_path = row.get("candidate")
            if not candidate_path:
                continue
            path = Path(candidate_path)
            if not path.is_absolute():
                path = ROOT / path
            reasons = []
            try:
                payload = np.load(path, allow_pickle=False)
                original, fixed = payload["original"], payload["fixed"]
                passed, reasons, metrics = validate_pair(original, fixed, row.get("auto_fix") or [])
                index = int(row.get("index"))
                if not 0 <= index < len(manifest_rows):
                    raise IndexError("manifest index is outside the current pool")
                current_ident = _row_identity(pool, manifest_rows[index])
                _, source = _paths(dataset, current_ident, "myo")
                context_reasons, k = candidate_context_reasons(
                    row, current_ident, dataset, source, path
                )
                if context_reasons:
                    passed = False
                    reasons.extend(context_reasons)
                source_key = str(source.resolve())
                if source_key not in source_cache:
                    source_cache[source_key], _ = _stack(
                        np.asanyarray(nib.load(source_key).dataobj)
                    )
                current = source_cache[source_key][..., k] > .5
                if not np.array_equal(current, original > 0):
                    passed = False
                    reasons.append("stale_source_label")
            except Exception as exc:
                passed, metrics = False, {}
                reasons.append(f"load_error:{type(exc).__name__}")
            records.append({
                "pool": pool, "index": row.get("index"), "candidate": str(path),
                "operation": row.get("auto_fix") or [], "passed": passed,
                "reasons": reasons, "metrics": metrics,
            })
    passed = sum(bool(row["passed"]) for row in records)
    return {
        "schema_version": "batch-candidate-validation-v1",
        "total": len(records), "passed": passed, "rejected": len(records) - passed,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=ROOT / "runtime/batch_candidates")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runtime/batch_candidates/candidate_validation.json")
    args = parser.parse_args()
    report = validate_reports(args.dataset.resolve(), args.candidates.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
