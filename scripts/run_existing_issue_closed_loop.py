#!/usr/bin/env python3
"""Run a governed shadow -> validation -> writeback decision loop.

This adapter evaluates the isolated A1/A2/A3 experiments on real unresolved
LGE middle-slice masks.  Experimental candidates never write labels.  The
report explicitly records whether an already validated production repair had
an eligible candidate; only those repairs may enter the writeback stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from experiments.advisory_rules import evaluate_mi_fragment_rule
from experiments.lge_shadow import ShadowParameters, extract_slice_features, simulate_candidates
from experiments.topology_shadow import evaluate_topology_shadow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-slices", type=int, default=120)
    parser.add_argument("--max-topology-patients", type=int, default=10)
    return parser.parse_args()


def load_2d(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.squeeze(np.asanyarray(image.dataobj))
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D LGE mask: {path} shape={data.shape}")
    return data > 0, image


def numeric_stem(path: Path) -> float:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    try:
        return float(stem)
    except ValueError:
        return float("inf")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read_rows(manifest: Path, limit: int) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("patient", ""), row.get("slice", ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def candidate_metrics(original: np.ndarray, candidate: np.ndarray, myo: np.ndarray) -> dict[str, Any]:
    added = candidate & ~original
    removed = original & ~candidate
    return {
        "changed_pixels": int(np.count_nonzero(candidate != original)),
        "added_pixels": int(np.count_nonzero(added)),
        "removed_pixels": int(np.count_nonzero(removed)),
        "outside_myo_pixels": int(np.count_nonzero(candidate & ~myo)),
        "candidate_pixels": int(np.count_nonzero(candidate)),
    }


def evaluate_lge(root: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    params = ShadowParameters()
    for row in rows:
        patient, name = row["patient"], row["slice"]
        myo_path = root / "labels/lge_checked/lge_Myo_labels" / patient / name
        mi_path = root / "labels/lge_checked/lge_MI_labels" / patient / name
        if not myo_path.exists() or not mi_path.exists():
            skipped.append({"patient": patient, "slice": name, "reason": "missing_label"})
            continue
        try:
            myo, image = load_2d(myo_path)
            mi, _ = load_2d(mi_path)
        except (ValueError, OSError) as exc:
            skipped.append({"patient": patient, "slice": name, "reason": str(exc)})
            continue
        if myo.shape != mi.shape:
            skipped.append({"patient": patient, "slice": name, "reason": "shape_mismatch"})
            continue
        layer = int(float(row.get("layer_index") or 0))
        features = extract_slice_features(f"{patient}/{name}", layer, mi, myo)
        candidates = simulate_candidates(mi, myo, parameters=params)
        zooms = image.header.get_zooms()
        voxel_area = float(zooms[0] * zooms[1]) if len(zooms) >= 2 else 1.0
        advisory = evaluate_mi_fragment_rule(features, voxel_area_mm2=voxel_area)
        variants = {key: candidate_metrics(mi, value["mask"], myo) | {
            "pseudo_flag": bool(value["pseudo_flag"])
        } for key, value in candidates.items()}
        output.append({
            "patient": patient,
            "slice": name,
            "layer_index": layer,
            "source_issues": row.get("issues", ""),
            "features": features,
            "advisory": advisory,
            "variants": variants,
            "source_hashes": {"myo": hash_file(myo_path), "mi": hash_file(mi_path)},
        })
    summary = {
        "selected": len(rows),
        "evaluated": len(output),
        "skipped": skipped,
        "advisory_positive": sum(bool(item["advisory"]["pseudo_flag"]) for item in output),
        "variant_flag_counts": dict(Counter(
            name for item in output for name, value in item["variants"].items() if value["pseudo_flag"]
        )),
    }
    return output, summary


def evaluate_topology(root: Path, rows: list[dict[str, str]], max_patients: int) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(row)
    reports: dict[str, Any] = {}
    for patient in sorted(by_patient)[:max_patients]:
        myo_dir = root / "labels/lge_checked/lge_Myo_labels" / patient
        mi_dir = root / "labels/lge_checked/lge_MI_labels" / patient
        names = sorted((path for path in myo_dir.glob("*.nii.gz") if (mi_dir / path.name).exists()), key=numeric_stem)
        myo_stack: list[np.ndarray] = []
        mi_stack: list[np.ndarray] = []
        positions: list[float] = []
        for path in names:
            try:
                myo, _ = load_2d(path)
                mi, _ = load_2d(mi_dir / path.name)
            except (ValueError, OSError):
                continue
            if myo.shape != mi.shape or (myo_stack and myo.shape != myo_stack[0].shape):
                continue
            myo_stack.append(myo)
            mi_stack.append(mi)
            positions.append(numeric_stem(path))
        if len(myo_stack) >= 3:
            reports[patient] = evaluate_topology_shadow(
                myo_stack, mi_stack, band_slices=myo_stack, slice_positions=positions
            )
    return reports


def verify_source_hashes(root: Path, evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    changed: list[dict[str, str]] = []
    for item in evaluated:
        patient, name = item["patient"], item["slice"]
        paths = {
            "myo": root / "labels/lge_checked/lge_Myo_labels" / patient / name,
            "mi": root / "labels/lge_checked/lge_MI_labels" / patient / name,
        }
        for label, path in paths.items():
            if hash_file(path) != item["source_hashes"][label]:
                changed.append({"patient": patient, "slice": name, "label": label})
    return {"checked_files": len(evaluated) * 2, "changed_files": changed, "unchanged": not changed}


def latest_safe_summaries(root: Path) -> dict[str, Any]:
    patterns = {
        "tiny_mi": "label_iteration/casebook_runs/extended_tiny_*/summary.json",
        "mi_merge": "label_iteration/mi_merge_runs/*/summary.json",
        "trajectory": "label_iteration/myo_trajectory_runs/*/summary.json",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        matches = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
        if matches:
            result[key] = json.loads(matches[-1].read_text(encoding="utf-8"))
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = root / "labels/check/manifest_backup/lge_middle_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "label_iteration/agent_closed_loop_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = read_rows(manifest, args.max_slices)
    evaluated, lge_summary = evaluate_lge(root, rows)
    topology = evaluate_topology(root, rows, args.max_topology_patients)
    safe = latest_safe_summaries(root)
    safe_candidates = {
        "tiny_mi": int(safe.get("tiny_mi", {}).get("candidates", 0)),
        "mi_merge": int(safe.get("mi_merge", {}).get("candidates", 0)),
        "trajectory": int(safe.get("trajectory", {}).get("candidate_actions", 0)),
    }
    experimental_changes = sum(
        value["changed_pixels"] > 0
        for item in evaluated for name, value in item["variants"].items() if name != "no_op"
    )
    source_integrity = verify_source_hashes(root, evaluated)
    gate = {
        "decision": "REJECT_EXPERIMENTAL_WRITEBACK",
        "production_safe_candidates": safe_candidates,
        "production_write_count": sum(safe_candidates.values()),
        "experimental_candidate_slice_variants": experimental_changes,
        "reasons": [
            "A1/A2/A3 are shadow/advisory experiments and have no adjudicated reference Dice.",
            "False-repair rate and MI/Myo Dice non-regression cannot be established from issue flags alone.",
            "All currently validated deterministic repair tools returned zero eligible candidates.",
        ],
        "source_labels_mutated": False,
        "source_integrity": source_integrity,
    }
    report = {
        "root": str(root),
        "run_dir": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "stages": ["analysis", "shadow_execution", "validation_gate", "writeback_decision"],
        "lge": lge_summary,
        "topology_patients_evaluated": len(topology),
        "topology_status": "evaluated" if topology else "blocked_by_A2_batch_performance_validation",
        "safe_tool_summaries": safe,
        "gate": gate,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    (run_dir / "candidate_details.json").write_text(json.dumps(evaluated, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    (run_dir / "topology_shadow.json").write_text(json.dumps(topology, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    with (run_dir / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "slice", "source_issues", "advisory_positive", "largest_changed", "small_remove_changed", "merge_changed"])
        writer.writeheader()
        for item in evaluated:
            writer.writerow({
                "patient": item["patient"], "slice": item["slice"], "source_issues": item["source_issues"],
                "advisory_positive": item["advisory"]["pseudo_flag"],
                "largest_changed": item["variants"]["largest_component_only"]["changed_pixels"],
                "small_remove_changed": item["variants"]["absolute_small_area_removal"]["changed_pixels"],
                "merge_changed": item["variants"]["restricted_morphological_merge"]["changed_pixels"],
            })
    (run_dir / "writeback_decision.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
