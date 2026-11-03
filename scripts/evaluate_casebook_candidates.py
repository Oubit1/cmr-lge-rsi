#!/usr/bin/env python3
"""Replay experimental MI candidates on historical before/after casebook pairs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from experiments.lge_shadow import ShadowParameters, simulate_candidates


LABEL_MARKER = Path("labels/lge_checked/lge_MI_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--casebook-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-passed-controls", type=int, default=0,
                        help="0 evaluates every QC-passed LGE slice")
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    data = np.squeeze(np.asanyarray(nib.load(str(path)).dataobj))
    if data.ndim != 2:
        raise ValueError(f"not 2-D: {path} shape={data.shape}")
    return data > 0


def dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(np.count_nonzero(left) + np.count_nonzero(right))
    return 1.0 if denominator == 0 else float(2 * np.count_nonzero(left & right) / denominator)


def relative_target(path: Path) -> Path | None:
    parts = path.parts
    marker = LABEL_MARKER.parts
    for index in range(len(parts) - len(marker) + 1):
        if parts[index:index + len(marker)] == marker:
            tail = parts[index + len(marker):]
            return Path(*tail) if len(tail) == 2 else None
    return None


def collect_pairs(root: Path) -> list[dict[str, Any]]:
    iteration = root / "label_iteration"
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in iteration.rglob("*.nii.gz"):
        if "backup" not in str(path).lower() and not path.name.endswith("_before.nii.gz"):
            continue
        relative = relative_target(path)
        if relative is not None:
            grouped[relative].append(path)

    # Explicit manual patches use a different layout and are promoted to tier A.
    manual: dict[Path, Path] = {}
    patch_root = iteration / "manual_patches"
    if patch_root.exists():
        for path in patch_root.rglob("lge_MI_before.nii.gz"):
            patient = path.parents[1].name
            slice_stem = path.parent.name
            candidates = list((root / "labels/lge_checked/lge_MI_labels" / patient).glob(f"{slice_stem}*.nii.gz"))
            if len(candidates) == 1:
                manual[Path(patient) / candidates[0].name] = path

    pairs: list[dict[str, Any]] = []
    for relative, backups in grouped.items():
        after = root / LABEL_MARKER / relative
        myo = root / "labels/lge_checked/lge_Myo_labels" / relative
        if not after.exists() or not myo.exists():
            continue
        before = manual.get(relative) or max(backups, key=lambda item: item.stat().st_mtime)
        pairs.append({
            "patient": relative.parts[0], "slice": relative.parts[1],
            "before": before, "after": after, "myo": myo,
            "tier": "A_manual" if relative in manual else "C_retrospective",
        })
    for relative, before in manual.items():
        if any(item["patient"] == relative.parts[0] and item["slice"] == relative.parts[1] for item in pairs):
            continue
        after = root / LABEL_MARKER / relative
        myo = root / "labels/lge_checked/lge_Myo_labels" / relative
        if after.exists() and myo.exists():
            pairs.append({"patient": relative.parts[0], "slice": relative.parts[1],
                          "before": before, "after": after, "myo": myo, "tier": "A_manual"})
    return pairs


def collect_passed_controls(root: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Treat slices absent from every current LGE issue pool as accepted GT."""
    issue_keys: set[tuple[str, str]] = set()
    manifest = root / "labels/check/manifest_backup/lge_issue_manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                patient, name = row.get("patient"), row.get("slice")
                if patient and name:
                    issue_keys.add((patient, name))
    issue_dirs = [
        root / "labels/check/lge_issue_slices",
        root / "labels/check/lge_isolated_component_slices",
        root / "labels/check/lge_myo_gap_slices",
    ]
    mi_root = root / LABEL_MARKER
    controls: list[dict[str, Any]] = []
    for after in sorted(mi_root.glob("*/*.nii.gz")):
        relative = after.relative_to(mi_root)
        patient, name = relative.parts
        stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
        has_issue_png = any((directory / f"{patient}_{stem}.png").exists() for directory in issue_dirs)
        if (patient, name) in issue_keys or has_issue_png:
            continue
        myo = root / "labels/lge_checked/lge_Myo_labels" / relative
        if not myo.exists():
            continue
        controls.append({"patient": patient, "slice": name, "before": after,
                         "after": after, "myo": myo, "tier": "B_qc_passed"})
        if limit and len(controls) >= limit:
            break
    return controls


def main() -> None:
    args = parse_args()
    root, output = args.casebook_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_pairs = collect_pairs(root)
    source_pairs.extend(collect_passed_controls(root, args.max_passed_controls))
    for item in source_pairs:
        try:
            before, after, myo = load_mask(item["before"]), load_mask(item["after"]), load_mask(item["myo"])
        except (OSError, ValueError) as exc:
            skipped.append({"patient": item["patient"], "slice": item["slice"], "reason": str(exc)})
            continue
        if before.shape != after.shape or before.shape != myo.shape:
            skipped.append({"patient": item["patient"], "slice": item["slice"], "reason": "shape_mismatch"})
            continue
        baseline = dice(before, after)
        candidates = simulate_candidates(before, myo, parameters=ShadowParameters())
        scores = {name: dice(value["mask"], after) for name, value in candidates.items()}
        changed = {name: bool(np.any(value["mask"] != before)) for name, value in candidates.items()}
        best_name = max(scores, key=scores.get)
        rows.append({
            "patient": item["patient"], "slice": item["slice"], "tier": item["tier"],
            "baseline_dice": baseline, **{f"{name}_dice": score for name, score in scores.items()},
            "best_candidate": best_name, "best_dice": scores[best_name],
            "best_improvement": scores[best_name] - baseline,
            "before_pixels": int(np.count_nonzero(before)), "after_pixels": int(np.count_nonzero(after)),
            **{f"{name}_changed": value for name, value in changed.items()},
        })

    by_tier: dict[str, Any] = {}
    for tier in sorted({row["tier"] for row in rows}):
        subset = [row for row in rows if row["tier"] == tier]
        by_tier[tier] = {
            "count": len(subset),
            "mean_baseline_dice": float(np.mean([row["baseline_dice"] for row in subset])),
            "mean_best_dice": float(np.mean([row["best_dice"] for row in subset])),
            "improved": sum(row["best_improvement"] > 1e-6 for row in subset),
            "regressed": sum(row["best_improvement"] < -1e-6 for row in subset),
            "best_candidate_counts": {name: sum(row["best_candidate"] == name for row in subset)
                                      for name in sorted({row["best_candidate"] for row in subset})},
            "candidate_metrics": {
                name: {
                    "mean_dice": float(np.mean([row[f"{name}_dice"] for row in subset])),
                    "changed": sum(bool(row[f"{name}_changed"]) for row in subset),
                    "change_rate": float(np.mean([bool(row[f"{name}_changed"]) for row in subset])),
                }
                for name in ("largest_component_only", "absolute_small_area_removal",
                             "restricted_morphological_merge", "no_op")
            },
        }
    manual_count = by_tier.get("A_manual", {}).get("count", 0)
    accepted_count = by_tier.get("B_qc_passed", {}).get("count", 0)
    accepted_metrics = by_tier.get("B_qc_passed", {}).get("candidate_metrics", {})
    safe_candidates = [name for name, metrics in accepted_metrics.items()
                       if metrics["change_rate"] <= 0.01]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "casebook_root": str(root), "evaluated_pairs": len(rows), "skipped": skipped,
        "tiers": by_tier,
        "promotion_gate": {
            "decision": "METRIC_REVIEW_REQUIRED" if accepted_count else "REJECT",
            "manual_gold_pairs": manual_count,
            "accepted_qc_gt_pairs": accepted_count,
            "max_false_repair_rate": 0.01,
            "candidates_passing_accepted_gt_false_repair_gate": safe_candidates,
            "reason": "QC-passed slices define the false-repair gate; retrospective pairs test recovery."
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else ["patient", "slice", "tier"]
    with (output / "pair_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
