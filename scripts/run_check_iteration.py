#!/usr/bin/env python3
"""Reproducible, label-read-only CMR check iteration and residual accounting.

The pipeline stages candidate edits, validates each against the current source,
then evaluates the complete spatial volume in memory. A passing pixel gate is
not a clinical acceptance and is never applied by this command.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
DATASET = _public_paths.DATASET_ROOT


def summarize(candidate_root: Path, dataset: Path) -> dict:
    screen = json.loads((candidate_root / "cmr_boundary_manifest_report.json").read_text())
    gate = json.loads((candidate_root / "candidate_validation.json").read_text())
    evaluation = json.loads((candidate_root / "evaluation.json").read_text())
    check = dataset / "labels/check"
    count = lambda state: sum(1 for _ in (check / state / "cmr").glob("*/*.png"))
    clusters = Counter()
    for row in evaluation["records"]:
        if row["decision"] == "COUNTERFACTUAL_SALVAGE":
            continue
        if "after_area" not in row or "reference_area" not in row:
            clusters[row["decision"]] += 1
            continue
        low_area = row["after_area"] < .45 * row["reference_area"]
        low_cavity = row["after_cavity"] < .60
        clusters[("low_area_" if low_area else "adequate_area_")
                 + ("low_cavity" if low_cavity else "adequate_cavity")] += 1
    pending = count("待筛")
    if evaluation.get("total_volumes", evaluation.get("total_pending")) != pending:
        raise RuntimeError("pending count changed during evaluation; refusing to publish")
    agent_path = candidate_root / "agent_rule_review.json"
    agent = json.loads(agent_path.read_text()) if agent_path.is_file() else {}
    spatial_path = candidate_root / "spatial_boundary_shadow.json"
    spatial = json.loads(spatial_path.read_text()) if spatial_path.is_file() else {}
    quality = evaluation.get("quality") or {"status": "NOT_RUN", "scored_volumes": 0}
    holdout_path = ROOT / "runtime/check_iterations/holdout_v1.json"
    holdout = json.loads(holdout_path.read_text()) if holdout_path.is_file() else {}
    holdout_keys = {row.get("key") for row in holdout.get("holdout", [])
                    if isinstance(row, dict)}
    pending_keys = {
        f"{path.parent.name}/{path.name.removesuffix('.png')}.nii.gz"
        for path in (check / "待筛" / "cmr").glob("*/*.png")
    }
    manifest = dataset / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
    identities: list[str] = []
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            identities = [f'{row["patient"]}/{row["volume"]}' for row in csv.DictReader(handle)]
    repair_keys = set()
    for row in gate.get("records", []):
        if row.get("pool") != "cmr_boundary_manifest":
            continue
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(identities):
            repair_keys.add(identities[index])
    spatial_keys = {
        row.get("key") for row in spatial.get("records", [])
        if row.get("state") == "待筛"
        and row.get("decision") == "spatial_boundary_candidate"
    }
    route_conflicts = pending_keys & repair_keys & spatial_keys - holdout_keys
    promotion_log = ROOT / "runtime/promotion/promotion_log.jsonl"
    promotion = Counter()
    if promotion_log.is_file():
        for line in promotion_log.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            promotion[item.get("action", "unknown")] += 1
            promotion["fixed_frames"] += len(item.get("frames") or [])
    return {
        "schema_version": "check-iteration-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "MEASURED_NO_LABEL_WRITE",
        "pending": pending, "work": count("待处理"),
        "checked": count("checked"), "trash": count("垃圾桶"),
        "scanned_frames": screen["scanned"],
        "candidate_frames": gate["total"], "gate_passed_frames": gate["passed"],
        "volumes_with_candidates": evaluation["volumes_with_candidates"],
        "counterfactual_salvage": evaluation["decisions"].get("COUNTERFACTUAL_SALVAGE", 0),
        "residual_pending": pending - evaluation["decisions"].get("COUNTERFACTUAL_SALVAGE", 0),
        "residual_clusters": dict(clusters),
        "candidate_quality": quality,
        "promotion": dict(promotion),
        "agent_decision": (agent.get("supervisor") or {}).get("decision"),
        "reviewer_decision": (agent.get("independent_reviewer") or {}).get("decision"),
        "holdout_count": (holdout.get("counts") or {}).get("holdout"),
        "holdout_sha256": holdout.get("holdout_sha256"),
        "pending_holdout": len(pending_keys & holdout_keys),
        "pending_non_holdout": len(pending_keys - holdout_keys),
        "pending_route_conflicts": len(route_conflicts),
        "control_gate_status": spatial.get("control_gate", "NOT_RUN_FOR_NEW_RULE"),
        "spatial_boundary_candidates": spatial.get("pending_spatial_boundary_candidates"),
        "checked_false_routes": spatial.get("checked_false_routes"),
        "next_step": ("旧 SAX 核心规则与已晋升样本冲突，控制闸失败；空间边界候选仅供诊断，不得自动移动"
                      if spatial.get("control_gate") == "FAIL" else
                      "待新规则通过独立测试与误修门槛；现有局部候选未使待筛空间层达标"),
        "report_path": str(candidate_root / "evaluation.json"),
    }


def run_script(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--candidate-root", type=Path,
                    help="Reuse a completed immutable probe; otherwise run a fresh one")
    args = ap.parse_args()
    out = args.candidate_root or ROOT / "runtime/check_iterations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.candidate_root is None:
        out.mkdir(parents=True, exist_ok=False)
        run_script("batch_screen.py", "--dataset", str(args.dataset),
                   "--pool", "cmr_boundary_manifest", "--review-status", "pending", "--out", str(out))
        run_script("validate_batch_candidates.py", "--dataset", str(args.dataset),
                   "--candidates", str(out), "--output", str(out / "candidate_validation.json"))
        quality_profile = ROOT / "artifacts/image_label_quality_v1/calibration_profile.json"
        quality_args = (["--quality-profile", str(quality_profile)]
                        if quality_profile.is_file() else [])
        run_script("evaluate_pending_check.py", "--dataset", str(args.dataset),
                   "--candidates", str(out), "--state", "待筛",
                   *quality_args, "--output", str(out / "evaluation.json"))
        run_script("shadow_spatial_boundary.py", "--dataset", str(args.dataset),
                   "--output", str(out / "spatial_boundary_shadow.json"))
    summary = summarize(out, args.dataset)
    latest = ROOT / "runtime/check_iterations/latest.json"
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
