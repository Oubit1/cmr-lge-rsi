#!/usr/bin/env python3
"""Pre-register a stratified, immutable check holdout before rule tuning.

The identity-bearing artifact stays on the remote host. Neither this command
nor the subsequent shadow experiments may mutate a label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"adequate_area_low_cavity": 37,
         "low_area_low_cavity": 12,
         "low_area_adequate_cavity": 9}


def phenotype(row: dict) -> str:
    if row["decision"] != "REVIEW":
        raise ValueError("holdout must contain only currently pending review cases")
    low_area = row["after_area"] < .45 * row["reference_area"]
    low_cavity = row["after_cavity"] < .60
    return ("low_area_" if low_area else "adequate_area_") + (
        "low_cavity" if low_cavity else "adequate_cavity")


def freeze(evaluation: dict) -> dict:
    by_stratum = defaultdict(list)
    for row in evaluation["records"]:
        by_stratum[phenotype(row)].append(row["key"])
    if set(by_stratum) != set(SIZES):
        raise ValueError("unexpected phenotype distribution")
    holdout, development = [], []
    for group, target in SIZES.items():
        cases = sorted(by_stratum[group], key=lambda key: hashlib.sha256(
            ("cmr-check-holdout-v1:" + key).encode()).hexdigest())
        if len(cases) < target:
            raise ValueError(f"insufficient cases in {group}")
        holdout.extend({"key": key, "stratum": group} for key in cases[:target])
        development.extend({"key": key, "stratum": group} for key in cases[target:])
    if len(holdout) != 58 or len(development) != 135:
        raise ValueError("pending population changed from preregistered 193")
    canonical = json.dumps(sorted(holdout, key=lambda item: item["key"]),
                           sort_keys=True, separators=(",", ":"))
    return {"schema_version": "cmr-check-holdout-v1", "split_method": "sha256_key_v1",
            "holdout_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "holdout": holdout, "development": development,
            "counts": {"holdout": len(holdout), "development": len(development)},
            "strata": SIZES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "runtime/check_iterations/holdout_v1.json")
    args = ap.parse_args()
    split = freeze(json.loads(args.evaluation.read_text()))
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing != split:
            raise RuntimeError("frozen holdout differs; refusing overwrite")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": split["counts"],
                      "holdout_sha256": split["holdout_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
