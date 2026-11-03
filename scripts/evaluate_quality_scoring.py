#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.quality_scoring import PairSpec, build_profile, extract_pair, score_record  # noqa: E402


def pair(record: dict, label_key: str) -> PairSpec:
    return PairSpec(
        dataset=record.get("dataset", "eval"), tier=record.get("tier", "gold"),
        target="cmr_myo", image=Path(record["image"]), label=Path(record[label_key]),
        myo=None, relative_id=record.get("key", Path(record[label_key]).name),
        case_id=record.get("key", Path(record[label_key]).stem), slice_position=0,
        reviewed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate quality scores on frozen injected defects")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "runtime/eval_set/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/image_label_quality_v1/defect_validation.json")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    clean_rows = [json.loads(line) for line in (args.eval_dir / "manifest.jsonl").read_text().splitlines() if line]
    clean_rows = [row for row in clean_rows if row["modality"] == "cmr" and row["tier"] in {"gold", "quasi"}]
    synthetic_rows = [json.loads(line) for line in (args.eval_dir / "synthetic/manifest.jsonl").read_text().splitlines() if line]
    clean_specs = [pair(row, "label") for row in clean_rows]
    synthetic_specs = [pair(row, "synthetic_label") for row in synthetic_rows]
    counterpart_specs = [pair(row, "clean_label") for row in synthetic_rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        clean_records = list(executor.map(extract_pair, clean_specs, chunksize=8))
        synthetic_records = list(executor.map(extract_pair, synthetic_specs, chunksize=8))
        counterpart_records = list(executor.map(extract_pair, counterpart_specs, chunksize=8))
    profile = build_profile(clean_records)
    clean_scores = np.asarray([score_record(row, profile)["score"] for row in clean_records], dtype=float)
    synthetic_scores = [score_record(row, profile)["score"] for row in synthetic_records]
    counterpart_scores = [score_record(row, profile)["score"] for row in counterpart_records]
    grouped = defaultdict(list)
    for source, injected, clean in zip(synthetic_rows, synthetic_scores, counterpart_scores):
        grouped[source["class"]].append((injected, clean))
    report = {
        "schema_version": "quality-score-defect-validation-v1",
        "clean_reference": {
            "n": int(len(clean_scores)), "mean": round(float(clean_scores.mean()), 3),
            "below_80": round(float(np.mean(clean_scores < 80)), 3),
            "below_70": round(float(np.mean(clean_scores < 70)), 3),
        },
        "classes": {},
    }
    for name, values in sorted(grouped.items()):
        injected = np.asarray([x[0] for x in values], dtype=float)
        clean = np.asarray([x[1] for x in values], dtype=float)
        delta = clean - injected
        report["classes"][name] = {
            "n": len(values), "clean_mean": round(float(clean.mean()), 3),
            "injected_mean": round(float(injected.mean()), 3),
            "mean_score_drop": round(float(delta.mean()), 3),
            "detected_drop_ge_5": round(float(np.mean(delta >= 5)), 3),
            "detected_drop_ge_10": round(float(np.mean(delta >= 10)), 3),
            "injected_below_70": round(float(np.mean(injected < 70)), 3),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
