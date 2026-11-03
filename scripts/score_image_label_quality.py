#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.quality_scoring import (  # noqa: E402
    add_interslice_context,
    build_profile,
    extract_pair,
    load_specs,
    score_record,
    write_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score original CMR/LGE image-label pairs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/quality_scoring_datasets.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/image_label_quality_v1")
    parser.add_argument("--limit", type=int, default=0, help="debug-only number of pairs")
    parser.add_argument("--per-group-limit", type=int, default=0,
                        help="debug-only cap for each dataset/target group")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    specs, config = load_specs(args.config)
    if args.per_group_limit:
        counts: dict[tuple[str, str], int] = {}
        selected = []
        for spec in specs:
            key = (spec.dataset, spec.target)
            if counts.get(key, 0) < args.per_group_limit:
                selected.append(spec)
                counts[key] = counts.get(key, 0) + 1
        specs = selected
    if args.limit:
        specs = specs[:args.limit]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, record in enumerate(executor.map(extract_pair, specs, chunksize=8), 1):
            records.append(record)
            if index % 250 == 0:
                print(json.dumps({"processed": index, "total": len(specs)}, ensure_ascii=False), flush=True)
    add_interslice_context(records)
    profile = build_profile(records)
    scored = [score_record(record, profile) for record in records]
    write_reports(scored, profile, args.output)
    provenance = {
        "config": str(args.config.resolve()),
        "datasets": config["datasets"],
        "pair_count": len(scored),
        "output": str(args.output.resolve()),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "pairs": len(scored), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
