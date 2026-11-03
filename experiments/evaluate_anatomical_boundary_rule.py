#!/usr/bin/env python3
"""Sensitivity analysis for the shadow CMR anatomical-boundary rule."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fill_single_gaps(values):
    out = list(values)
    for index in range(1, len(out) - 1):
        if not out[index] and out[index - 1] and out[index + 1]:
            out[index] = True
    return out


def runs(values):
    result, start = [], None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--middle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = read_csv(args.middle_manifest)
    patient_metrics = {item["patient"]: item["volumes"] for item in report["patients"]}
    results = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        for minimum_core in (3, 4, 5):
            boundary = set()
            reliable_patients = 0
            ambiguous_segments = 0
            for patient, volumes in patient_metrics.items():
                trusted = fill_single_gaps([
                    float(volume["ring_fraction"]) >= threshold for volume in volumes
                ])
                segments = runs(trusted)
                valid = [segment for segment in segments
                         if segment[1] - segment[0] + 1 >= minimum_core]
                if len(segments) > 1:
                    ambiguous_segments += 1
                if len(segments) != 1 or len(valid) != 1:
                    continue
                start, end = valid[0]
                reliable_patients += 1
                for index, volume in enumerate(volumes):
                    if index < start or index > end:
                        boundary.add((patient, volume["volume"]))
            selected = [row for row in rows
                        if (row["patient"], row["volume"]) in boundary]
            results.append({
                "ring_fraction": threshold,
                "minimum_core_volumes": minimum_core,
                "candidate_rows": len(selected),
                "unique_checks": len({row.get("check", "") for row in selected}),
                "patients": len({row["patient"] for row in selected}),
                "volumes": len({(row["patient"], row["volume"]) for row in selected}),
                "reliable_patients": reliable_patients,
                "ambiguous_multi_segment_patients": ambiguous_segments,
                "internal_failure_moves": 0,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"output": str(args.output), "grid_points": len(results),
                      "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
