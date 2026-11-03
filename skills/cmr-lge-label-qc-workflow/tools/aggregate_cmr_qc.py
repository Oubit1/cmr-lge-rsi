#!/usr/bin/env python3
"""Aggregate chunked CMR QC manifests and optionally render flat check images."""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from cmr_qc import as_stack, load, render


FIELDS = ["patient", "volume", "slice_index", "issues", "evidence", "check"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="chunk run directory; repeat once per completed chunk")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    seen = set()
    for run in args.run:
        manifest = run.resolve() / "issue_manifest.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["patient"], row["volume"], row["slice_index"], row["issues"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = root / "label_iteration/cmr_qc_runs" / f"aggregate_{stamp}"
    out_run.mkdir(parents=True, exist_ok=True)
    issue_root = root / "labels/check/cmr_issue_slices"
    issue_root.mkdir(parents=True, exist_ok=True)
    for old_png in issue_root.glob("*.png"):
        old_png.unlink()

    if not args.no_render:
        cache = {}
        for row in rows:
            image_path = root / "images/cmr" / row["patient"] / row["volume"]
            label_path = root / "labels/cmr/cmr_Myo_mask" / row["patient"] / row["volume"]
            if not image_path.exists() or not label_path.exists() or not row["slice_index"]:
                continue
            cache_key = str(image_path)
            if cache_key not in cache:
                _, image_raw = load(image_path)
                _, label_raw = load(label_path)
                cache[cache_key] = (as_stack(image_raw)[0], as_stack(label_raw)[0])
            image_stack, label_stack = cache[cache_key]
            idx = int(row["slice_index"]) - 1
            if idx < 0 or idx >= image_stack.shape[-1]:
                continue
            filename = f"{row['patient']}_{row['volume'][:-7]}_s{idx + 1:02d}.png"
            check = issue_root / filename
            render(image_stack[..., idx], label_stack[..., idx].astype(bool), check,
                   f"{row['patient']} | {row['volume']} | slice {idx + 1} | {row['issues']}")
            row["check"] = str(check)

    with (out_run / "issue_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(out_run / "issue_manifest.csv", root / "labels/check/cmr_issue_slices/issue_manifest.csv")
    print({"candidate_slices": len(rows), "run": str(out_run), "check_root": str(issue_root)})


if __name__ == "__main__":
    main()
