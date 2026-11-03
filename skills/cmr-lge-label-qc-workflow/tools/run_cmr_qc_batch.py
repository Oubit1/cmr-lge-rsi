#!/usr/bin/env python3
"""Run CMR QC in patient chunks so large datasets do not time out."""
from __future__ import annotations

import argparse
from pathlib import Path

from cmr_qc import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "apply-safe"], default="audit")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    patient_count = len([p for p in (args.root / "images/cmr").iterdir() if p.is_dir()])
    for start in range(0, patient_count, args.chunk_size):
        audit(args.root.resolve(), args.mode, render_checks=not args.no_render,
              start_patient=start, max_patients=args.chunk_size)


if __name__ == "__main__":
    main()
