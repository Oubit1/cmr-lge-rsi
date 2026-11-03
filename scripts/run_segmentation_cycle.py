#!/usr/bin/env python3
"""Run an externally reviewed segmentation plan; read-only preflight by default.

No role/provider dispatch, approval synthesis, budget grants or production label
writes. See app.segmentation_host for required independent DB evidence messages.
Exit 0: completed/preflight; 2: invalid/failed; 3: waiting/paused/cancelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.segmentation_cycle import run_cycle
from app.segmentation_host import SourceUnavailable, make_controller, public_result
from app.segmentation_training import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Trusted control/project root")
    parser.add_argument("--plan", type=Path, required=True, help="Immutable reviewed cycle plan JSON")
    parser.add_argument("--database", type=Path, required=True, help="Existing trusted governance DB; opened read-only")
    parser.add_argument("--prediction-seconds", type=float, help="Optional lower prediction walltime cap")
    parser.add_argument("--baseline-vendor", type=Path,
                        help="Existing baseline target vendor directory, verified and frozen once on execution")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = read_json(args.plan)
        controller = make_controller(args.root.absolute(), plan, database=args.database.absolute(),
                                     prediction_seconds=args.prediction_seconds,
                                     baseline_vendor=args.baseline_vendor.absolute() if args.baseline_vendor else None)
        result = run_cycle(args.root.absolute(), plan, controller=controller, execute=args.execute)
        summary = public_result(result, controller, plan)
        summary["execute"] = args.execute
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        if not args.execute:
            return 0
        if result["stage"] == "FAILED":
            return 2
        return 0 if result["stage"] in ("ACTIVATED", "RETAIN_PARENT") else 3
    except SourceUnavailable:
        print(json.dumps({"stage": "AWAITING_MODEL_SOURCES", "execute": args.execute,
                          "missing_evidence": ["model_source_bundle_or_training_receipt"]}))
        return 3
    except Exception as exc:
        # Full source/DB/worker exceptions can contain private paths. CLI exposes
        # only the class; private cycle journals/logs carry execution details.
        print(json.dumps({"stage": "ERROR", "error_code": type(exc).__name__, "execute": args.execute}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
