#!/usr/bin/env python3
"""Preflight a private training JSON; --execute explicitly starts one attempt.

See app.segmentation_training's module docstring for the complete v1 job schema.
This command grants no reviewer authorization and never promotes a candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.segmentation_training import JobBusy, JobError, read_json, run_job, worker_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help="local JSON job (required outside private worker mode)")
    parser.add_argument("--execute", action="store_true", help="explicitly launch training; default is preflight")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.execute or args.job is not None:
            parser.error("private worker requires --execute and stdin payload, without --job")
        worker_main()
        return 0
    if args.job is None:
        parser.error("--job is required")
    try:
        result = run_job(read_json(args.job), execute=args.execute)
    except (JobError, JobBusy, OSError, ValueError) as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("PREFLIGHT", "TRAINED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
