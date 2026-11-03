#!/usr/bin/env python3
"""Refresh HANDOFF.md while a workflow run is active.

This monitor is deliberately read-only with respect to medical data and run
state.  It records status for handoff; the worker remains the sole executor.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from update_handoff import DB_PATH, update_handoff  # noqa: E402


def run_status(run_id: int) -> tuple[str, int]:
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT status FROM workflow_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        open_tasks = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE run_id=? AND status IN ('QUEUED','RUNNING')",
            (run_id,),
        ).fetchone()[0]
    return str(row[0]), int(open_tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    previous = None
    while True:
        update_handoff(verbose=False)
        status, open_tasks = run_status(args.run_id)
        state = (status, open_tasks)
        if state != previous:
            print(f"run_id={args.run_id} status={status} open_tasks={open_tasks}", flush=True)
            previous = state
        if open_tasks == 0 and status not in {"QUEUED", "RUNNING"}:
            break
        time.sleep(max(args.interval, 10.0))


if __name__ == "__main__":
    main()
