#!/usr/bin/env python3
"""Dispatch a reviewer-approved check experiment to its token-bound owner.

This creates only an isolated maintenance snapshot. It never promotes source
files or writes a medical label; deterministic and held-out gates are separate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.llm_maintenance import run_llm_maintenance  # noqa: E402
from agents.governance import agent_action_authorized  # noqa: E402
from app.database import Database  # noqa: E402

import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
SKILL = _public_paths.SKILL_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "runtime/workflow.sqlite")
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--action-id", required=True)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "runtime/check_iterations/agent_experiments")
    args = ap.parse_args()
    db = Database(args.db)
    db.initialize()
    row = db.get_maintenance_action(args.run_id, args.action_id)
    action = row["payload"]
    if not agent_action_authorized(db, args.run_id, row, action):
        raise SystemExit("action lacks persisted supervisor assignment and independent approval")
    action["handoff_messages"] = db.agent_inbox(
        args.run_id, row["owner_role"],
        thread_id=f"run-{args.run_id}:{args.action_id}",
    )
    db.update_maintenance_action(args.run_id, args.action_id, status="RUNNING")
    try:
        result = run_llm_maintenance(
            action=action, output_root=args.output, orchestrator_source=ROOT,
            skill_source=SKILL, providers_path=ROOT / "configs/providers.yaml",
            agents_path=ROOT / "configs/agents.aliyun-roles.yaml",
            provider_role=action["owner_role"], timeout_seconds=240,
            db=db, run_id=args.run_id)
        status = "SUCCEEDED" if result["result"].get("ready_for_validation") else "HUMAN_REQUIRED"
        db.update_maintenance_action(
            args.run_id, args.action_id, status=status,
            artifact_path=Path(result["workspace"]),
        )
    except Exception:
        db.update_maintenance_action(args.run_id, args.action_id, status="HUMAN_REQUIRED")
        raise
    print(json.dumps({"action_id": action["action_id"],
                      "owner_role": action["owner_role"],
                      "workspace": result["workspace"],
                      "executor": result["result"].get("executor"),
                      "ready_for_validation": result["result"].get("ready_for_validation"),
                      "changed_files": result["result"].get("changed_files")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
