#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.shadow_analysis import run_shadow_analysis
from agents.governance import agent_action_authorized
from app.database import Database
from app.existing_qc import snapshot_existing_qc
from app.orchestrator import WorkflowOrchestrator
from app.state_machine import RunState
from experiments.promotion_gate import evaluate_metrics
from experiments.worktree_manager import WorktreeManager
from experiments.llm_maintenance import run_llm_maintenance
from contracts.compat import model_dump


DEFAULT_DB = ROOT / "runtime/workflow.sqlite"
import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import paths as _public_paths
DEFAULT_SKILL = _public_paths.SKILL_ROOT
DEFAULT_PYTHON = _public_paths.PYTHON
DEFAULT_SKILL_SOURCE = _public_paths.SKILL_ROOT


def dump(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def database(path: Path) -> Database:
    db = Database(path)
    db.initialize()
    return db


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Persistent CMR/LGE workflow orchestrator")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="initialize the SQLite database")

    submit = sub.add_parser("submit", help="register a dataset workflow run")
    submit.add_argument("--dataset", type=Path, required=True)
    submit.add_argument("--raw-input", type=Path)
    submit.add_argument("--mode", choices=("audit", "apply"), default="audit")
    submit.add_argument("--allow-label-writes", action="store_true")
    submit.add_argument("--run-annotation", action="store_true")
    submit.add_argument("--device", default="auto")
    submit.add_argument("--workers", type=int, default=16)

    run = sub.add_parser("run", help="execute one queued workflow run")
    run.add_argument("--run-id", type=int, required=True)
    run.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL)
    run.add_argument("--python", type=Path, default=DEFAULT_PYTHON)

    worker = sub.add_parser("worker", help="poll and execute queued workflow runs")
    worker.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL)
    worker.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    worker.add_argument("--poll-seconds", type=float, default=10.0)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--auto-analyze", action="store_true",
                        help="run de-identified shadow agents after a QC_FAILED result")
    worker.add_argument("--auto-maintain", action="store_true",
                        help="run reviewer-approved maintenance actions in isolated snapshots")
    worker.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    worker.add_argument("--agents", type=Path, default=ROOT / "configs/agents.yaml")

    status = sub.add_parser("status")
    status.add_argument("--run-id", type=int)
    status.add_argument("--limit", type=int, default=20)

    report = sub.add_parser("report")
    report.add_argument("--run-id", type=int, required=True)

    retry = sub.add_parser("retry", help="requeue an ERROR/QC_FAILED/NEEDS_REVIEW run")
    retry.add_argument("--run-id", type=int, required=True)

    recover = sub.add_parser("recover-stale", help="requeue a confirmed stale RUNNING run")
    recover.add_argument("--run-id", type=int, required=True)
    recover.add_argument("--confirm-stale", action="store_true", required=True)

    abandon = sub.add_parser("abandon-stale", help="close a confirmed stale RUNNING run without requeueing it")
    abandon.add_argument("--run-id", type=int, required=True)
    abandon.add_argument("--confirm-stale", action="store_true", required=True)

    analyze = sub.add_parser("analyze", help="run de-identified read-only shadow agents")
    analyze.add_argument("--run-id", type=int, required=True)
    analyze.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    analyze.add_argument("--agents", type=Path, default=ROOT / "configs/agents.yaml")

    analyze_existing = sub.add_parser(
        "analyze-existing", help="snapshot existing issue manifests and immediately run shadow agents"
    )
    analyze_existing.add_argument("--dataset", type=Path, required=True)
    analyze_existing.add_argument("--source-run-id", type=int)
    analyze_existing.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    analyze_existing.add_argument("--agents", type=Path, default=ROOT / "configs/agents.yaml")

    maintenance = sub.add_parser("maintenance-run", help="run one approved action in an isolated snapshot")
    maintenance.add_argument("--run-id", type=int, required=True)
    maintenance.add_argument("--action-id", required=True)
    maintenance.add_argument("--skill-source", type=Path, default=DEFAULT_SKILL_SOURCE)
    maintenance.add_argument("--timeout-seconds", type=int, default=3600)
    maintenance.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    maintenance.add_argument("--agents", type=Path, default=ROOT / "configs/agents.aliyun-roles.yaml")

    gate = sub.add_parser("gate", help="compare baseline and candidate metric JSON")
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--thresholds", type=Path, default=ROOT / "configs/acceptance_thresholds.yaml")

    create_exp = sub.add_parser("experiment-create", help="create an isolated Git worktree experiment")
    create_exp.add_argument("--run-id", type=int, required=True)
    create_exp.add_argument("--hypothesis", type=Path, required=True)
    create_exp.add_argument("--repository", type=Path, required=True)
    create_exp.add_argument("--worktree-root", type=Path, default=ROOT / "runtime/worktrees")
    create_exp.add_argument("--base-ref", default="HEAD")

    eval_exp = sub.add_parser("experiment-evaluate", help="run the deterministic promotion gate")
    eval_exp.add_argument("--experiment-id", type=int, required=True)
    eval_exp.add_argument("--baseline", type=Path, required=True)
    eval_exp.add_argument("--candidate", type=Path, required=True)
    eval_exp.add_argument("--thresholds", type=Path, default=ROOT / "configs/acceptance_thresholds.yaml")
    return ap


def execute_maintenance_actions(db: Database, run_id: int, *,
                                skill_source: Path = DEFAULT_SKILL_SOURCE,
                                providers_path: Path = ROOT / "configs/providers.yaml",
                                agents_path: Path = ROOT / "configs/agents.aliyun-roles.yaml",
                                max_actions: int = 2) -> list[dict]:
    run = db.get_run(run_id)
    if run["status"] != RunState.EXPERIMENT_PROPOSED.value:
        return []
    details = db.run_details(run_id)
    completed = []
    eligible = {"isolated_code_experiment", "rule_draft", "skill_draft"}
    for row in details["maintenance_actions"]:
        action = json.loads(row["payload_json"])
        if (len(completed) >= max_actions
                or not agent_action_authorized(db, run_id, row, action)):
            continue
        if action["action_class"] not in eligible or not action.get("auto_execute", False):
            continue
        db.update_maintenance_action(run_id, action["action_id"], status="RUNNING")
        try:
            action["handoff_messages"] = db.agent_inbox(
                run_id, row["owner_role"], thread_id=f"run-{run_id}:{row['action_id']}"
            )
            result = run_llm_maintenance(
                action=action,
                output_root=ROOT / "runtime/maintenance" / f"run_{run_id:06d}",
                orchestrator_source=ROOT,
                skill_source=skill_source,
                providers_path=providers_path,
                agents_path=agents_path,
                provider_role=action.get("owner_role") or "qc_code_optimizer",
                db=db, run_id=run_id,
            )
            workspace = Path(result["workspace"])
            status = "SUCCEEDED" if result["result"].get("ready_for_validation") else "NEEDS_REVIEW"
            db.update_maintenance_action(run_id, action["action_id"], status=status,
                                         artifact_path=workspace)
            completed.append(result)
        except Exception as exc:
            db.update_maintenance_action(run_id, action["action_id"], status="FAILED")
            completed.append({"action_id": action["action_id"], "error": str(exc)})
    return completed


def main() -> None:
    args = parser().parse_args()
    db = database(args.db)
    if args.command == "init":
        dump({"database": str(db.path), "initialized": True})
    elif args.command == "submit":
        if args.mode == "apply" and not args.allow_label_writes:
            raise SystemExit("apply mode requires --allow-label-writes")
        if not args.dataset.exists():
            raise SystemExit(f"dataset path does not exist: {args.dataset}")
        orchestrator = WorkflowOrchestrator(db, ROOT, DEFAULT_SKILL, DEFAULT_PYTHON)
        run_id = orchestrator.submit(dataset_root=args.dataset, raw_input=args.raw_input,
                                     execute_mode=args.mode, device=args.device,
                                     workers=args.workers, run_annotation=args.run_annotation)
        dump({"run_id": run_id, "status": "QUEUED", "mode": args.mode})
    elif args.command == "run":
        orchestrator = WorkflowOrchestrator(db, ROOT, args.skill_root, args.python)
        dump(orchestrator.execute(args.run_id))
    elif args.command == "worker":
        orchestrator = WorkflowOrchestrator(db, ROOT, args.skill_root, args.python)
        while True:
            run_id = db.next_queued_run_id()
            if run_id is None:
                if args.once:
                    dump({"worker": "idle", "queued": 0})
                    break
                time.sleep(max(args.poll_seconds, 0.5))
                continue
            try:
                result = orchestrator.execute(run_id)
                dump(result)
                if args.auto_analyze and result["status"] == RunState.QC_FAILED.value:
                    analysis_result = asyncio.run(run_shadow_analysis(db, run_id, args.providers, args.agents))
                    dump(analysis_result)
                    if args.auto_maintain and analysis_result["final_state"] == RunState.EXPERIMENT_PROPOSED.value:
                        dump(execute_maintenance_actions(db, run_id))
            except Exception as exc:
                print(json.dumps({"run_id": run_id, "worker_error": str(exc)}, ensure_ascii=False),
                      file=sys.stderr, flush=True)
            if args.once:
                break
    elif args.command == "status":
        dump(db.get_run(args.run_id) if args.run_id else db.list_runs(args.limit))
    elif args.command == "report":
        dump(db.run_details(args.run_id))
    elif args.command == "retry":
        run = db.get_run(args.run_id)
        if run["status"] not in {"ERROR", "QC_FAILED", "NEEDS_REVIEW"}:
            raise SystemExit(f"run {args.run_id} in state {run['status']} cannot be retried")
        db.transition(args.run_id, RunState.QUEUED, current_step="manual_retry")
        dump({"run_id": args.run_id, "status": "QUEUED"})
    elif args.command == "recover-stale":
        run = db.get_run(args.run_id)
        if run["status"] != "RUNNING":
            raise SystemExit(f"run {args.run_id} is not RUNNING")
        failed_tasks = db.fail_open_tasks(args.run_id)
        db.transition(args.run_id, RunState.ERROR, current_step="stale_recovery",
                      error_message="manually confirmed stale worker")
        db.transition(args.run_id, RunState.QUEUED, current_step="stale_requeued")
        dump({"run_id": args.run_id, "status": "QUEUED", "recovered": True,
              "failed_open_tasks": failed_tasks})
    elif args.command == "abandon-stale":
        run = db.get_run(args.run_id)
        if run["status"] != "RUNNING":
            raise SystemExit(f"run {args.run_id} is not RUNNING")
        failed_tasks = db.fail_open_tasks(args.run_id)
        db.transition(args.run_id, RunState.SUPERSEDED, current_step="stale_abandoned",
                      error_message="manually confirmed stale worker; superseded by a newer run")
        dump({"run_id": args.run_id, "status": "SUPERSEDED", "abandoned": True,
              "failed_open_tasks": failed_tasks})
    elif args.command == "analyze":
        dump(asyncio.run(run_shadow_analysis(db, args.run_id, args.providers, args.agents)))
    elif args.command == "analyze-existing":
        snapshot = snapshot_existing_qc(db, ROOT, args.dataset, args.source_run_id)
        result = asyncio.run(run_shadow_analysis(
            db, snapshot["analysis_run_id"], args.providers, args.agents
        ))
        dump({"snapshot": snapshot, "analysis": result})
    elif args.command == "maintenance-run":
        action_row = db.get_maintenance_action(args.run_id, args.action_id)
        run = db.get_run(args.run_id)
        if run["status"] != RunState.EXPERIMENT_PROPOSED.value:
            raise SystemExit("maintenance-run requires an EXPERIMENT_PROPOSED run")
        action = action_row["payload"]
        if not agent_action_authorized(db, args.run_id, action_row, action):
            raise SystemExit(
                f"action {args.action_id} lacks supervisor assignment and independent approval"
            )
        if action["action_class"] not in {"isolated_code_experiment", "rule_draft", "skill_draft"}:
            raise SystemExit("only isolated code/rule/skill maintenance actions are executable")
        db.update_maintenance_action(args.run_id, args.action_id, status="RUNNING")
        try:
            action["handoff_messages"] = db.agent_inbox(
                args.run_id, action_row["owner_role"],
                thread_id=f"run-{args.run_id}:{args.action_id}",
            )
            result = run_llm_maintenance(
                action=action,
                output_root=ROOT / "runtime/maintenance" / f"run_{args.run_id:06d}",
                orchestrator_source=ROOT,
                skill_source=args.skill_source,
                providers_path=args.providers,
                agents_path=args.agents,
                provider_role=action.get("owner_role") or "qc_code_optimizer",
                timeout_seconds=args.timeout_seconds,
                db=db, run_id=args.run_id,
            )
            status = "SUCCEEDED" if result["result"].get("ready_for_validation") else "NEEDS_REVIEW"
            db.update_maintenance_action(args.run_id, args.action_id, status=status,
                                         artifact_path=Path(result["workspace"]))
            dump(result)
        except Exception:
            db.update_maintenance_action(args.run_id, args.action_id, status="FAILED")
            raise
    elif args.command == "gate":
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
        dump(model_dump(evaluate_metrics(baseline, candidate, thresholds)))
    elif args.command == "experiment-create":
        run = db.get_run(args.run_id)
        if run["status"] != RunState.EXPERIMENT_PROPOSED.value:
            raise SystemExit("experiment-create requires an EXPERIMENT_PROPOSED run")
        hypothesis = json.loads(args.hypothesis.read_text(encoding="utf-8"))
        experiment_id = db.create_experiment(args.run_id, hypothesis)
        manager = WorktreeManager(args.repository, args.worktree_root)
        try:
            worktree = manager.create(experiment_id, args.base_ref)
            request = {
                "experiment_id": experiment_id,
                "source_run_id": args.run_id,
                "hypothesis": hypothesis,
                "constraints": [
                    "Do not access or modify production labels.",
                    "Add tests for every algorithm change.",
                    "Write predictions and metrics only under the experiment artifact directory.",
                ],
            }
            (worktree / "CMR_LGE_EXPERIMENT_REQUEST.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            db.update_experiment(experiment_id, status="WORKTREE_READY", worktree_path=worktree)
        except Exception:
            db.update_experiment(experiment_id, status="ERROR")
            raise
        dump({"experiment_id": experiment_id, "worktree": str(worktree), "status": "WORKTREE_READY"})
    elif args.command == "experiment-evaluate":
        experiment = db.get_experiment(args.experiment_id)
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
        run = db.get_run(experiment["source_run_id"])
        if run["status"] != RunState.EXPERIMENT_PROPOSED.value:
            raise SystemExit(f"source run is {run['status']}, expected EXPERIMENT_PROPOSED")
        db.transition(run["run_id"], RunState.VALIDATING, current_step="promotion_gate")
        decision = evaluate_metrics(baseline, candidate, thresholds)
        decision_data = model_dump(decision)
        final_state = {
            "promote": RunState.PROMOTED,
            "reject": RunState.REJECTED,
            "needs_human_review": RunState.NEEDS_REVIEW,
        }[decision.decision]
        db.update_experiment(args.experiment_id, status=final_state.value,
                             baseline_metrics=baseline, candidate_metrics=candidate,
                             decision=decision_data)
        db.transition(run["run_id"], final_state, current_step="promotion_gate_complete")
        dump({"experiment_id": args.experiment_id, **decision_data})


if __name__ == "__main__":
    main()
