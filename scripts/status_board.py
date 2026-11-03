#!/usr/bin/env python3
"""Human-readable board: who is running, which agent is working, how far the issues got."""
from __future__ import annotations

import json
import subprocess
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "runtime/workflow.sqlite"
STATE = ROOT / "runtime/autopilot_state.json"
import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
CHECK = (_public_paths.DATASET_ROOT / "labels/check")

POOLS = {
    "cmr_boundary_manifest": "manifest_backup/cmr_boundary_manifest.csv",
    "cmr_middle_manifest": "manifest_backup/cmr_middle_manifest.csv",
    "lge_boundary_manifest": "manifest_backup/lge_boundary_manifest.csv",
    "lge_issue_manifest": "manifest_backup/lge_issue_manifest.csv",
    "lge_middle_manifest": "manifest_backup/lge_middle_manifest.csv",
    "trajectory_manifest": "manifest_backup/trajectory_manifest.csv",
}
STAGE_ORDER = ["PROPOSED", "APPROVED", "RUNNING", "NEEDS_REVIEW", "SUCCEEDED", "VALIDATED", "PROMOTED", "HUMAN_REQUIRED", "FAILED", "REJECTED"]


def local_time(value, with_date: bool = False) -> str:
    """The DB stores UTC; the operator reads Beijing time."""
    if not value:
        return "-"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return text[11:19] if len(text) > 19 else text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone()
    return dt.strftime("%m-%d %H:%M:%S") if with_date else dt.strftime("%H:%M:%S")


def pid_of(pattern: str) -> str:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else "-"
    except Exception:
        return "-"


def pool_for(action_id: str, objective: str) -> str:
    """Map an action to one issue pool, keyed on the action id first."""
    key = action_id.lower()
    if "lge" in key:
        if any(t in key for t in ("mi", "infarct", "scar", "frag")):
            return "lge_issue_manifest"
        if "paired" in key or "myo" in key:
            return "lge_boundary_manifest"
        return "lge_issue_manifest"
    if "traj" in key or "centroid" in key:
        return "trajectory_manifest"
    if "cmr" in key:
        if any(t in key for t in ("small", "area", "middle", "excess")):
            return "cmr_middle_manifest"
        return "cmr_boundary_manifest"
    if "evidence" in key or "dedup" in key:
        return "cmr_boundary_manifest"
    text = objective.lower()
    if "lge" in text:
        return "lge_issue_manifest"
    if "cmr" in text:
        return "cmr_middle_manifest"
    return ""


def main() -> None:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)

    print("=" * 78)
    print("控制器")
    print("=" * 78)
    print(f"  autopilot pid={pid_of('scripts/autopilot.py')}   worker pid={pid_of('cli.py worker')}")
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        events = state.get("events", [])
        last = events[-1] if events else {}
        print(f"  最近事件: {last.get('event')}  {local_time(last.get('timestamp'))}  "
              f"cycles={state.get('cycles')} stagnant={state.get('stagnant_cycles')}")
        running = [e for e in events if e.get("event") == "AGENT_ANALYSIS_STARTED"][-1:]
        if running:
            print(f"  正在分析: run {running[0].get('run_id')}")

    print()
    print("=" * 78)
    print("运行")
    print("=" * 78)
    for row in db.execute("select run_id,status,execute_mode,current_step,created_at from workflow_runs order by run_id desc limit 6"):
        print(f"  run {row['run_id']:<3} {row['status']:<22} {row['current_step']:<28} {local_time(row['created_at'], True)}")

    print()
    print("=" * 78)
    print("Agent 活动（最近 12 次模型调用）")
    print("=" * 78)
    print(f"  {'时间':<9} {'run':<4} {'角色':<22} {'模型':<22} {'状态'}")
    for row in db.execute("select run_id,role,model,status,created_at from agent_outputs order by output_id desc limit 12"):
        ts = local_time(row["created_at"])
        print(f"  {ts:<9} {row['run_id']:<4} {row['role']:<22} {row['model']:<22} {row['status']}")

    print()
    print("=" * 78)
    print("维护动作 → 覆盖的 issue 池 → 阶段")
    print("=" * 78)
    covered: dict[str, str] = {}
    for row in db.execute("select run_id,action_id,owner_role,action_class,status,payload_json from maintenance_actions order by run_id desc, action_id"):
        payload = json.loads(row["payload_json"])
        pool = pool_for(row["action_id"], payload.get("objective", ""))
        print(f"  run{row['run_id']} {row['action_id'][:40]:<40} {row['status']:<13} → {pool or '(未映射)'}")
        if pool:
            best = covered.get(pool)
            if best is None or STAGE_ORDER.index(row["status"] if row["status"] in STAGE_ORDER else "PROPOSED") > STAGE_ORDER.index(best):
                covered[pool] = row["status"]

    print()
    print("=" * 78)
    print("Issue 池")
    print("=" * 78)
    print(f"  {'池':<26} {'文件数':>7}   {'覆盖动作阶段':<14} 距离能改标签")
    for key, relative in POOLS.items():
        path = CHECK / relative
        count = max(sum(1 for _ in path.open(encoding="utf-8-sig")) - 1, 0) if path.is_file() else 0
        stage = covered.get(key, "无动作")
        gaps = {
            "无动作": "还没有提案",
            "PROPOSED": "已提案，等待独立评审",
            "APPROVED": "智能体交接与独立评审已通过",
            "RUNNING": "正在隔离实现/验证",
            "NEEDS_REVIEW": "待 gate 与晋级",
            "SUCCEEDED": "待 gate 与晋级",
            "PROMOTED": "评审通过，待应用到生产并重跑 QC",
            "HUMAN_REQUIRED": "重试耗尽，需人工判断",
            "FAILED": "失败，待重试",
            "REJECTED": "已否决",
        }
        gap = gaps.get(stage, "未知")
        print(f"  {key:<26} {count:>7}   {stage:<14} {gap}")

    print()
    print("  说明：任务数字来自 labels/check/manifest_backup 的扁平 manifest；")
    print("        “覆盖动作阶段”取该池所有动作里最靠后的一步（提案→实现→验证→晋级）。")


if __name__ == "__main__":
    main()
