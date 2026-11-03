#!/usr/bin/env python3
"""Record overnight workflow evidence and keep a compact human-readable status."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from board_server import DATASET, ROOT, collect


def refresh_candidate_gate() -> None:
    root = ROOT / "runtime/batch_candidates"
    reports = list(root.glob("*_report.json"))
    output = root / "candidate_validation.json"
    newest = max((path.stat().st_mtime for path in reports), default=0)
    if output.exists() and output.stat().st_mtime >= newest:
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_batch_candidates.py"),
         "--dataset", str(DATASET)], cwd=ROOT, timeout=300, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def render(rows: list[dict], output: Path) -> None:
    first, last = rows[0], rows[-1]
    pools0 = first["pools"]
    pools1 = last["pools"]
    lines = ["# 夜间运行状态", "", f"- 开始：{first['timestamp']}",
             f"- 最近：{last['timestamp']}",
             f"- 真实 HTTP 调用：{last['calls']}（新增 {last['calls'] - first['calls']}）",
             f"- Token：{last['tokens']}（新增 {last['tokens'] - first['tokens']}）",
             f"- 调用错误：{last['call_errors']}",
             f"- 固定评估 gate：{'PASS' if last['eval_pass'] else 'FAIL/MISSING'}", "",
             "## Issue 趋势", "", "| 池 | 起始 | 当前 | 变化 |", "|---|---:|---:|---:|"]
    for name in sorted(set(pools0) | set(pools1)):
        before, after = pools0.get(name, 0), pools1.get(name, 0)
        lines.append(f"| {name} | {before} | {after} | {after - before:+d} |")
    issue_delta = sum(pools1.values()) - sum(pools0.values())
    token_delta = last["tokens"] - first["tokens"]
    staged = last.get("batch_summary") or {}
    efficiency = (f"{token_delta / abs(issue_delta):.0f} token/净减少 issue"
                  if issue_delta < 0 else f"NO_GAIN（{token_delta} token，issue 净变化 {issue_delta:+d}）")
    lines.extend(["", "## 业务产出", "",
                  f"- 自动效率：**{efficiency}**",
                  f"- 隔离修复候选：{staged.get('gate_passed', 0)} / "
                  f"{staged.get('gate_total', staged.get('auto_fixed', 0))} 通过；"
                  f"拒绝 {staged.get('gate_rejected', 0)}（未写正式标签）",
                  f"- 自动循环：{'无增益熔断' if last.get('state', {}).get('paused') else '运行中'}",
                  "", "## 进程", "",
                  f"- autopilot: {last['controllers'].get('autopilot', '-')}",
                  f"- board: {last['controllers'].get('board', '-')}",
                  f"- edit-agent: {last['controllers'].get('manual_edit_agent', '-')}", "",
                  "> 候选只在隔离区；正式标签写入仍受固定评估、复核和应用 gate 约束。"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot() -> dict:
    refresh_candidate_gate()
    data = collect()
    rules = data.get("evaluation", {}).get("rules", {})
    rates = rules.get("detection_rate", {})
    eval_pass = (float(rules.get("clean_false_positive_rate", 1)) <= .10
                 and float(rules.get("true_small_false_repair", 1)) <= .05
                 and len(rates) == 4 and all(float(x) >= .60 for x in rates.values()))
    controllers = dict(data["controllers"])
    controllers["board"] = controllers.get("board", "-")
    return {"timestamp": datetime.now().astimezone().isoformat(),
            "pools": {row["key"]: row["count"] for row in data["pools"]},
            "calls": data["calls"]["calls"], "tokens": data["calls"]["tokens"],
            "call_errors": data["calls"]["errors"], "eval_pass": eval_pass,
            "evaluation": rules, "human_required": data["human_required_count"],
            "controllers": controllers, "state": data["state"], "batch": data["batch"],
            "batch_summary": data.get("batch_summary", {})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts" / "overnight_latest")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "trend.jsonl"
    deadline = datetime.now() + timedelta(hours=args.hours) if args.hours > 0 else None
    while deadline is None or datetime.now() < deadline:
        row = snapshot()
        append_jsonl(log, row)
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
        render(rows, args.output_dir / "NIGHTLY_STATUS.md")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
