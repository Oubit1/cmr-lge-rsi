#!/usr/bin/env python3
"""Unified scorecard: every test/scoring artifact in the project, one JSON.

Answers the four operator questions:
  1. 工作流是否有效        -> gate pass/reject + reasons
  2. 自动 vs 手工修复对比   -> edits by origin + prompt jobs by decision + batch stats
  3. 泛化能力              -> rules_v2 frozen-set metrics (gold/quasi/external)
  4. 处理结果评估          -> batch reports per pool + quality scoring summary
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_score_card() -> dict:
    card = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    ev = ROOT / "runtime/eval_set/v1"
    card["eval_set"] = _load(ev / "summary.json")
    card["rules_v2"] = _load(ev / "rules_v2_report.json")

    batch = {}
    for f in sorted((ROOT / "runtime/batch_candidates").glob("*_report*.json")):
        d = _load(f)
        if not d:
            continue
        key = d.get("pool") or f.stem
        running = f.name.endswith(".partial.json")
        if key in batch and not batch[key].get("running") and running:
            continue
        d = dict(d)
        d["running"] = running
        d.pop("records", None)
        batch[key] = d
    card["batch_screening"] = batch

    gate = _load(ROOT / "runtime/batch_candidates/candidate_validation.json")
    if gate:
        reasons = Counter()
        for r in gate.get("records", []):
            if not r.get("passed"):
                for reason in r.get("reasons", []):
                    reasons[reason] += 1
        card["gate"] = {"total": gate.get("total"), "passed": gate.get("passed"),
                        "rejected": gate.get("rejected"),
                        "pass_rate": round(gate.get("passed", 0) / max(1, gate.get("total", 1)), 4),
                        "reject_reasons": dict(reasons.most_common(10))}

    edits = {"by_origin": Counter(), "by_status": Counter(), "total": 0}
    edir = ROOT / "runtime/manual_annotations/edits"
    if edir.is_dir():
        for d in edir.iterdir():
            for f in d.glob("*.json"):
                j = _load(f)
                if isinstance(j, dict) and ("edit_origin" in j or "status" in j):
                    edits["total"] += 1
                    edits["by_origin"][j.get("edit_origin", "?")] += 1
                    edits["by_status"][j.get("status", "?")] += 1
                    break
    edits["by_origin"] = dict(edits["by_origin"])
    edits["by_status"] = dict(edits["by_status"])
    card["edits"] = edits

    jobs = Counter()
    pj = ROOT / "runtime/manual_annotations/prompt_jobs"
    if pj.is_dir():
        for d in pj.iterdir():
            plan = _load(d / "agent_plan.json")
            if plan:
                jobs[(plan.get("plan") or {}).get("decision", "?")] += 1
            else:
                jobs["pending_or_failed"] += 1
    card["prompt_agent_jobs"] = dict(jobs)

    tri = (_load(ROOT / "runtime/boundary_triage/cmr_boundary_manifest_report.json")
           or _load(ROOT / "runtime/boundary_triage/report.partial.json"))
    if tri:
        card["boundary_triage"] = tri

    qdir = ROOT / "artifacts/image_label_quality_v1"
    if qdir.is_dir():
        summary = None
        for name in ("summary.json", "report.json", "index.json"):
            summary = _load(qdir / name)
            if summary:
                break
        card["quality_scoring"] = summary or {"dir": str(qdir),
                                              "files": len(list(qdir.rglob('*.json')))}
    return card


if __name__ == "__main__":
    print(json.dumps(build_score_card(), ensure_ascii=False, indent=2))
