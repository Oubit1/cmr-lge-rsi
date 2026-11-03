"""Read-only, provenance-aware inventory of the four check folders.

Folder membership is an observed workflow state, never a clinical judgment.
Shadow-rule signals are displayed separately from audited automatic applications.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


FOLDERS = {"pending": "待筛", "work": "待处理", "checked": "checked", "trash": "垃圾桶"}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_check_inventory(dataset_root: Path, runtime_root: Path) -> dict[str, Any]:
    """Describe current files and prior recorded decisions without reclassifying them."""
    check = dataset_root / "labels/check"
    promotion = {row.get("key"): row for row in
                 _jsonl(runtime_root / "promotion/promotion_log.jsonl") if row.get("key")}
    auto_events = [row for row in _jsonl(runtime_root / "check_iterations/auto_apply.jsonl")
                   if row.get("event") == "applied" and row.get("key")]
    auto_applied = {row["key"]: row for row in auto_events}
    history = {row.get("relative_key"): row for row in
               _jsonl(check / "boundary_review_history.jsonl") if row.get("relative_key")}
    rows: list[dict[str, Any]] = []
    for state, folder in FOLDERS.items():
        for modality in ("cmr", "lge"):
            for png in sorted((check / folder / modality).glob("*/*.png")):
                key = f"{png.parent.name}/{png.stem}.nii.gz"
                promoted = promotion.get(key) if state == "checked" else None
                auto = auto_applied.get(key) if modality == "cmr" else None
                reviewed = history.get(key)
                source = ("自动筛选／评分晋升（有审计）" if auto and state == "checked" and auto.get("route") == "checked"
                          else "自动筛选剔除（有审计）" if auto and state == "trash" and auto.get("route") == "trash"
                          else "自动直通晋升" if promoted and promoted.get("action") == "direct_promoted"
                          else "自动修正后晋升" if promoted and promoted.get("action") == "fix_applied"
                          else "界面裁定记录" if reviewed and reviewed.get("status") == state
                          else "初始化分桶／来源未单独核验")
                rows.append({"key": key, "modality": modality, "state": state,
                             "source": source, "shadow_boundary": None,
                             "candidate_frames": 0, "quality_delta": None})
    counts = Counter(row["state"] for row in rows)
    current = {state: {row["key"] for row in rows if row["state"] == state and
                       row["modality"] == "cmr"} for state in FOLDERS}
    latest = _json(runtime_root / "check_iterations/latest.json")
    report = Path(str(latest.get("report_path") or "")).resolve()
    iteration_root = (runtime_root / "check_iterations").resolve()
    report_safe = iteration_root in report.parents and report.is_file()
    shadow = _json(report.parent / "spatial_boundary_shadow.json") if report_safe else {}
    evaluation = _json(report) if report_safe else {}
    shadow_rows = shadow.get("records") or []
    evaluated_rows = evaluation.get("records") or []
    shadow_matches = bool(shadow_rows) and {
        (str(item.get("state")), str(item.get("key"))) for item in shadow_rows
    } == {(FOLDERS[state], key) for state in ("pending", "checked") for key in current[state]}
    evaluation_matches = bool(evaluated_rows) and {
        str(item.get("key")) for item in evaluated_rows
    } == current["pending"]
    snapshot_current = shadow_matches and evaluation_matches and all(
        int(latest.get(field, -1)) == counts[field]
        for field in FOLDERS
    )
    if snapshot_current:
        shadow_by_key = {(item["state"], item["key"]): item for item in shadow_rows}
        evaluation_by_key = {item["key"]: item for item in evaluated_rows}
        for row in rows:
            if row["modality"] != "cmr" or row["state"] != "pending":
                continue
            shadow_item = shadow_by_key.get(("待筛", row["key"])) or {}
            row["shadow_boundary"] = shadow_item.get("decision") == "spatial_boundary_candidate"
            evaluated = evaluation_by_key.get(row["key"]) or {}
            row["candidate_frames"] = len(evaluated.get("frames") or [])
            row["quality_delta"] = evaluated.get("quality_delta")
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "counts": {state: counts[state] for state in FOLDERS},
        "total": len(rows),
        "rows": rows,
        "auto_apply": {"applied": len(auto_events),
                       "last_at": auto_events[-1].get("timestamp") if auto_events else None},
        "shadow_current": snapshot_current,
        "shadow_control_gate": shadow.get("control_gate") if snapshot_current else "STALE",
        "shadow_created_at": latest.get("created_at") if snapshot_current else None,
        "note": ("空间边界控制闸未通过，该分支不自动剔除；合格的修正候选仍可按筛选／评分自动晋升"
                 if snapshot_current and shadow.get("control_gate") != "PASS"
                 else "影子评估与当前目录不一致，隐藏候选信号" if not snapshot_current
                 else "影子评估已同步；候选仍非人工或独立复核结论"),
    }
