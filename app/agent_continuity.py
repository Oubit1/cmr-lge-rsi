"""Model-neutral role telemetry and durable continuity capsules.

The capsule is assembled only from persisted orchestration records.  It does
not contain raw images, masks, patient identifiers, or a model's hidden state,
so changing provider/model keeps the same role history and responsibility
contract.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from app.database import Database
from app.role_routing import ROLE_RESPONSIBILITIES


SCHEMA_VERSION = "agent-continuity-v1"


def _table_exists(connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _score_trend(values: list[float]) -> dict[str, Any]:
    recent = values[-3:]
    previous = values[-6:-3]
    recent_mean = mean(recent) if recent else None
    previous_mean = mean(previous) if previous else None
    return {
        "samples": len(values),
        "latest": _rounded(values[-1] if values else None),
        "recent_mean": _rounded(recent_mean),
        "delta": _rounded(
            recent_mean - previous_mean
            if recent_mean is not None and previous_mean is not None else None
        ),
    }


def _recent_progress(iteration_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in iteration_history:
        for action in run.get("actions", []):
            rows.append({
                "run_id": run.get("run_id"),
                "action_id": action.get("action_id"),
                "owner_role": action.get("owner_role"),
                "action_class": action.get("action_class"),
                "workflow_stage": action.get("workflow_stage"),
                "status": action.get("status"),
                "objective": str(action.get("objective") or "")[:240],
            })
            if len(rows) >= 8:
                return rows
    return rows


def build_agent_continuity(
    db: Database, *, iteration_history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return a compact, persisted score/lineage capsule for every role."""
    roles = {
        role: {
            "responsibilities": list(responsibilities),
            "invocations": 0,
            "successful_invocations": 0,
            "failed_invocations": 0,
            "success_rate": None,
            "total_tokens": 0,
            "average_latency_ms": None,
            "latest_run_id": None,
            "current_model": None,
            "model_history": [],
            "model_switches": 0,
            "score": _score_trend([]),
            "handoffs_sent": 0,
            "handoffs_received": 0,
            "owned_actions": {},
        }
        for role, responsibilities in ROLE_RESPONSIBILITIES.items()
    }
    model_rows: dict[str, list[str]] = defaultdict(list)
    latencies: dict[str, list[int]] = defaultdict(list)
    score_rows: dict[str, list[float]] = defaultdict(list)
    action_rows: dict[str, Counter[str]] = defaultdict(Counter)
    queried_progress: list[dict[str, Any]] = []

    with db.connect() as connection:
        if _table_exists(connection, "agent_invocations"):
            event_join = ""
            event_fields = "NULL AS actual_model,NULL AS call_status,NULL AS latency_ms,0 AS total_tokens"
            if _table_exists(connection, "agent_invocation_events"):
                event_join = (
                    " LEFT JOIN (SELECT invocation_id,MAX(event_id) AS event_id "
                    "FROM agent_invocation_events GROUP BY invocation_id) latest "
                    "ON latest.invocation_id=i.invocation_id "
                    "LEFT JOIN agent_invocation_events e ON e.event_id=latest.event_id"
                )
                event_fields = (
                    "e.actual_model,e.status AS call_status,e.latency_ms,e.total_tokens"
                )
            query = (
                "SELECT i.run_id,i.role,i.requested_model,i.status," + event_fields +
                " FROM agent_invocations i" + event_join + " ORDER BY i.started_at"
            )
            for row in connection.execute(query):
                role = str(row["role"])
                if role not in roles:
                    continue
                item = roles[role]
                item["invocations"] += 1
                ok = str(row["status"]) == "SUCCEEDED" and str(row["call_status"] or "SUCCEEDED") == "SUCCEEDED"
                item["successful_invocations"] += int(ok)
                item["failed_invocations"] += int(not ok)
                item["total_tokens"] += int(row["total_tokens"] or 0)
                item["latest_run_id"] = int(row["run_id"])
                model = str(row["actual_model"] or row["requested_model"] or "unknown")
                model_rows[role].append(model)
                if row["latency_ms"] is not None:
                    latencies[role].append(int(row["latency_ms"]))

        if _table_exists(connection, "agent_scorecards"):
            for row in connection.execute(
                "SELECT role,score_json FROM agent_scorecards ORDER BY scorecard_id"
            ):
                try:
                    score = json.loads(row["score_json"])
                    score_rows[str(row["role"])].append(float(score["composite_score"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue

        if _table_exists(connection, "agent_messages"):
            for row in connection.execute(
                "SELECT sender_role,recipient_role,COUNT(*) AS count FROM agent_messages "
                "GROUP BY sender_role,recipient_role"
            ):
                sender, recipient, count = str(row["sender_role"]), str(row["recipient_role"]), int(row["count"])
                if sender in roles:
                    roles[sender]["handoffs_sent"] += count
                if recipient in roles:
                    roles[recipient]["handoffs_received"] += count

        if _table_exists(connection, "maintenance_actions"):
            for row in connection.execute(
                "SELECT owner_role,status,COUNT(*) AS count FROM maintenance_actions "
                "GROUP BY owner_role,status"
            ):
                action_rows[str(row["owner_role"])][str(row["status"])] += int(row["count"])
            for row in connection.execute(
                "SELECT run_id,action_id,owner_role,action_class,status,payload_json "
                "FROM maintenance_actions ORDER BY run_id DESC,action_row_id DESC LIMIT 8"
            ):
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                queried_progress.append({
                    "run_id": int(row["run_id"]),
                    "action_id": row["action_id"],
                    "owner_role": row["owner_role"],
                    "action_class": row["action_class"],
                    "workflow_stage": payload.get("workflow_stage"),
                    "status": row["status"],
                    "objective": str(payload.get("objective") or "")[:240],
                })

    for role, item in roles.items():
        calls = int(item["invocations"])
        item["success_rate"] = _rounded(item["successful_invocations"] / calls) if calls else None
        item["average_latency_ms"] = (
            int(round(mean(latencies[role]))) if latencies[role] else None
        )
        history: list[str] = []
        for model in model_rows[role]:
            if not history or history[-1] != model:
                history.append(model)
        item["current_model"] = history[-1] if history else None
        item["model_history"] = history[-4:]
        item["model_switches"] = max(0, len(history) - 1)
        item["score"] = _score_trend(score_rows[role])
        item["owned_actions"] = dict(sorted(action_rows[role].items()))

    progress = _recent_progress(iteration_history or []) or queried_progress
    from app.rsi_runtime import inheritance
    evolution = inheritance(db)
    for role, item in roles.items():
        item["evolution_memory"] = (evolution if role in {
            "supervisor", "independent_reviewer", "failure_triage"
        } else inheritance(db, role=role))
    core = {
        "schema_version": SCHEMA_VERSION,
        "roles": roles,
        "recent_progress": progress,
        "evolution": evolution,
        "inheritance_contract": (
            "A replacement model inherits this persisted role capsule, the shared progress ledger, "
            "reviewer feedback, rules/casebook and Skill artifacts. Provider chat memory is not authority."
        ),
    }
    revision = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        **core,
        "revision": revision,
        "inheritance_ready": bool(roles and progress),
    }
