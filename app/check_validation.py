"""Operator adjudications for a frozen, stratified check holdout.

This module records route truth only. It never changes a PNG or a label.
Predictions are frozen before the first adjudication so later rule changes
cannot silently improve a reported false-route rate.
"""
from __future__ import annotations

import hashlib
import json
import os
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.check_inventory import build_check_inventory


ROUTES = {"checked", "repair", "trash"}
QUOTAS = {"adequate_area_low_cavity": 13,
          "low_area_low_cavity": 4,
          "low_area_adequate_cavity": 3}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wilson_upper(errors: int, total: int) -> float | None:
    if not total:
        return None
    z = 1.96
    p = errors / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** .5
    return round((center + radius) / denominator, 4)


def propose_plan(dataset_root: Path, runtime_root: Path) -> dict[str, Any]:
    holdout = _read(runtime_root / "check_iterations/holdout_v1.json")
    rows = holdout.get("holdout")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen check holdout is missing")
    canonical = sorted(rows, key=lambda row: row["key"])
    if holdout.get("holdout_sha256") != _digest(canonical):
        raise ValueError("frozen holdout hash mismatch")
    inventory = build_check_inventory(dataset_root, runtime_root)
    if not inventory["shadow_current"]:
        raise ValueError("current shadow evidence is stale")
    by_key = {(row["modality"], row["key"]): row for row in inventory["rows"]}
    manifest_path = dataset_root / "labels/check/manifest_backup/cmr_boundary_manifest.csv"
    editor_indices: dict[str, int] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                editor_indices.setdefault(f'{row["patient"]}/{row["volume"]}', index)
    selected = []
    for stratum, quota in QUOTAS.items():
        candidates = [row for row in rows if row["stratum"] == stratum
                      and by_key.get(("cmr", row["key"]), {}).get("state") == "pending"]
        candidates.sort(key=lambda row: hashlib.sha256(
            ("cmr-check-validation-v1:" + row["key"]).encode()).hexdigest())
        if len(candidates) < quota:
            raise ValueError(f"insufficient pending frozen cases in {stratum}")
        for row in candidates[:quota]:
            current = by_key[("cmr", row["key"])]
            suggestion = ("trash" if current["shadow_boundary"] is True else
                          "repair" if current["candidate_frames"] else None)
            selected.append({"key": row["key"], "stratum": stratum,
                             "suggestion": suggestion,
                             "editor_index": editor_indices.get(row["key"])})
    plan = {"schema_version": "cmr-check-validation-v1", "holdout_sha256": holdout["holdout_sha256"],
            "shadow_created_at": inventory["shadow_created_at"],
            "shadow_control_gate": inventory["shadow_control_gate"],
            "items": selected}
    plan["plan_sha256"] = _digest(plan)
    return plan


def freeze_plan(dataset_root: Path, runtime_root: Path) -> dict[str, Any]:
    """Persist the selected cohort and baseline prediction before automatic changes."""
    path = runtime_root / "check_validation/plan.json"
    existing = _read(path)
    if existing:
        return existing
    plan = propose_plan(dataset_root, runtime_root)
    _atomic(path, plan)
    return plan


def review_state(dataset_root: Path, runtime_root: Path) -> dict[str, Any]:
    base = runtime_root / "check_validation"
    plan = _read(base / "plan.json")
    if not plan:
        plan = propose_plan(dataset_root, runtime_root)
    reviews: dict[str, dict[str, Any]] = {}
    audit = base / "decisions.jsonl"
    if audit.is_file():
        for line in audit.read_text(encoding="utf-8").splitlines():
            decision = json.loads(line)
            if decision.get("plan_sha256") == plan["plan_sha256"]:
                reviews[decision["key"]] = decision
    items = [{**item, "decision": reviews.get(item["key"], {}).get("route"),
              "candidate_accepted": reviews.get(item["key"], {}).get("candidate_accepted"),
              "reviewed_at": reviews.get(item["key"], {}).get("timestamp")}
             for item in plan["items"]]
    explicit = [item for item in items if item["decision"] and item["suggestion"]]
    false_routes = sum(item["decision"] != item["suggestion"] or
                       (item["suggestion"] == "repair" and item["candidate_accepted"] is False)
                       for item in explicit)
    strata = {name: {"selected": sum(item["stratum"] == name for item in items),
                     "reviewed": sum(item["stratum"] == name and item["decision"] is not None
                                     for item in items)} for name in QUOTAS}
    report = {"schema_version": "cmr-check-validation-report-v1",
              "plan_sha256": plan["plan_sha256"], "holdout_sha256": plan["holdout_sha256"],
              "shadow_control_gate": plan["shadow_control_gate"],
              "selected": len(items), "reviewed": sum(item["decision"] is not None for item in items),
              "suggested": len(explicit),
              "total_suggestions": sum(item["suggestion"] is not None for item in items),
              "false_routes": false_routes,
              "false_route_rate": round(false_routes / len(explicit), 4) if explicit else None,
              "false_route_wilson_upper_95": _wilson_upper(false_routes, len(explicit)),
              "suggestion_coverage": round(sum(item["suggestion"] is not None for item in items)
                                           / len(items), 4) if items else 0,
              "strata": strata,
              "ready_for_independent_review": all(item["decision"] is not None for item in items)}
    return {"plan": plan, "items": items, "report": report}


def submit_decision(dataset_root: Path, runtime_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    key = str(payload.get("key") or "")
    route = str(payload.get("route") or "")
    reviewer = str(payload.get("reviewer") or "").strip()
    if route not in ROUTES or not reviewer or len(reviewer) > 80:
        raise ValueError("route and reviewer are required")
    if route == "repair" and payload.get("candidate_accepted") not in (True, False):
        raise ValueError("repair requires candidate_accepted true or false")
    base = runtime_root / "check_validation"
    plan_path = base / "plan.json"
    plan = _read(plan_path)
    if not plan:
        plan = propose_plan(dataset_root, runtime_root)
    if payload.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("validation plan changed; reload")
    if key not in {item["key"] for item in plan["items"]}:
        raise ValueError("case is not in the frozen validation plan")
    inventory = build_check_inventory(dataset_root, runtime_root)
    current = next((row for row in inventory["rows"] if row["modality"] == "cmr"
                    and row["key"] == key), None)
    if current is None or current["state"] != "pending":
        raise ValueError("case left the pending pool; do not adjudicate stale state")
    if not plan_path.is_file():
        _atomic(plan_path, plan)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan["plan_sha256"],
              "key": key, "route": route, "reviewer": reviewer,
              "candidate_accepted": payload.get("candidate_accepted") if route == "repair" else None,
              "note": str(payload.get("note") or "")[:300]}
    _append(base / "decisions.jsonl", record)
    state = review_state(dataset_root, runtime_root)
    if state["report"]["ready_for_independent_review"]:
        _atomic(base / "report.json", state["report"])
    return state
