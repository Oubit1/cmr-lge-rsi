"""Read-only shadow spatial-boundary discriminator (S763).

Scope (shadow_triage_only, zero production writes):
  * derive a routing discriminator from the supplied anonymous
    spatial_control records (checked-control + pending-candidate rank/core);
  * suppress checked records that were mis-routed to
    spatial_boundary_candidate (checked_false_routes -> 0);
  * preserve pending candidates whose (reason, rank) is not observed in the
    checked-control set (pending_spatial_boundary_candidates > 0);
  * emit module hash, audit diff, rollback command and zero-write counters.

Reuses the existing zero-write shadow-triage conventions. Read-only: writes no
production labels, code, datasets or runtime databases. Concordance, precision
and CI against independent reference labels are reported unavailable, never
imputed. Missing spatial_control fields are reported unavailable, not imputed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, List, Optional, Set, Tuple

MODULE_VERSION = "shadow_spatial_boundary/1.0"

ROUTING_MODE = "shadow_triage_only"

CANDIDATE_DECISION = "spatial_boundary_candidate"
UNRESOLVED_DECISION = "unresolved"

OUTSIDE_REASON = "outside_unique_sax_core"

NO_WRITE_CLAUSE = (
    "Read-only shadow spatial-boundary discriminator. No production labels, "
    "code, datasets or runtime databases are written or mutated."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_spatial_boundary.py "
    "tests/test_shadow_spatial_boundary.py"
)


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(module_hash))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _core_key(core: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(core, (list, tuple)) or len(core) != 2:
        return None
    try:
        return (int(core[0]), int(core[1]))
    except (TypeError, ValueError):
        return None


def _checked_rank_core_set(records: List[Dict[str, Any]]) -> Set[Tuple[Optional[Tuple[int, int]], int]]:
    """Ranks observed on checked records, keyed by (core, rank).

    Records missing rank or core are reported as unavailable and excluded from
    the observed set rather than imputed.
    """
    observed: Set[Tuple[Optional[Tuple[int, int]], int]] = set()
    for rec in records:
        if rec.get("state") != "checked":
            continue
        rank = rec.get("rank")
        if rank is None:
            continue
        try:
            rank_i = int(rank)
        except (TypeError, ValueError):
            continue
        observed.add((_core_key(rec.get("core")), rank_i))
    return observed


def _discriminate(
    rec: Dict[str, Any],
    observed: Set[Tuple[Optional[Tuple[int, int]], int]],
) -> Dict[str, Any]:
    """Apply the derived discriminator to a single spatial_control record."""
    state = rec.get("state")
    reason = rec.get("reason")
    rank = rec.get("rank")
    core = _core_key(rec.get("core"))

    missing: List[str] = []
    if state is None:
        missing.append("state")
    if reason is None:
        missing.append("reason")
    if rank is None:
        missing.append("rank")

    if missing:
        return {
            "case_id": rec.get("case_id"),
            "state": state,
            "reason": reason,
            "rank": rank,
            "core": list(core) if core else None,
            "decision": UNRESOLVED_DECISION,
            "suppressed": False,
            "status": "unavailable_missing_fields",
            "missing_fields": missing,
        }

    try:
        rank_i = int(rank)
    except (TypeError, ValueError):
        return {
            "case_id": rec.get("case_id"),
            "state": state,
            "reason": reason,
            "rank": rank,
            "core": list(core) if core else None,
            "decision": UNRESOLVED_DECISION,
            "suppressed": False,
            "status": "unavailable_non_numeric_rank",
            "missing_fields": ["rank"],
        }

    is_outside = reason == OUTSIDE_REASON
    rank_observed = (core, rank_i) in observed

    if state == "checked" and is_outside:
        # Previously mis-routed checked record: suppress to unresolved.
        return {
            "case_id": rec.get("case_id"),
            "state": state,
            "reason": reason,
            "rank": rank_i,
            "core": list(core) if core else None,
            "decision": UNRESOLVED_DECISION,
            "suppressed": True,
            "status": "ok",
            "missing_fields": [],
        }

    if state == "待筛" and is_outside and not rank_observed:
        return {
            "case_id": rec.get("case_id"),
            "state": state,
            "reason": reason,
            "rank": rank_i,
            "core": list(core) if core else None,
            "decision": CANDIDATE_DECISION,
            "suppressed": False,
            "status": "ok",
            "missing_fields": [],
        }

    return {
        "case_id": rec.get("case_id"),
        "state": state,
        "reason": reason,
        "rank": rank_i,
        "core": list(core) if core else None,
        "decision": UNRESOLVED_DECISION,
        "suppressed": False,
        "status": "ok",
        "missing_fields": [],
    }


def audit_diff(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    if previous is None:
        return {
            "baseline": "none",
            "changed_cases": [],
            "added_cases": sorted(current.get("cases", {}).keys()),
            "removed_cases": [],
            "production_writes": 0,
        }
    prev_cases = previous.get("cases", {})
    cur_cases = current.get("cases", {})
    changed = [c for c in cur_cases if c in prev_cases and prev_cases[c] != cur_cases[c]]
    added = [c for c in cur_cases if c not in prev_cases]
    removed = [c for c in prev_cases if c not in cur_cases]
    return {
        "baseline": previous.get("module_hash"),
        "changed_cases": changed,
        "added_cases": added,
        "removed_cases": removed,
        "production_writes": 0,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only shadow run over the supplied spatial_control records."""
    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    if not isinstance(spatial, dict):
        return {
            "module_version": MODULE_VERSION,
            "module_hash": module_hash(),
            "mode": "read_only_shadow_spatial_boundary",
            "routing_mode": ROUTING_MODE,
            "status": "unavailable_no_spatial_control",
            "checked_false_routes": None,
            "pending_spatial_boundary_candidates": None,
            "no_write_clause": NO_WRITE_CLAUSE,
            "rollback_command": ROLLBACK_COMMAND,
            "writes": {"labels": 0, "code": 0, "datasets": 0},
        }

    records = spatial.get("records", []) or []
    observed = _checked_rank_core_set(records)

    cases: Dict[str, Any] = {}
    checked_false_routes = 0
    pending_candidates = 0
    suppressed_checked: List[str] = []
    preserved_pending: List[str] = []
    unavailable_cases: List[str] = []

    for rec in records:
        cid = rec.get("case_id")
        out = _discriminate(rec, observed)
        cases[cid] = out
        if out["status"] != "ok":
            unavailable_cases.append(cid)
            continue
        if out["state"] == "checked" and out["decision"] == CANDIDATE_DECISION:
            checked_false_routes += 1
        if out["suppressed"]:
            suppressed_checked.append(cid)
        if out["state"] == "待筛" and out["decision"] == CANDIDATE_DECISION:
            pending_candidates += 1
            preserved_pending.append(cid)

    n = len(records)
    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_spatial_boundary",
        "routing_mode": ROUTING_MODE,
        "rule": {
            "candidate_decision": CANDIDATE_DECISION,
            "outside_reason": OUTSIDE_REASON,
            "suppress_checked_outside": True,
            "preserve_pending_unobserved_rank": True,
        },
        "n_records": n,
        "checked_false_routes": checked_false_routes,
        "pending_spatial_boundary_candidates": pending_candidates,
        "suppressed_checked_cases": suppressed_checked,
        "preserved_pending_cases": preserved_pending,
        "unavailable_cases": unavailable_cases,
        "coverage": {
            "checked_false_routes": {"count": checked_false_routes, "denominator": n},
            "pending_spatial_boundary_candidates": {
                "count": pending_candidates,
                "denominator": n,
            },
            "suppressed_checked": {
                "count": len(suppressed_checked),
                "denominator": n,
            },
            "unavailable": {"count": len(unavailable_cases), "denominator": n},
        },
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "spatial_control_fields": "missing fields reported per-case, not imputed",
        },
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
        "cases": cases,
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
