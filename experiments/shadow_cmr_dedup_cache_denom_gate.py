"""Read-only shadow dedup cache + denominator reconciliation gate.

Scope (shadow_triage_only, zero production writes):
  * reuse a single connected-component scan per (case_id, issue_type) key so
    repeated issue types on the same volume hit the cache instead of
    re-scanning (no second harness);
  * reconcile the 13/39/61 denominators (non_holdout_in_queue /
    spatial pending_candidates / full_pending_queue) BEFORE executing and
    block execution when they disagree;
  * emit dedup_ratio, cache_hit_rate and token_delta;
  * report concordance, precision and CI as unavailable, never imputed;
  * emit module hash, rollback command and a zero-write audit diff.

Read-only: writes no production labels, code, datasets or runtime databases.
Does not touch app/check_auto_apply.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from experiments.cmr_cavity_cluster_holdout_transition_check import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
    _classify,
)
from experiments.cmr_checked_cluster_replay import audit_diff, module_hash
from experiments.cmr_low_cavity_discriminator_scan import (
    COHORT_LABEL as S784_COHORT_LABEL,
    EXPECTED_PENDING as S784_EXPECTED_PENDING,
    NO_TRANSITION_BASELINE_S783,
    TOPOLOGY_FIELDS,
    _area_ratio,
    _is_unrouted,
    _spatial_decision,
    _topology_flags,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "shadow_cmr_dedup_cache_denom_gate/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_with_frozen_split"

# Denominator reconciliation: the three cohort counts that must agree before
# any execution is allowed.
DENOM_NON_HOLDOUT = 13
DENOM_SPATIAL_PENDING = 39
DENOM_FULL_PENDING = 61

# The three denominators are different views of the same queue, so the gate
# checks each against the supplied input rather than requiring them to be equal.
DENOM_KEYS = (
    ("non_holdout_in_queue", DENOM_NON_HOLDOUT),
    ("spatial_pending_candidates", DENOM_SPATIAL_PENDING),
    ("full_pending_queue", DENOM_FULL_PENDING),
)

# Issue types whose connected-component scan is cached per (case_id, issue_type).
ISSUE_TYPES = (
    "area_ratio",
    "cavity",
    "topology",
    "spatial_boundary",
    "secondary_components",
)

CACHE_HIT_RATE_MIN = 0.5

NO_WRITE_CLAUSE = (
    "Read-only shadow dedup cache + denominator gate. No production labels, "
    "code, datasets or runtime databases are written or mutated. Shadow "
    "triage only. app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_cmr_dedup_cache_denom_gate.py "
    "tests/test_shadow_cmr_dedup_cache_denom_gate.py"
)


def _reconcile_denominators(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile the 13/39/61 denominators against the supplied input.

    Each denominator is checked against the corresponding supplied count. A
    mismatch on any of the three blocks execution (``blocked`` is True).
    """
    frozen = data.get("frozen_split") if isinstance(data, dict) else None
    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    supplied: Dict[str, Optional[int]] = {
        "non_holdout_in_queue": (
            frozen.get("non_holdout_in_queue") if isinstance(frozen, dict) else None
        ),
        "spatial_pending_candidates": (
            spatial.get("pending_candidates") if isinstance(spatial, dict) else None
        ),
        "full_pending_queue": (
            data.get("full_pending_queue") if isinstance(data, dict) else None
        ),
    }
    checks: Dict[str, Any] = {}
    mismatches: List[str] = []
    for key, expected in DENOM_KEYS:
        actual = supplied.get(key)
        ok = actual == expected
        checks[key] = {
            "expected": expected,
            "supplied": actual,
            "match": ok,
        }
        if not ok:
            mismatches.append(key)
    return {
        "checks": checks,
        "mismatches": mismatches,
        "blocked": bool(mismatches),
    }


def _scan_issue(
    rec: Dict[str, Any],
    issue_type: str,
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: set,
) -> Dict[str, Any]:
    """Single connected-component scan for one (case_id, issue_type) key."""
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    cavity = rec.get("before_cavity")
    cluster = _classify(area_ratio, cavity)
    if issue_type == "area_ratio":
        value: Any = area_ratio
    elif issue_type == "cavity":
        value = cavity
    elif issue_type == "topology":
        value = _topology_flags(rec)
    elif issue_type == "spatial_boundary":
        value = _spatial_decision(cid, spatial_index, observed)
    else:
        value = rec.get("secondary_components")
    return {
        "case_id": cid,
        "issue_type": issue_type,
        "cluster": cluster,
        "value": value,
        "unrouted": _is_unrouted(cluster, cavity),
    }


def _run_with_cache(
    records: List[Dict[str, Any]],
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: set,
) -> Dict[str, Any]:
    """Execute the per-issue scan with a (case_id, issue_type) dedup cache."""
    cache: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    hits = 0
    misses = 0
    scans = 0
    per_case: Dict[str, Any] = {}
    for rec in records:
        cid = rec.get("case_id")
        case_out: Dict[str, Any] = {}
        for issue_type in ISSUE_TYPES:
            key = (cid, issue_type)
            if key in cache:
                hits += 1
                case_out[issue_type] = cache[key]
                continue
            misses += 1
            scans += 1
            result = _scan_issue(rec, issue_type, spatial_index, observed)
            cache[key] = result
            case_out[issue_type] = result
        per_case[cid] = case_out
    total = hits + misses
    cache_hit_rate: Optional[float] = None
    if total:
        cache_hit_rate = round(hits / total, 6)
    dedup_ratio: Optional[float] = None
    if scans:
        dedup_ratio = round(1.0 - (scans / total), 6) if total else None
    return {
        "cache": cache,
        "hits": hits,
        "misses": misses,
        "scans": scans,
        "total_lookups": total,
        "cache_hit_rate": cache_hit_rate,
        "dedup_ratio": dedup_ratio,
        "per_case": per_case,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only dedup cache + denominator gate over the supplied volumes."""
    raw_records = data.get("records", []) if isinstance(data, dict) else []
    holdout_ids = data.get("holdout_case_ids") if isinstance(data, dict) else None
    holdout = set(holdout_ids or [])

    denom = _reconcile_denominators(data)

    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    spatial_records: List[Dict[str, Any]] = []
    if isinstance(spatial, dict):
        spatial_records = spatial.get("records", []) or []
    observed = _checked_rank_core_set(spatial_records)
    spatial_index: Dict[Any, Dict[str, Any]] = {
        r.get("case_id"): r for r in spatial_records
    }

    # Deduplicate case_ids; the frozen holdout is excluded from the pending
    # execution set by case_id.
    selected: List[Dict[str, Any]] = []
    seen: set = set()
    duplicates: List[str] = []
    holdout_excluded = 0
    for rec in raw_records:
        cid = rec.get("case_id")
        if cid in holdout:
            holdout_excluded += 1
            continue
        if cid in seen:
            duplicates.append(cid)
            continue
        seen.add(cid)
        selected.append(rec)

    n = len(selected)
    executed = not denom["blocked"]

    if executed:
        run = _run_with_cache(selected, spatial_index, observed)
    else:
        run = {
            "cache": {},
            "hits": 0,
            "misses": 0,
            "scans": 0,
            "total_lookups": 0,
            "cache_hit_rate": None,
            "dedup_ratio": None,
            "per_case": {},
        }

    # token_delta: tokens saved by reusing cached scans instead of re-scanning.
    # Each avoided scan is one token; a negative delta means fewer tokens used.
    token_delta = -run["hits"] if executed else 0

    cache_hit_rate = run["cache_hit_rate"]
    cache_hit_rate_meets = (
        cache_hit_rate is not None and cache_hit_rate > CACHE_HIT_RATE_MIN
    )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_cmr_dedup_cache_denom_gate",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "dedup/cache metrics are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "topology_fields": list(TOPOLOGY_FIELDS),
            "issue_types": list(ISSUE_TYPES),
            "cache_key": "case_id+issue_type",
        },
        "denominator_gate": {
            "expected": {
                "non_holdout_in_queue": DENOM_NON_HOLDOUT,
                "spatial_pending_candidates": DENOM_SPATIAL_PENDING,
                "full_pending_queue": DENOM_FULL_PENDING,
            },
            "checks": denom["checks"],
            "mismatches": denom["mismatches"],
            "denominator_mismatch_blocked": denom["blocked"],
            "executed": executed,
        },
        "n_records": n,
        "volumes_scanned": n,
        "holdout_excluded_count": holdout_excluded,
        "duplicate_case_ids": duplicates,
        "input_record_count": len(raw_records),
        "label_write": False,
        "target_path": "experiments/shadow_cmr_dedup_cache_denom_gate.py",
        "cache": {
            "hits": run["hits"],
            "misses": run["misses"],
            "scans": run["scans"],
            "total_lookups": run["total_lookups"],
            "cache_hit_rate": cache_hit_rate,
            "cache_hit_rate_threshold": CACHE_HIT_RATE_MIN,
            "cache_hit_rate_meets_threshold": cache_hit_rate_meets,
            "dedup_ratio": run["dedup_ratio"],
            "token_delta": token_delta,
            "token_delta_negative": token_delta < 0,
        },
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "topology_deltas": "no masks, frame geometry or centroid series supplied",
            "missing_fields": "missing fields reported per-case, not imputed",
        },
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
        "cases": run["per_case"],
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
