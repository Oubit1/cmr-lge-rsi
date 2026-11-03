"""Read-only S780 scan of unrouted pending CMR volumes (cavity>0.5).

Scope (shadow_triage_only, zero production writes):
  * take the S779 pending-volume cluster check output and isolate the
    unrouted cases (cluster is None, i.e. cavity > CAVITY_MAX);
  * emit per-case cavity ratio, topology flags (connected components,
    holes, border contact, temporal holes, centroid drift) read from
    explicit supplied fields, never imputed;
  * map each unrouted case onto the existing spatial rule
    (shadow_spatial_boundary) and the existing area rule
    (AREA_RATIO_MIN / CAVITY_MAX) and report overlap counts;
  * simulate checked/trash transitions with the shared S779 helper and
    report counts summing to the unrouted denominator;
  * emit module hash, rollback command and an empty zero-write audit diff.

Reuses existing helpers (no second harness). Read-only: writes no production
labels, code, datasets or runtime databases. Concordance, precision and CI
against independent reference labels are reported unavailable, never imputed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, List, Optional

from experiments.cmr_cavity_cluster_holdout_transition_check import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
    _classify,
    _simulate_transition,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
    _discriminate,
)

MODULE_VERSION = "cmr_unrouted_cavity_scan/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_TOTAL = 156

EXPECTED_UNROUTED = 22

TOPOLOGY_FIELDS = (
    "connected_components",
    "holes",
    "touches_border",
    "temporal_holes",
    "centroid_drift",
)

NO_WRITE_CLAUSE = (
    "Read-only unrouted-cavity scan. No production labels, code, datasets "
    "or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_unrouted_cavity_scan.py "
    "tests/test_cmr_unrouted_cavity_scan.py"
)


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(module_hash))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


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


def _area_ratio(rec: Dict[str, Any]) -> Optional[float]:
    ref = rec.get("reference_area")
    area = rec.get("before_area")
    if ref in (None, 0) or area is None:
        return None
    return area / ref


def _topology_flags(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Per-case topology flags from explicit supplied fields.

    Each field is read from the record; when absent it is reported
    unavailable (None), never assumed zero.
    """
    out: Dict[str, Any] = {}
    for field in TOPOLOGY_FIELDS:
        val = rec.get(field)
        out[field] = {
            "value": val,
            "status": "ok" if val is not None else "unavailable_no_masks_or_geometry",
        }
    return out


def _spatial_decision(
    cid: Any,
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: Any,
) -> Dict[str, Any]:
    rec = spatial_index.get(cid)
    if rec is None:
        return {
            "decision": "unavailable_no_spatial_control_record",
            "reason": None,
            "rank": None,
            "core": None,
            "status": "unavailable",
        }
    out = _discriminate(rec, observed)
    return {
        "decision": out.get("decision"),
        "reason": out.get("reason"),
        "rank": out.get("rank"),
        "core": out.get("core"),
        "status": out.get("status"),
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only unrouted-cavity scan over the supplied CMR volumes."""
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)

    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    spatial_records: List[Dict[str, Any]] = []
    if isinstance(spatial, dict):
        spatial_records = spatial.get("records", []) or []
    observed = _checked_rank_core_set(spatial_records)
    spatial_index: Dict[Any, Dict[str, Any]] = {
        r.get("case_id"): r for r in spatial_records
    }

    unrouted_cases: Dict[str, Any] = {}
    unrouted_ids: List[str] = []
    missing_measurement: List[str] = []
    spatial_overlap = 0
    area_overlap = 0
    both_overlap = 0
    simulated_checked = 0
    simulated_trash = 0
    simulated_no_transition = 0
    topology_unavailable: Dict[str, int] = {f: 0 for f in TOPOLOGY_FIELDS}

    for rec in records:
        cid = rec.get("case_id")
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        cluster = _classify(area_ratio, cavity)
        if area_ratio is None or cavity is None:
            missing_measurement.append(cid)
        if cluster is not None:
            continue

        unrouted_ids.append(cid)

        topology = _topology_flags(rec)
        for field, info in topology.items():
            if info["value"] is None:
                topology_unavailable[field] += 1

        spatial_out = _spatial_decision(cid, spatial_index, observed)
        spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION
        area_hit = cavity is not None and cavity > CAVITY_MAX
        if spatial_hit:
            spatial_overlap += 1
        if area_hit:
            area_overlap += 1
        if spatial_hit and area_hit:
            both_overlap += 1

        transition = _simulate_transition(cluster, area_ratio, cavity)
        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        unrouted_cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "topology": topology,
            "spatial_rule": spatial_out,
            "spatial_rule_overlap": spatial_hit,
            "area_rule_overlap": area_hit,
            "simulated_transition": transition,
        }

    unrouted_count = len(unrouted_ids)
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_unrouted_cavity_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "concordance, precision and CI against independent labels stay "
            "unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
        },
        "n_records": n,
        "expected_total": EXPECTED_TOTAL,
        "expected_unrouted": EXPECTED_UNROUTED,
        "unrouted_count": unrouted_count,
        "unrouted_count_matches_expected": unrouted_count == EXPECTED_UNROUTED,
        "unrouted_denominator": n,
        "unrouted_cases": unrouted_ids,
        "missing_measurement_cases": missing_measurement,
        "spatial_rule_overlap_count": spatial_overlap,
        "area_rule_overlap_count": area_overlap,
        "spatial_and_area_overlap_count": both_overlap,
        "simulated_transitions": {
            "checked": simulated_checked,
            "trash": simulated_trash,
            "no_transition": simulated_no_transition,
            "total": simulated_total,
            "denominator": unrouted_count,
            "cohort": COHORT_LABEL,
        },
        "topology_unavailable_counts": {
            f: {"count": topology_unavailable[f], "denominator": unrouted_count}
            for f in TOPOLOGY_FIELDS
        },
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "topology_flags": "no masks, frame geometry or centroid series supplied",
            "missing_fields": "missing fields reported per-case, not imputed",
        },
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
        "cases": unrouted_cases,
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
