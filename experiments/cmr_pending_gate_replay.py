"""Read-only S781 replay of the S777 spatial gate over 146 pending CMR volumes.

Scope (shadow_triage_only, zero production writes):
  * take the supplied pending CMR volumes and apply the existing S777 spatial
    gate (shadow_spatial_boundary) together with the existing area/cavity
    cluster classifier (cmr_cavity_cluster_holdout_transition_check);
  * emit per-cluster simulated checked/trash transition counts whose total
    equals the pending denominator (146);
  * emit per-case topology flags read from explicit supplied fields, never
    imputed (unavailable when absent);
  * reconcile the unrouted (cavity > CAVITY_MAX) count against the S780
    baseline of 22 and report the delta, or mark it unavailable;
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

MODULE_VERSION = "cmr_pending_gate_replay/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_PENDING = 146

UNROUTED_BASELINE = 22

TOPOLOGY_FIELDS = (
    "connected_components",
    "holes",
    "touches_border",
    "temporal_holes",
    "centroid_drift",
)

NO_WRITE_CLAUSE = (
    "Read-only S777 spatial-gate replay. No production labels, code, datasets "
    "or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_pending_gate_replay.py "
    "tests/test_cmr_pending_gate_replay.py"
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


def _is_unrouted(cluster: Optional[str], cavity: Optional[float]) -> bool:
    """A case is unrouted when the classifier cannot place it (cluster is None)
    or when its cavity exceeds CAVITY_MAX, matching the S780 unrouted-cavity
    semantics. Missing measurements are not treated as unrouted here; they are
    reported separately and never imputed."""
    if cluster is None:
        return True
    if cavity is not None and cavity > CAVITY_MAX:
        return True
    return False


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only S777 spatial-gate replay over the supplied pending CMR volumes."""
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

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    topology_unavailable: Dict[str, int] = {f: 0 for f in TOPOLOGY_FIELDS}
    cases: Dict[str, Any] = {}
    unrouted_ids: List[str] = []
    missing_measurement: List[str] = []
    spatial_candidate_count = 0
    spatial_suppressed_count = 0

    for rec in records:
        cid = rec.get("case_id")
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        cluster = _classify(area_ratio, cavity)
        if area_ratio is None or cavity is None:
            missing_measurement.append(cid)

        topology = _topology_flags(rec)
        for field, info in topology.items():
            if info["value"] is None:
                topology_unavailable[field] += 1

        spatial_out = _spatial_decision(cid, spatial_index, observed)
        spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION
        if spatial_hit:
            spatial_candidate_count += 1

        if _is_unrouted(cluster, cavity):
            unrouted_ids.append(cid)
            transition = "no_transition"
            routing = "unrouted"
        elif spatial_hit:
            # S777 gate suppresses spatial-boundary candidates before routing.
            spatial_suppressed_count += 1
            transition = "no_transition"
            routing = "spatial_gate_suppressed"
        else:
            counts[cluster] += 1
            transition = _simulate_transition(cluster, area_ratio, cavity)
            transitions[cluster][transition] += 1
            routing = ROUTING_MODE

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "topology": topology,
            "spatial_rule": spatial_out,
            "spatial_gate_suppressed": spatial_hit,
            "routing": routing,
            "simulated_transition": transition,
        }

    classified_total = sum(counts.values())
    simulated_checked = sum(transitions[c]["checked"] for c in CLUSTERS)
    simulated_trash = sum(transitions[c]["trash"] for c in CLUSTERS)
    simulated_no_transition = sum(transitions[c]["no_transition"] for c in CLUSTERS)
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    unrouted_count = len(unrouted_ids)

    if n == EXPECTED_PENDING:
        reconciliation_delta: Optional[int] = unrouted_count - UNROUTED_BASELINE
        reconciliation_status = "ok"
    else:
        reconciliation_delta = None
        reconciliation_status = "unavailable_pending_denominator_mismatch"

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_pending_gate_replay",
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
        "expected_pending": EXPECTED_PENDING,
        "pending_denominator_matches_expected": n == EXPECTED_PENDING,
        "cluster_counts": {
            c: {"count": counts[c], "denominator": n} for c in CLUSTERS
        },
        "classified_total": classified_total,
        "simulated_transitions": {
            c: {
                "checked": transitions[c]["checked"],
                "trash": transitions[c]["trash"],
                "no_transition": transitions[c]["no_transition"],
                "denominator": counts[c],
                "cohort": COHORT_LABEL,
            }
            for c in CLUSTERS
        },
        "simulated_checked_total": simulated_checked,
        "simulated_trash_total": simulated_trash,
        "simulated_no_transition_total": simulated_no_transition,
        "simulated_total": simulated_total,
        "simulated_total_matches_pending": simulated_total == n,
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
        "unrouted_count": unrouted_count,
        "unrouted_cases": unrouted_ids,
        "unrouted_baseline": UNROUTED_BASELINE,
        "unrouted_reconciliation_delta_vs_22": reconciliation_delta,
        "unrouted_reconciliation_status": reconciliation_status,
        "topology_unavailable_counts": {
            f: {"count": topology_unavailable[f], "denominator": n}
            for f in TOPOLOGY_FIELDS
        },
        "missing_measurement_cases": missing_measurement,
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
        "cases": cases,
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
