"""Read-only S785 per-cluster transition-threshold replay over 20 stratified pending CMR volumes.

Scope (shadow_triage_only, zero production writes):
  * reuse the existing S784 discriminator scan
    (cmr_low_cavity_discriminator_scan) and the S782/S783 cluster classifier
    (cmr_cavity_cluster_holdout_transition_check) plus the S777 spatial gate
    (shadow_spatial_boundary);
  * derive a per-cluster transition threshold on the area_ratio axis that
    separates the adequate_area_low_cavity cluster from the low_area cluster,
    with an explicit separation margin;
  * emit per-cluster simulated checked/trash/no_transition counts whose total
    equals the stratified pending denominator (20);
  * reconcile the simulated no_transition total against the S783 108-case
    no_transition baseline and the S784 95-case adequate_area_low_cavity
    denominator, or mark reconciliation unavailable when the denominator does
    not match;
  * emit module hash, rollback command and an empty zero-write audit diff.

Reuses existing helpers (no second harness). Read-only: writes no production
labels, code, datasets or runtime databases. Topology flags, concordance,
precision and CI against independent reference labels are reported unavailable,
never imputed.
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
from experiments.cmr_low_cavity_discriminator_scan import (
    COHORT_LABEL as S784_COHORT_LABEL,
    EXPECTED_PENDING as S784_EXPECTED_PENDING,
    NO_TRANSITION_BASELINE_S783,
    _area_ratio,
    _is_unrouted,
    _spatial_decision,
    _topology_flags,
    TOPOLOGY_FIELDS,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "cmr_low_cavity_threshold_replay/1.0"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "stratified_pending_queue_not_independent_holdout"

EXPECTED_PENDING = 20

ADEQUATE_CLUSTER = "adequate_area_low_cavity"

LOW_AREA_CLUSTER = "low_area"

NO_WRITE_CLAUSE = (
    "Read-only S785 low-cavity threshold replay. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_low_cavity_threshold_replay.py "
    "tests/test_cmr_low_cavity_threshold_replay.py"
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


def _cluster_area_ratios(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect area_ratio values per cluster for the routed (non-unrouted) cases."""
    out: Dict[str, List[float]] = {c: [] for c in CLUSTERS}
    for rec in records:
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        cluster = _classify(area_ratio, cavity)
        if cluster is None or area_ratio is None:
            continue
        if _is_unrouted(cluster, cavity):
            continue
        out[cluster].append(area_ratio)
    return out


def _derive_threshold(
    adequate: List[float], low_area: List[float]
) -> Dict[str, Any]:
    """Derive a threshold on area_ratio separating adequate_area_low_cavity from
    low_area, with an explicit separation margin.

    The threshold is the midpoint between the minimum adequate area_ratio and the
    maximum low_area area_ratio. The margin is the gap between those two values.
    When either cluster is empty or the clusters overlap, the threshold is
    reported unavailable and never imputed.
    """
    if not adequate or not low_area:
        return {
            "threshold": None,
            "margin": None,
            "status": "unavailable_empty_cluster",
            "adequate_min": min(adequate) if adequate else None,
            "low_area_max": max(low_area) if low_area else None,
        }
    adequate_min = min(adequate)
    low_area_max = max(low_area)
    margin = adequate_min - low_area_max
    if margin <= 0:
        return {
            "threshold": None,
            "margin": margin,
            "status": "unavailable_clusters_overlap",
            "adequate_min": adequate_min,
            "low_area_max": low_area_max,
        }
    return {
        "threshold": (adequate_min + low_area_max) / 2.0,
        "margin": margin,
        "status": "ok",
        "adequate_min": adequate_min,
        "low_area_max": low_area_max,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only S785 per-cluster transition-threshold replay."""
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
    suppression_reasons: Dict[str, int] = {
        ROUTED_REASON: 0,
        "unrouted": 0,
        "spatial_gate_suppressed": 0,
    }
    topology_unavailable: Dict[str, int] = {f: 0 for f in TOPOLOGY_FIELDS}
    cases: Dict[str, Any] = {}
    unrouted_ids: List[str] = []
    missing_measurement: List[str] = []
    spatial_candidate_count = 0
    spatial_suppressed_count = 0

    simulated_checked = 0
    simulated_trash = 0
    simulated_no_transition = 0

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
            transition = _simulate_transition(cluster, area_ratio, cavity)
            routing = "unrouted"
        elif spatial_hit:
            spatial_suppressed_count += 1
            transition = "no_transition"
            routing = "spatial_gate_suppressed"
        else:
            counts[cluster] += 1
            transition = _simulate_transition(cluster, area_ratio, cavity)
            transitions[cluster][transition] += 1
            routing = ROUTED_REASON

        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        suppression_reasons[routing] = suppression_reasons.get(routing, 0) + 1

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
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    unrouted_count = len(unrouted_ids)

    cluster_ratios = _cluster_area_ratios(records)
    threshold = _derive_threshold(
        cluster_ratios.get(ADEQUATE_CLUSTER, []),
        cluster_ratios.get(LOW_AREA_CLUSTER, []),
    )

    if n == EXPECTED_PENDING:
        reconciliation_delta: Optional[int] = (
            simulated_no_transition - NO_TRANSITION_BASELINE_S783
        )
        reconciliation_status = "ok"
    else:
        reconciliation_delta = None
        reconciliation_status = "unavailable_pending_denominator_mismatch"

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_low_cavity_threshold_replay",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are stratified pending candidates, not verified "
            "truth; concordance, precision and CI against independent labels "
            "stay unavailable."
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
        "suppression_reasons": suppression_reasons,
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
        "unrouted_count": unrouted_count,
        "unrouted_cases": unrouted_ids,
        "threshold": {
            "adequate_cluster": ADEQUATE_CLUSTER,
            "low_area_cluster": LOW_AREA_CLUSTER,
            "axis": "area_ratio",
            "value": threshold["threshold"],
            "margin": threshold["margin"],
            "status": threshold["status"],
            "adequate_min": threshold["adequate_min"],
            "low_area_max": threshold["low_area_max"],
        },
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "no_transition_reconciliation_delta_vs_s783": reconciliation_delta,
        "no_transition_reconciliation_status": reconciliation_status,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
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
