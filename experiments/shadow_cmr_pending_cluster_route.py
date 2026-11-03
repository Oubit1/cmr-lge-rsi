"""Read-only shadow scanner: per-cluster route simulation for the 13 non-holdout
pending CMR volumes, selected explicitly by case_id.

Scope (shadow_triage_only, zero production writes):
  * select exactly the non-holdout pending volumes by case_id, excluding the
    frozen holdout (holdout_selected==0) and deduplicating case_ids;
  * reuse the existing area/cavity subscore helpers, S783 cluster classifier,
    S785 threshold replay, S825 secondary-component fields and the anonymous
    spatial rank/core gate (no second harness);
  * simulate checked/trash/no_transition per cluster and populate a suppression
    reason for every no_transition (unrouted / spatial_gate_suppressed /
    missing_measurement);
  * report concordance, precision, CI and mask-derived deltas as unavailable
    when no independent verdicts or masks are supplied, and never impute them;
  * emit module hash, rollback command and a zero-write audit diff.

Read-only: writes no production labels, code, datasets or runtime databases.
Does not touch app/check_auto_apply.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from experiments.cmr_cavity_cluster_holdout_transition_check import (
    AREA_RATIO_MIN,
    CAVITY_MAX,
    CLUSTERS,
    SUPPRESSION_CONDITIONS,
    _classify,
)
from experiments.cmr_checked_cluster_replay import (
    _apply_threshold,
    audit_diff,
    module_hash,
)
from experiments.cmr_checked_excess_comp_volume_scan import (
    SECONDARY_AREA_RATIO_THRESHOLD,
    _secondary_component_fields,
)
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
from experiments.cmr_low_cavity_threshold_replay import (
    ADEQUATE_CLUSTER,
    LOW_AREA_CLUSTER,
    _cluster_area_ratios,
    _derive_threshold,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "shadow_cmr_pending_cluster_route/1.0"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_NON_HOLDOUT = 13

REASON_POPULATED_RATE_MIN = 0.9

NO_WRITE_CLAUSE = (
    "Read-only shadow per-cluster route simulation. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_cmr_pending_cluster_route.py "
    "tests/test_shadow_cmr_pending_cluster_route.py"
)


def _select_non_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select non-holdout volumes by case_id, deduplicating case_ids.

    The frozen holdout is excluded by case_id; ``holdout_selected`` is the
    number of holdout volumes that ended up in the selected set, which is
    always 0 by construction. ``holdout_excluded_count`` records how many
    holdout volumes were filtered out for transparency.
    """
    holdout = set(holdout_ids or [])
    selected: List[Dict[str, Any]] = []
    holdout_excluded = 0
    seen: set = set()
    duplicates: List[str] = []
    for rec in records:
        cid = rec.get("case_id")
        if cid in holdout:
            holdout_excluded += 1
            continue
        if cid in seen:
            duplicates.append(cid)
            continue
        seen.add(cid)
        selected.append(rec)
    holdout_selected = sum(1 for r in selected if r.get("case_id") in holdout)
    return {
        "selected": selected,
        "holdout_selected": holdout_selected,
        "holdout_excluded_count": holdout_excluded,
        "duplicate_case_ids": duplicates,
        "input_count": len(records),
    }


def _route_case(
    rec: Dict[str, Any],
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: set,
) -> Dict[str, Any]:
    """Simulate the route for one volume and always populate a reason."""
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    cavity = rec.get("before_cavity")
    cluster = _classify(area_ratio, cavity)
    topology = _topology_flags(rec)
    secondary = _secondary_component_fields(rec)
    spatial_out = _spatial_decision(cid, spatial_index, observed)
    spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION

    if area_ratio is None or cavity is None:
        transition = "no_transition"
        reason = "missing_measurement"
    elif _is_unrouted(cluster, cavity):
        transition = "no_transition"
        reason = "unrouted"
    elif spatial_hit:
        transition = "no_transition"
        reason = "spatial_gate_suppressed"
    else:
        transition = None
        reason = ROUTED_REASON

    return {
        "case_id": cid,
        "area_ratio": area_ratio,
        "cavity": cavity,
        "cluster": cluster,
        "topology": topology,
        "secondary_components": secondary,
        "spatial_rule": spatial_out,
        "spatial_gate_suppressed": spatial_hit,
        "routing": reason,
        "suppression_reason": reason,
        "simulated_transition": transition,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only per-cluster route simulation over the supplied pending volumes."""
    raw_records = data.get("records", []) if isinstance(data, dict) else []
    holdout_ids = data.get("holdout_case_ids") if isinstance(data, dict) else None

    selection = _select_non_holdout(raw_records, holdout_ids)
    records = selection["selected"]
    n = len(records)

    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    spatial_records: List[Dict[str, Any]] = []
    if isinstance(spatial, dict):
        spatial_records = spatial.get("records", []) or []
    observed = _checked_rank_core_set(spatial_records)
    spatial_index: Dict[Any, Dict[str, Any]] = {
        r.get("case_id"): r for r in spatial_records
    }

    cluster_ratios = _cluster_area_ratios(records)
    threshold = _derive_threshold(
        cluster_ratios.get(ADEQUATE_CLUSTER, []),
        cluster_ratios.get(LOW_AREA_CLUSTER, []),
    )

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    suppression_reasons: Dict[str, int] = {
        ROUTED_REASON: 0,
        "unrouted": 0,
        "spatial_gate_suppressed": 0,
        "missing_measurement": 0,
    }
    cases: Dict[str, Any] = {}
    no_transition_without_reason: List[str] = []
    spatial_candidate_count = 0
    spatial_suppressed_count = 0
    simulated_checked = 0
    simulated_trash = 0
    simulated_no_transition = 0

    for rec in records:
        routed = _route_case(rec, spatial_index, observed)
        cid = routed["case_id"]
        cluster = routed["cluster"]
        transition = routed["simulated_transition"]
        reason = routed["suppression_reason"]

        if routed["spatial_rule"].get("decision") == CANDIDATE_DECISION:
            spatial_candidate_count += 1
        if routed["spatial_gate_suppressed"]:
            spatial_suppressed_count += 1

        if transition is None:
            counts[cluster] += 1
            transition = _apply_threshold(cluster, routed["area_ratio"], threshold)
            transitions[cluster][transition] += 1
            routed["simulated_transition"] = transition

        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1
        if transition == "no_transition" and not reason:
            no_transition_without_reason.append(cid)

        cases[cid] = routed

    classified_total = sum(counts.values())
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    reason_populated = simulated_no_transition - len(no_transition_without_reason)
    reason_populated_rate: Optional[float] = None
    if simulated_no_transition:
        reason_populated_rate = round(reason_populated / simulated_no_transition, 6)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_cmr_pending_cluster_route",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "per-cluster route counts are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "secondary_area_ratio_threshold": SECONDARY_AREA_RATIO_THRESHOLD,
            "topology_fields": list(TOPOLOGY_FIELDS),
        },
        "n_records": n,
        "volumes_scanned": n,
        "non_holdout_volumes_scanned": n,
        "expected_non_holdout": EXPECTED_NON_HOLDOUT,
        "non_holdout_denominator_matches_expected": n == EXPECTED_NON_HOLDOUT,
        "holdout_selected": selection["holdout_selected"],
        "holdout_excluded_count": selection["holdout_excluded_count"],
        "input_record_count": selection["input_count"],
        "duplicate_case_ids": selection["duplicate_case_ids"],
        "label_write": False,
        "target_path": "experiments/shadow_cmr_pending_cluster_route.py",
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
        "simulated_total_matches_denominator": simulated_total == n,
        "suppression_reasons": suppression_reasons,
        "no_transition_without_reason": no_transition_without_reason,
        "no_transition_without_reason_count": len(no_transition_without_reason),
        "no_transition_reason_populated_rate": reason_populated_rate,
        "no_transition_reason_populated_rate_threshold": REASON_POPULATED_RATE_MIN,
        "no_transition_reason_populated_rate_meets_threshold": (
            reason_populated_rate is not None
            and reason_populated_rate > REASON_POPULATED_RATE_MIN
        ),
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
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
        "topology_deltas": {
            "components_delta": None,
            "holes_delta": None,
            "border_contact_delta": None,
            "status": "unavailable_no_masks",
            "note": (
                "No masks, frame geometry or centroid series supplied; "
                "component/hole/border deltas are reported unavailable and "
                "never imputed."
            ),
        },
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "topology_deltas": "no masks, frame geometry or centroid series supplied",
            "secondary_component_fields": (
                "secondary component count/area/adjacency/hole/border/drift "
                "reported per-case when supplied, otherwise unavailable; never "
                "imputed"
            ),
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
