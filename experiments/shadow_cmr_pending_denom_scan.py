"""Read-only shadow scanner: exact 13 non-holdout pending CMR volume selection
by case_id, with per-cluster simulated checked/trash/no_transition counts and a
suppression reason for every no_transition.

Scope (shadow_triage_only, zero production writes):
  * select exactly the 13 non-holdout pending CMR volumes by case_id from the
    supplied pending queue, excluding the frozen holdout (holdout_selected==0);
  * report the 61-vs-13 dedup ratio that reconciles the pending denominator
    mismatch that blocked S826/S828 rule gates;
  * reuse the existing S783/S784 cluster classifier, S785 per-cluster threshold
    replay, S777 spatial rank/core gate and S825 secondary-component fields
    (no second harness);
  * emit per-cluster simulated checked/trash/no_transition counts whose total
    equals the selected denominator (13);
  * report a suppression reason for every no_transition (routed /
    unrouted / spatial_gate_suppressed);
  * report concordance, precision, CI and topology deltas as unavailable when
    no independent verdicts or masks are supplied, and never impute them;
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
    _simulate_transition,
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

MODULE_VERSION = "shadow_cmr_pending_denom_scan/1.1"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_PENDING = 13

FULL_PENDING_QUEUE = 61

HOLDOUT_STATE = "holdout"

NO_WRITE_CLAUSE = (
    "Read-only shadow denominator scan. No production labels, code, datasets "
    "or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_cmr_pending_denom_scan.py "
    "tests/test_shadow_cmr_pending_denom_scan.py"
)


def _select_non_holdout_pending(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select exactly the non-holdout pending CMR volumes by case_id.

    The frozen holdout is excluded by case_id; the 61-volume pending queue is
    not treated as an independent holdout. Selection is deterministic and
    preserves the supplied order. ``holdout_selected`` is the number of holdout
    volumes that ended up in the selected set, which is always 0 by
    construction; ``holdout_excluded_count`` records how many holdout volumes
    were filtered out for transparency.
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


def _dedup_ratio(selected_count: int, queue_count: int) -> Dict[str, Any]:
    """Report the 61-vs-13 dedup ratio that reconciles the denominator."""
    ratio: Optional[float] = None
    if queue_count:
        ratio = round(selected_count / queue_count, 6)
    return {
        "full_pending_queue": queue_count,
        "non_holdout_selected": selected_count,
        "dedup_ratio_selected_over_queue": ratio,
        "dedup_ratio_queue_over_selected": (
            round(queue_count / selected_count, 6) if selected_count else None
        ),
        "status": "ok" if selected_count == EXPECTED_PENDING else "denominator_mismatch",
        "note": (
            "The 61-volume pending queue is not an independent holdout; the "
            "frozen holdout stays excluded and the 13-volume subset is "
            "monitoring-grade only."
        ),
    }


def _reconciliation_table(
    n: int,
    simulated_checked: int,
    simulated_trash: int,
    simulated_no_transition: int,
) -> Dict[str, Any]:
    """Reconcile the three-cluster simulated total against the S783 baseline.

    Reconciliation is only reported when the selected denominator matches the
    expected 13; otherwise it is marked unavailable and never imputed.
    """
    denominator_ok = n == EXPECTED_PENDING
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    if denominator_ok:
        no_transition_delta: Optional[int] = (
            simulated_no_transition - NO_TRANSITION_BASELINE_S783
        )
        status = "ok"
    else:
        no_transition_delta = None
        status = "unavailable_pending_denominator_mismatch"
    return {
        "pending_denominator": n,
        "expected_pending_denominator": EXPECTED_PENDING,
        "pending_denominator_matches_expected": denominator_ok,
        "simulated_checked": simulated_checked,
        "simulated_trash": simulated_trash,
        "simulated_no_transition": simulated_no_transition,
        "simulated_total": simulated_total,
        "simulated_total_matches_denominator": simulated_total == n,
        "s783_no_transition_baseline": NO_TRANSITION_BASELINE_S783,
        "no_transition_delta_vs_s783": no_transition_delta,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "status": status,
        "note": (
            "S783/S784 baselines are aggregate cohort counts, not per-record "
            "truth; deltas are monitoring-grade reconciliation only."
        ),
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only shadow denominator scan over the 13 non-holdout volumes."""
    raw_records = data.get("records", []) if isinstance(data, dict) else []
    holdout_ids = data.get("holdout_case_ids") if isinstance(data, dict) else None
    queue_count = data.get("full_pending_queue") if isinstance(data, dict) else None
    if not isinstance(queue_count, int) or queue_count <= 0:
        queue_count = FULL_PENDING_QUEUE

    selection = _select_non_holdout_pending(raw_records, holdout_ids)
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
    }
    topology_unavailable: Dict[str, int] = {f: 0 for f in TOPOLOGY_FIELDS}
    cases: Dict[str, Any] = {}
    unrouted_ids: List[str] = []
    missing_measurement: List[str] = []
    secondary_unavailable: List[str] = []
    no_transition_without_reason: List[str] = []
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

        secondary = _secondary_component_fields(rec)
        if not secondary["secondary_fields_available"]:
            secondary_unavailable.append(cid)

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
            transition = _apply_threshold(cluster, area_ratio, threshold)
            transitions[cluster][transition] += 1
            routing = ROUTED_REASON

        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        suppression_reasons[routing] = suppression_reasons.get(routing, 0) + 1

        if transition == "no_transition" and not routing:
            no_transition_without_reason.append(cid)

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "topology": topology,
            "secondary_components": secondary,
            "spatial_rule": spatial_out,
            "spatial_gate_suppressed": spatial_hit,
            "routing": routing,
            "suppression_reason": routing,
            "simulated_transition": transition,
        }

    classified_total = sum(counts.values())
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    reconciliation = _reconciliation_table(
        n, simulated_checked, simulated_trash, simulated_no_transition
    )
    dedup = _dedup_ratio(n, queue_count)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_cmr_pending_denom_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "simulated counts are monitoring-grade and concordance, precision "
            "and CI against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "secondary_area_ratio_threshold": SECONDARY_AREA_RATIO_THRESHOLD,
        },
        "n_records": n,
        "non_holdout_volumes_scanned": n,
        "expected_pending": EXPECTED_PENDING,
        "pending_denominator_matches_expected": n == EXPECTED_PENDING,
        "holdout_selected": selection["holdout_selected"],
        "holdout_excluded_count": selection["holdout_excluded_count"],
        "input_record_count": selection["input_count"],
        "duplicate_case_ids": selection["duplicate_case_ids"],
        "dedup_ratio": dedup,
        "label_write": False,
        "target_path": "experiments/shadow_cmr_pending_denom_scan.py",
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
        "simulated_transitions_per_cluster": {
            c: transitions[c]["checked"]
            + transitions[c]["trash"]
            + transitions[c]["no_transition"]
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
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
        "unrouted_count": len(unrouted_ids),
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
        "reconciliation": reconciliation,
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "no_transition_reconciliation_delta_vs_s783": reconciliation[
            "no_transition_delta_vs_s783"
        ],
        "reconciliation_status": reconciliation["status"],
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "secondary_component_unavailable_cases": secondary_unavailable,
        "secondary_component_unavailable_count": len(secondary_unavailable),
        "topology_unavailable_counts": {
            f: {"count": topology_unavailable[f], "denominator": n}
            for f in TOPOLOGY_FIELDS
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
        "missing_measurement_cases": missing_measurement,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "topology_flags": "no masks, frame geometry or centroid series supplied",
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
