"""Read-only shadow scanner: apply the expand_undersegmented_region topology guard
to the 48 frozen-holdout CMR volumes, selected explicitly by holdout_case_ids,
and emit per-case simulated checked/trash/no_transition with a populated
suppression reason for every no_transition.

Scope (shadow_triage_only, zero production writes):
  * select exactly the frozen-holdout volumes by case_id (holdout_selected is
    the count of holdout volumes in the selected set, which is 0 by
    construction because the selection is the holdout set itself);
  * reuse the existing area/cavity subscore helpers, S783 cluster classifier,
    S785 threshold replay and the anonymous spatial rank/core gate (no second
    harness);
  * apply the expand_undersegmented_region topology guard per volume and report
    a guard applicability rate plus per-cluster simulated transition counts;
  * populate a suppression reason for every no_transition (unrouted /
    spatial_gate_suppressed / missing_measurement / topology_guard_suppressed);
  * report concordance, precision, CI and mask-derived topology deltas as
    unavailable when no independent verdicts or masks are supplied, and never
    impute them;
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

MODULE_VERSION = "shadow_cmr_holdout_topology_guard_scan/1.0"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

TOPOLOGY_GUARD_SUPPRESSED = "topology_guard_suppressed"

COHORT_LABEL = "frozen_holdout_queue_not_independent_truth"

# S839: the frozen holdout cohort is the 48-volume split reported by the
# frozen_split block of the supplied evaluation input.
EXPECTED_HOLDOUT = 48

GUARD_APPLICABLE_RATE_MIN = 0.9

REASON_POPULATED_RATE_MIN = 0.9

# expand_undersegmented_region topology guard: a small-area cluster whose
# area_ratio is below the guard floor is treated as an undersegmented region
# that the guard would expand, so the shadow route is suppressed rather than
# promoted to a checked/trash transition.
GUARD_AREA_RATIO_FLOOR = 0.5

NO_WRITE_CLAUSE = (
    "Read-only shadow holdout topology-guard scan. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_cmr_holdout_topology_guard_scan.py "
    "tests/test_shadow_cmr_holdout_topology_guard_scan.py"
)


def _select_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select exactly the frozen-holdout volumes by case_id, deduplicating.

    ``holdout_selected`` counts selected volumes that are NOT in the holdout
    set; because the selection is the holdout set itself, this is 0 by
    construction. ``non_holdout_excluded_count`` records how many non-holdout
    volumes were filtered out for transparency.
    """
    holdout = set(holdout_ids or [])
    selected: List[Dict[str, Any]] = []
    non_holdout_excluded = 0
    seen: set = set()
    duplicates: List[str] = []
    for rec in records:
        cid = rec.get("case_id")
        if cid not in holdout:
            non_holdout_excluded += 1
            continue
        if cid in seen:
            duplicates.append(cid)
            continue
        seen.add(cid)
        selected.append(rec)
    holdout_selected = sum(1 for r in selected if r.get("case_id") not in holdout)
    return {
        "selected": selected,
        "holdout_selected": holdout_selected,
        "non_holdout_excluded_count": non_holdout_excluded,
        "duplicate_case_ids": duplicates,
        "input_count": len(records),
    }


def _topology_guard(area_ratio: Optional[float]) -> Dict[str, Any]:
    """Apply the expand_undersegmented_region topology guard to one volume.

    The guard is applicable whenever an area_ratio measurement exists; it
    suppresses the route when the ratio falls below the guard floor, which is
    the undersegmented-region condition the guard expands.
    """
    if area_ratio is None:
        return {
            "guard": "expand_undersegmented_region",
            "applicable": False,
            "suppressed": False,
            "reason": "missing_measurement",
            "area_ratio": None,
            "floor": GUARD_AREA_RATIO_FLOOR,
        }
    suppressed = area_ratio < GUARD_AREA_RATIO_FLOOR
    return {
        "guard": "expand_undersegmented_region",
        "applicable": True,
        "suppressed": suppressed,
        "reason": TOPOLOGY_GUARD_SUPPRESSED if suppressed else ROUTED_REASON,
        "area_ratio": area_ratio,
        "floor": GUARD_AREA_RATIO_FLOOR,
    }


def _scan_case(
    rec: Dict[str, Any],
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: set,
) -> Dict[str, Any]:
    """Simulate the holdout route for one volume and always populate a reason."""
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    cavity = rec.get("before_cavity")
    cluster = _classify(area_ratio, cavity)
    topology = _topology_flags(rec)
    spatial_out = _spatial_decision(cid, spatial_index, observed)
    spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION
    guard = _topology_guard(area_ratio)

    if area_ratio is None or cavity is None:
        transition = "no_transition"
        reason = "missing_measurement"
    elif _is_unrouted(cluster, cavity):
        transition = "no_transition"
        reason = "unrouted"
    elif spatial_hit:
        transition = "no_transition"
        reason = "spatial_gate_suppressed"
    elif guard["suppressed"]:
        transition = "no_transition"
        reason = TOPOLOGY_GUARD_SUPPRESSED
    else:
        transition = None
        reason = ROUTED_REASON

    return {
        "case_id": cid,
        "area_ratio": area_ratio,
        "cavity": cavity,
        "cluster": cluster,
        "topology": topology,
        "spatial_rule": spatial_out,
        "spatial_gate_suppressed": spatial_hit,
        "topology_guard": guard,
        "routing": reason,
        "suppression_reason": reason,
        "simulated_transition": transition,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only holdout topology-guard scan over the supplied volumes."""
    raw_records = data.get("records", []) if isinstance(data, dict) else []
    holdout_ids = data.get("holdout_case_ids") if isinstance(data, dict) else None

    selection = _select_holdout(raw_records, holdout_ids)
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
        TOPOLOGY_GUARD_SUPPRESSED: 0,
    }
    cases: Dict[str, Any] = {}
    no_transition_without_reason: List[str] = []
    spatial_candidate_count = 0
    spatial_suppressed_count = 0
    guard_applicable_count = 0
    guard_suppressed_count = 0
    simulated_checked = 0
    simulated_trash = 0
    simulated_no_transition = 0

    for rec in records:
        scanned = _scan_case(rec, spatial_index, observed)
        cid = scanned["case_id"]
        cluster = scanned["cluster"]
        transition = scanned["simulated_transition"]
        reason = scanned["suppression_reason"]

        if scanned["spatial_rule"].get("decision") == CANDIDATE_DECISION:
            spatial_candidate_count += 1
        if scanned["spatial_gate_suppressed"]:
            spatial_suppressed_count += 1
        if scanned["topology_guard"]["applicable"]:
            guard_applicable_count += 1
        if scanned["topology_guard"]["suppressed"]:
            guard_suppressed_count += 1

        if transition is None:
            counts[cluster] += 1
            transition = _apply_threshold(cluster, scanned["area_ratio"], threshold)
            transitions[cluster][transition] += 1
            scanned["simulated_transition"] = transition

        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1
        if transition == "no_transition" and not reason:
            no_transition_without_reason.append(cid)

        cases[cid] = scanned

    classified_total = sum(counts.values())
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    reason_populated = simulated_no_transition - len(no_transition_without_reason)
    reason_populated_rate: Optional[float] = None
    if simulated_no_transition:
        reason_populated_rate = round(reason_populated / simulated_no_transition, 6)
    guard_applicable_rate: Optional[float] = None
    if n:
        guard_applicable_rate = round(guard_applicable_count / n, 6)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_cmr_holdout_topology_guard_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are the frozen holdout split, not verified truth; "
            "per-cluster counts are monitoring-grade and concordance, precision "
            "and CI against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "topology_fields": list(TOPOLOGY_FIELDS),
            "topology_guard": "expand_undersegmented_region",
            "topology_guard_area_ratio_floor": GUARD_AREA_RATIO_FLOOR,
        },
        "n_records": n,
        "volumes_scanned": n,
        "holdout_volumes_scanned": n,
        "expected_holdout": EXPECTED_HOLDOUT,
        "holdout_denominator_matches_expected": n == EXPECTED_HOLDOUT,
        "holdout_selected": selection["holdout_selected"],
        "non_holdout_excluded_count": selection["non_holdout_excluded_count"],
        "input_record_count": selection["input_count"],
        "duplicate_case_ids": selection["duplicate_case_ids"],
        "label_write": False,
        "target_path": "experiments/shadow_cmr_holdout_topology_guard_scan.py",
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
        "per_cluster_simulated_transition_counts_populated": all(
            transitions[c]["checked"]
            + transitions[c]["trash"]
            + transitions[c]["no_transition"]
            == counts[c]
            for c in CLUSTERS
        ),
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
        "topology_guard": {
            "guard": "expand_undersegmented_region",
            "applicable_count": guard_applicable_count,
            "suppressed_count": guard_suppressed_count,
            "denominator": n,
            "applicable_rate": guard_applicable_rate,
            "applicable_rate_threshold": GUARD_APPLICABLE_RATE_MIN,
            "applicable_rate_meets_threshold": (
                guard_applicable_rate is not None
                and guard_applicable_rate > GUARD_APPLICABLE_RATE_MIN
            ),
        },
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
