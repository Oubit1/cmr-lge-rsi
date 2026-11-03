"""Read-only CMR topology-delta shadow scan over the 13 non-holdout volumes.

Scope (shadow_triage_only, zero production writes):
  * select the 13 non-holdout CMR volumes by excluding holdout_case_ids;
  * per case emit connected-component / hole / border-contact / centroid
    deltas from supplied geometry fields (before/after counts, border flags,
    centroid coordinates); missing fields are reported unavailable, never
    imputed;
  * simulate the expand_undersegmented_region guard routing decision and
    record a no_transition_reason for every case;
  * emit topology_guard_applicable_rate and
    no_transition_reason_populated_rate;
  * report concordance, precision and CI as unavailable (no independent
    reference labels supplied);
  * emit module hash, rollback command and a zero-write audit diff.

Read-only: writes no production labels, code, datasets or runtime databases.
Does not touch app/check_auto_apply.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from experiments.cmr_checked_cluster_replay import audit_diff, module_hash
from experiments.cmr_low_cavity_discriminator_scan import (
    COHORT_LABEL as S784_COHORT_LABEL,
    EXPECTED_PENDING as S784_EXPECTED_PENDING,
    NO_TRANSITION_BASELINE_S783,
    TOPOLOGY_FIELDS,
    _area_ratio,
    _topology_flags,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "cmr_topology_delta_scan_exec_13/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "non_holdout_pending_cmr_topology_delta"

EXPECTED_NON_HOLDOUT = 13

GUARD_APPLICABLE_MIN = 0.9
NO_TRANSITION_REASON_MIN = 0.9

# Supplied geometry field aliases. First present, non-None value wins; nothing
# is imputed.
COMPONENT_BEFORE_FIELDS = (
    "before_components",
    "myo_components_before",
    "component_count_before",
    "components_before",
)
COMPONENT_AFTER_FIELDS = (
    "after_components",
    "myo_components_after",
    "component_count_after",
    "components_after",
)
HOLE_BEFORE_FIELDS = (
    "before_holes",
    "myo_holes_before",
    "hole_count_before",
    "holes_before",
)
HOLE_AFTER_FIELDS = (
    "after_holes",
    "myo_holes_after",
    "hole_count_after",
    "holes_after",
)
BORDER_BEFORE_FIELDS = (
    "before_touches_border",
    "touches_border_before",
    "border_contact_before",
)
BORDER_AFTER_FIELDS = (
    "after_touches_border",
    "touches_border_after",
    "border_contact_after",
)
CENTROID_BEFORE_FIELDS = (
    "before_centroid",
    "centroid_before",
    "myo_centroid_before",
)
CENTROID_AFTER_FIELDS = (
    "after_centroid",
    "centroid_after",
    "myo_centroid_after",
)

# expand_undersegmented_region guard: expansion is only allowed when the
# candidate is a spatial boundary candidate AND the area ratio is below the
# undersegmentation threshold AND the topology is not already fragmented.
UNDERSEG_AREA_RATIO_MAX = 0.6
FRAGMENTATION_FLAG = "fragmentation_candidate"

NO_WRITE_CLAUSE = (
    "Read-only CMR topology-delta shadow scan. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage "
    "only. app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_topology_delta_scan_exec_13.py "
    "tests/test_cmr_topology_delta_scan_exec_13.py"
)


def _first_present(rec: Dict[str, Any], fields: Tuple[str, ...]) -> Any:
    """Return the first present, non-None value, else None."""
    for name in fields:
        if name in rec and rec[name] is not None:
            return rec[name]
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _as_centroid(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _centroid_delta(
    before: Optional[List[float]], after: Optional[List[float]]
) -> Optional[Dict[str, float]]:
    if before is None or after is None:
        return None
    dx = round(after[0] - before[0], 6)
    dy = round(after[1] - before[1], 6)
    return {"dx": dx, "dy": dy, "magnitude": round((dx * dx + dy * dy) ** 0.5, 6)}


def _select_non_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
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
    return {
        "selected": selected,
        "holdout_excluded_count": holdout_excluded,
        "duplicate_case_ids": duplicates,
        "input_count": len(records),
    }


def _guard_route(
    rec: Dict[str, Any],
    spatial_rec: Optional[Dict[str, Any]],
    area_ratio: Optional[float],
    topology_flags: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate the expand_undersegmented_region guard routing decision."""
    spatial_decision = (
        spatial_rec.get("decision") if isinstance(spatial_rec, dict) else None
    )
    spatial_reason = (
        spatial_rec.get("reason") if isinstance(spatial_rec, dict) else None
    )
    flags = rec.get("quality_flags_before") or []
    fragmented = FRAGMENTATION_FLAG in flags
    underseg = area_ratio is not None and area_ratio < UNDERSEG_AREA_RATIO_MAX
    is_candidate = spatial_decision == CANDIDATE_DECISION

    applicable = is_candidate and underseg and not fragmented
    if applicable:
        route = "expand_undersegmented_region"
        reason = "undersegmented_spatial_candidate"
    elif is_candidate and fragmented:
        route = "hold_for_review"
        reason = "fragmentation_blocks_expansion"
    elif is_candidate and not underseg:
        route = "hold_for_review"
        reason = "area_ratio_above_undersegmentation_threshold"
    elif spatial_decision is None:
        route = "hold_for_review"
        reason = "no_spatial_control_record"
    else:
        route = "no_transition"
        reason = spatial_reason or OUTSIDE_REASON

    return {
        "guard_applicable": applicable,
        "guard_route": route,
        "no_transition_reason": reason,
        "spatial_decision": spatial_decision,
        "spatial_reason": spatial_reason,
        "fragmented": fragmented,
        "undersegmented": underseg,
        "topology_flags": topology_flags,
    }


def _scan_case(
    rec: Dict[str, Any],
    spatial_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    topology_flags = _topology_flags(rec)

    comp_before = _as_int(_first_present(rec, COMPONENT_BEFORE_FIELDS))
    comp_after = _as_int(_first_present(rec, COMPONENT_AFTER_FIELDS))
    hole_before = _as_int(_first_present(rec, HOLE_BEFORE_FIELDS))
    hole_after = _as_int(_first_present(rec, HOLE_AFTER_FIELDS))
    border_before = _as_bool(_first_present(rec, BORDER_BEFORE_FIELDS))
    border_after = _as_bool(_first_present(rec, BORDER_AFTER_FIELDS))
    centroid_before = _as_centroid(_first_present(rec, CENTROID_BEFORE_FIELDS))
    centroid_after = _as_centroid(_first_present(rec, CENTROID_AFTER_FIELDS))

    unavailable: List[str] = []
    if comp_before is None or comp_after is None:
        unavailable.append("component_delta")
    if hole_before is None or hole_after is None:
        unavailable.append("hole_delta")
    if border_before is None or border_after is None:
        unavailable.append("border_delta")
    if centroid_before is None or centroid_after is None:
        unavailable.append("centroid_delta")

    component_delta = (
        comp_after - comp_before
        if comp_before is not None and comp_after is not None
        else None
    )
    hole_delta = (
        hole_after - hole_before
        if hole_before is not None and hole_after is not None
        else None
    )
    border_delta = (
        int(border_after) - int(border_before)
        if border_before is not None and border_after is not None
        else None
    )
    centroid_delta = _centroid_delta(centroid_before, centroid_after)

    guard = _guard_route(rec, spatial_rec, area_ratio, topology_flags)

    return {
        "case_id": cid,
        "area_ratio": area_ratio,
        "before_area": rec.get("before_area"),
        "after_area": rec.get("after_area"),
        "before_cavity": rec.get("before_cavity"),
        "after_cavity": rec.get("after_cavity"),
        "quality_before": rec.get("quality_before"),
        "quality_after": rec.get("quality_after"),
        "quality_delta": rec.get("quality_delta"),
        "quality_flags_before": rec.get("quality_flags_before"),
        "quality_flags_after": rec.get("quality_flags_after"),
        "quality_subscores_before": rec.get("quality_subscores_before"),
        "quality_subscores_after": rec.get("quality_subscores_after"),
        "topology_deltas": {
            "component_delta": component_delta,
            "hole_delta": hole_delta,
            "border_delta": border_delta,
            "centroid_delta": centroid_delta,
            "component_delta_available": component_delta is not None,
            "hole_delta_available": hole_delta is not None,
            "border_delta_available": border_delta is not None,
            "centroid_delta_available": centroid_delta is not None,
        },
        "unavailable_fields": unavailable,
        "guard": guard,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only CMR topology-delta shadow scan over the supplied volumes."""
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

    cases: Dict[str, Any] = {}
    guard_applicable = 0
    no_transition_reason_populated = 0
    delta_filled = {
        "component_delta": 0,
        "hole_delta": 0,
        "border_delta": 0,
        "centroid_delta": 0,
    }
    delta_unavailable = {k: 0 for k in delta_filled}

    for rec in records:
        cid = rec.get("case_id")
        scanned = _scan_case(rec, spatial_index.get(cid))
        if scanned["guard"]["guard_applicable"]:
            guard_applicable += 1
        if scanned["guard"].get("no_transition_reason"):
            no_transition_reason_populated += 1
        deltas = scanned["topology_deltas"]
        for key in delta_filled:
            if deltas.get(f"{key}_available"):
                delta_filled[key] += 1
            else:
                delta_unavailable[key] += 1
        cases[cid] = scanned

    guard_applicable_rate: Optional[float] = None
    no_transition_reason_populated_rate: Optional[float] = None
    if n:
        guard_applicable_rate = round(guard_applicable / n, 6)
        no_transition_reason_populated_rate = round(
            no_transition_reason_populated / n, 6
        )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_cmr_topology_delta_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are non-holdout pending candidates, not verified "
            "truth; topology deltas are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "component_before_fields": list(COMPONENT_BEFORE_FIELDS),
            "component_after_fields": list(COMPONENT_AFTER_FIELDS),
            "hole_before_fields": list(HOLE_BEFORE_FIELDS),
            "hole_after_fields": list(HOLE_AFTER_FIELDS),
            "border_before_fields": list(BORDER_BEFORE_FIELDS),
            "border_after_fields": list(BORDER_AFTER_FIELDS),
            "centroid_before_fields": list(CENTROID_BEFORE_FIELDS),
            "centroid_after_fields": list(CENTROID_AFTER_FIELDS),
            "topology_fields": list(TOPOLOGY_FIELDS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "underseg_area_ratio_max": UNDERSEG_AREA_RATIO_MAX,
            "fragmentation_flag": FRAGMENTATION_FLAG,
            "guard_applicable_min": GUARD_APPLICABLE_MIN,
            "no_transition_reason_min": NO_TRANSITION_REASON_MIN,
        },
        "n_records": n,
        "volumes_scanned": n,
        "expected_non_holdout": EXPECTED_NON_HOLDOUT,
        "non_holdout_denominator_matches_expected": n == EXPECTED_NON_HOLDOUT,
        "holdout_selected": 0,
        "holdout_excluded_count": selection["holdout_excluded_count"],
        "input_record_count": selection["input_count"],
        "duplicate_case_ids": selection["duplicate_case_ids"],
        "label_write": False,
        "target_path": "experiments/cmr_topology_delta_scan_exec_13.py",
        "topology_guard_applicable": guard_applicable,
        "topology_guard_applicable_rate": guard_applicable_rate,
        "topology_guard_applicable_rate_threshold": GUARD_APPLICABLE_MIN,
        "topology_guard_applicable_rate_meets_threshold": (
            guard_applicable_rate is not None
            and guard_applicable_rate >= GUARD_APPLICABLE_MIN
        ),
        "no_transition_reason_populated": no_transition_reason_populated,
        "no_transition_reason_populated_rate": no_transition_reason_populated_rate,
        "no_transition_reason_populated_rate_threshold": NO_TRANSITION_REASON_MIN,
        "no_transition_reason_populated_rate_meets_threshold": (
            no_transition_reason_populated_rate is not None
            and no_transition_reason_populated_rate >= NO_TRANSITION_REASON_MIN
        ),
        "topology_delta_filled": delta_filled,
        "topology_delta_unavailable": delta_unavailable,
        "spatial_gate": {
            "checked_rank_core_set_size": len(observed),
            "denominator": n,
        },
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
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
