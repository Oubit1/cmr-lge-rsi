"""Read-only LGE geometry supply scan over the 13 non-holdout volumes.

Scope (shadow_triage_only, zero production writes):
  * select the 13 non-holdout LGE volumes by excluding holdout_case_ids;
  * fill component_count and hole_count per volume from supplied geometry
    fields (myo_components / mi_components / myo_holes / mi_holes and their
    *_count aliases); missing fields are reported unavailable, never imputed;
  * emit component_count_fill_rate and hole_count_fill_rate plus a per-case
    unavailable-field list and a geometry_supplied flag;
  * report concordance, precision and CI as unavailable (no independent
    reference labels supplied);
  * emit module hash, rollback command and a zero-write audit diff.

Read-only: writes no production labels, code, datasets or runtime databases.
Does not touch app/check_auto_apply.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from experiments.cmr_checked_cluster_replay import audit_diff, module_hash
from experiments.cmr_low_cavity_discriminator_scan import (
    COHORT_LABEL as S784_COHORT_LABEL,
    EXPECTED_PENDING as S784_EXPECTED_PENDING,
    NO_TRANSITION_BASELINE_S783,
    TOPOLOGY_FIELDS,
    _area_ratio,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "lge_geometry_supply_scan_exec_13/1.1"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "non_holdout_pending_lge_geometry_supply"

EXPECTED_NON_HOLDOUT = 13

FILL_RATE_MIN = 0.9

# Supplied geometry field aliases for the two topology counts. The first
# present, non-None value wins; nothing is imputed.
COMPONENT_FIELDS = (
    "myo_components",
    "mi_components",
    "component_count",
    "myo_component_count",
    "mi_component_count",
)

HOLE_FIELDS = (
    "myo_holes",
    "mi_holes",
    "hole_count",
    "myo_hole_count",
    "mi_hole_count",
)

NO_WRITE_CLAUSE = (
    "Read-only LGE geometry supply scan. No production labels, code, datasets "
    "or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/lge_geometry_supply_scan_exec_13.py "
    "tests/test_lge_geometry_supply_scan_exec_13.py"
)


def _first_present(rec: Dict[str, Any], fields: tuple) -> Optional[int]:
    """Return the first present, non-None integer-valued field, else None."""
    for name in fields:
        if name in rec and rec[name] is not None:
            value = rec[name]
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
    return None


def _select_non_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select non-holdout volumes by case_id, deduplicating."""
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


def _scan_case(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill component/hole counts for one volume from supplied geometry."""
    cid = rec.get("case_id")
    components = _first_present(rec, COMPONENT_FIELDS)
    holes = _first_present(rec, HOLE_FIELDS)
    unavailable: List[str] = []
    if components is None:
        unavailable.append("component_count")
    if holes is None:
        unavailable.append("hole_count")
    geometry_supplied = components is not None or holes is not None
    return {
        "case_id": cid,
        "area_ratio": _area_ratio(rec),
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
        "component_count": components,
        "hole_count": holes,
        "component_count_available": components is not None,
        "hole_count_available": holes is not None,
        "geometry_supplied": geometry_supplied,
        "unavailable_fields": unavailable,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE geometry supply scan over the supplied volumes."""
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
    component_filled = 0
    hole_filled = 0
    geometry_supplied_count = 0
    unavailable_fields: Dict[str, int] = {
        "component_count": 0,
        "hole_count": 0,
    }
    spatial_candidate_count = 0

    for rec in records:
        scanned = _scan_case(rec)
        cid = scanned["case_id"]
        if scanned["component_count_available"]:
            component_filled += 1
        else:
            unavailable_fields["component_count"] += 1
        if scanned["hole_count_available"]:
            hole_filled += 1
        else:
            unavailable_fields["hole_count"] += 1
        if scanned["geometry_supplied"]:
            geometry_supplied_count += 1
        spatial_rec = spatial_index.get(cid)
        if isinstance(spatial_rec, dict) and spatial_rec.get("decision") == CANDIDATE_DECISION:
            spatial_candidate_count += 1
            scanned["spatial_decision"] = CANDIDATE_DECISION
            scanned["spatial_reason"] = spatial_rec.get("reason", OUTSIDE_REASON)
        else:
            scanned["spatial_decision"] = (
                spatial_rec.get("decision") if isinstance(spatial_rec, dict) else None
            )
            scanned["spatial_reason"] = (
                spatial_rec.get("reason") if isinstance(spatial_rec, dict) else None
            )
        cases[cid] = scanned

    component_fill_rate: Optional[float] = None
    hole_fill_rate: Optional[float] = None
    if n:
        component_fill_rate = round(component_filled / n, 6)
        hole_fill_rate = round(hole_filled / n, 6)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_lge_geometry_supply_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are non-holdout pending candidates, not verified "
            "truth; geometry fill rates are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "component_fields": list(COMPONENT_FIELDS),
            "hole_fields": list(HOLE_FIELDS),
            "topology_fields": list(TOPOLOGY_FIELDS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "fill_rate_threshold": FILL_RATE_MIN,
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
        "target_path": "experiments/lge_geometry_supply_scan_exec_13.py",
        "geometry_supplied_count": geometry_supplied_count,
        "geometry_supplied": geometry_supplied_count > 0,
        "component_count_filled": component_filled,
        "hole_count_filled": hole_filled,
        "component_count_fill_rate": component_fill_rate,
        "hole_count_fill_rate": hole_fill_rate,
        "component_count_fill_rate_threshold": FILL_RATE_MIN,
        "hole_count_fill_rate_threshold": FILL_RATE_MIN,
        "component_count_fill_rate_meets_threshold": (
            component_fill_rate is not None and component_fill_rate > FILL_RATE_MIN
        ),
        "hole_count_fill_rate_meets_threshold": (
            hole_fill_rate is not None and hole_fill_rate > FILL_RATE_MIN
        ),
        "unavailable_field_counts": unavailable_fields,
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "denominator": n,
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
