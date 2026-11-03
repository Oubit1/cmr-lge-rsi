"""Read-only LGE myocardium topology scan over the 9 non-holdout volumes.

Scope (shadow_triage_only, zero production writes):
  * select the 9 non-holdout LGE myo volumes by excluding holdout_case_ids;
  * per volume emit connected-component count, secondary/primary area ratio,
    cross-slice continuity break count and border-contact proxy when the
    corresponding supplied fields exist;
  * report every missing field as explicit unavailable, never imputed;
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

MODULE_VERSION = "lge_myo_topology_readonly_scan9/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "non_holdout_pending_lge_myo_topology"

EXPECTED_NON_HOLDOUT = 9

# Supplied field aliases. The first present, non-None value wins; nothing is
# imputed. Missing fields are reported per-case as unavailable.
COMPONENT_FIELDS = (
    "myo_components",
    "component_count",
    "myo_component_count",
    "connected_components",
)

PRIMARY_AREA_FIELDS = (
    "primary_component_area",
    "myo_primary_area",
    "largest_component_area",
)

SECONDARY_AREA_FIELDS = (
    "secondary_component_area",
    "myo_secondary_area",
    "second_largest_component_area",
)

CONTINUITY_BREAK_FIELDS = (
    "cross_slice_continuity_breaks",
    "continuity_breaks",
    "slice_continuity_breaks",
)

BORDER_CONTACT_FIELDS = (
    "border_contact_proxy",
    "touches_border",
    "border_contact",
)

NO_WRITE_CLAUSE = (
    "Read-only LGE myocardium topology scan. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage "
    "only. app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/lge_myo_topology_readonly_scan9.py "
    "tests/test_lge_myo_topology_readonly_scan9.py"
)


def _first_present(rec: Dict[str, Any], fields: tuple) -> Any:
    """Return the first present, non-None value, else None."""
    for name in fields:
        if name in rec and rec[name] is not None:
            return rec[name]
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
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
    """Read-only topology scan for one volume; missing fields stay unavailable."""
    cid = rec.get("case_id")
    components = _as_int(_first_present(rec, COMPONENT_FIELDS))
    primary = _as_float(_first_present(rec, PRIMARY_AREA_FIELDS))
    secondary = _as_float(_first_present(rec, SECONDARY_AREA_FIELDS))
    breaks = _as_int(_first_present(rec, CONTINUITY_BREAK_FIELDS))
    border = _first_present(rec, BORDER_CONTACT_FIELDS)

    area_ratio: Optional[float] = None
    if primary is not None and secondary is not None and primary > 0.0:
        area_ratio = round(secondary / primary, 6)

    unavailable: List[str] = []
    if components is None:
        unavailable.append("component_count")
    if area_ratio is None:
        unavailable.append("secondary_primary_area_ratio")
    if breaks is None:
        unavailable.append("cross_slice_continuity_breaks")
    if border is None:
        unavailable.append("border_contact_proxy")

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
        "primary_component_area": primary,
        "secondary_component_area": secondary,
        "secondary_primary_area_ratio": area_ratio,
        "cross_slice_continuity_breaks": breaks,
        "border_contact_proxy": border,
        "component_count_available": components is not None,
        "area_ratio_available": area_ratio is not None,
        "continuity_breaks_available": breaks is not None,
        "border_contact_available": border is not None,
        "unavailable_fields": unavailable,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE myo topology scan over the supplied volumes."""
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
    area_ratio_filled = 0
    breaks_filled = 0
    border_filled = 0
    spatial_candidate_count = 0

    for rec in records:
        scanned = _scan_case(rec)
        cid = scanned["case_id"]
        if scanned["component_count_available"]:
            component_filled += 1
        if scanned["area_ratio_available"]:
            area_ratio_filled += 1
        if scanned["continuity_breaks_available"]:
            breaks_filled += 1
        if scanned["border_contact_available"]:
            border_filled += 1
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

    def _rate(filled: int) -> Optional[float]:
        return round(filled / n, 6) if n else None

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_lge_myo_topology_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are non-holdout pending candidates, not verified "
            "truth; topology fields are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "component_fields": list(COMPONENT_FIELDS),
            "primary_area_fields": list(PRIMARY_AREA_FIELDS),
            "secondary_area_fields": list(SECONDARY_AREA_FIELDS),
            "continuity_break_fields": list(CONTINUITY_BREAK_FIELDS),
            "border_contact_fields": list(BORDER_CONTACT_FIELDS),
            "topology_fields": list(TOPOLOGY_FIELDS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
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
        "target_path": "experiments/lge_myo_topology_readonly_scan9.py",
        "component_count_filled": component_filled,
        "area_ratio_filled": area_ratio_filled,
        "continuity_breaks_filled": breaks_filled,
        "border_contact_filled": border_filled,
        "component_count_fill_rate": _rate(component_filled),
        "area_ratio_fill_rate": _rate(area_ratio_filled),
        "continuity_breaks_fill_rate": _rate(breaks_filled),
        "border_contact_fill_rate": _rate(border_filled),
        "spatial_gate": {
            "checked_rank_core_set_size": len(observed),
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
