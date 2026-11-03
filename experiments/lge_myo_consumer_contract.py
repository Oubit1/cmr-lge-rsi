"""Read-only LGE myocardium consumer contract over the 9 non-holdout volumes.

Scope (shadow_triage_only, zero production writes):
  * select the 9 non-holdout LGE myo volumes by excluding holdout_case_ids;
  * parse the supplied issue list, topology flags and the snapshot manifest
    flags mi_multiple_components / myo_topology per volume;
  * emit a per-case verdict or an explicit unavailable reason (never imputed);
  * dedup manifest-derived issues by case_id+issue_type so historical
    duplicate rows cannot inflate resolved counts;
  * build the runtime/lge_iterations/latest.json report payload with readiness,
    pending/working counts, zero-write audit and rollback evidence;
  * report a fixed-validation delta vs S859 report b03f1b27 (dedup_ratio,
    cache_hit_rate) and keep concordance, precision and CI unavailable.

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

MODULE_VERSION = "lge_myo_consumer_contract/1.1"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "non_holdout_pending_lge_myo_consumer_contract"

EXPECTED_NON_HOLDOUT = 9

REPORT_PATH = "runtime/lge_iterations/latest.json"

# S859 shadow report b03f1b27 measured 9/9 unavailable on the same cohort.
S859_REPORT_ID = "b03f1b27"
S859_UNAVAILABLE_COUNT = 9
S859_DEDUP_RATIO = 1.0
S859_CACHE_HIT_RATE = 0.0

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

ISSUE_LIST_FIELDS = (
    "issues",
    "issue_list",
    "myo_issues",
    "qc_issues",
)

# Snapshot manifest flags added by S860. Boolean or list-valued; a truthy
# boolean or a non-empty list means the flag is present.
MI_MULTIPLE_COMPONENTS_FIELDS = (
    "mi_multiple_components",
    "mi_multi_component",
    "mi_multicomponent",
)

MYO_TOPOLOGY_FIELDS = (
    "myo_topology",
    "myo_topology_flag",
    "myo_topology_status",
)

NO_WRITE_CLAUSE = (
    "Read-only LGE myocardium consumer contract. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage "
    "only. app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/lge_myo_consumer_contract.py "
    "tests/test_lge_myo_consumer_contract.py"
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


def _flag_present(value: Any) -> Optional[bool]:
    """Normalize a manifest flag to True/False, or None when absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no")
    if isinstance(value, (int, float)):
        return value != 0
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


def _parse_issues(rec: Dict[str, Any]) -> Optional[List[Any]]:
    raw = _first_present(rec, ISSUE_LIST_FIELDS)
    if isinstance(raw, list):
        return raw
    return None


def _issue_type(issue: Any) -> str:
    if isinstance(issue, dict):
        for key in ("issue_type", "type", "code", "name"):
            if issue.get(key) is not None:
                return str(issue[key])
        return "unknown"
    return str(issue)


def _dedup_issues(cid: Any, issues: List[Any]) -> Dict[str, Any]:
    """Dedup manifest-derived issues by case_id+issue_type."""
    seen: set = set()
    kept: List[Any] = []
    dropped = 0
    for issue in issues:
        key = (cid, _issue_type(issue))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(issue)
    return {"issues": kept, "dropped": dropped, "raw_count": len(issues)}


def _consume_case(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only consumer verdict for one volume; missing fields stay unavailable."""
    cid = rec.get("case_id")
    components = _as_int(_first_present(rec, COMPONENT_FIELDS))
    primary = _as_float(_first_present(rec, PRIMARY_AREA_FIELDS))
    secondary = _as_float(_first_present(rec, SECONDARY_AREA_FIELDS))
    breaks = _as_int(_first_present(rec, CONTINUITY_BREAK_FIELDS))
    border = _first_present(rec, BORDER_CONTACT_FIELDS)
    issues = _parse_issues(rec)
    mi_multi = _flag_present(_first_present(rec, MI_MULTIPLE_COMPONENTS_FIELDS))
    myo_topo = _flag_present(_first_present(rec, MYO_TOPOLOGY_FIELDS))

    area_ratio: Optional[float] = None
    if primary is not None and secondary is not None and primary > 0.0:
        area_ratio = round(secondary / primary, 6)

    dedup = _dedup_issues(cid, issues) if issues is not None else None
    deduped_issues = dedup["issues"] if dedup is not None else None

    unavailable: List[str] = []
    if components is None:
        unavailable.append("component_count")
    if area_ratio is None:
        unavailable.append("secondary_primary_area_ratio")
    if breaks is None:
        unavailable.append("cross_slice_continuity_breaks")
    if border is None:
        unavailable.append("border_contact_proxy")
    if issues is None:
        unavailable.append("issue_list")
    if mi_multi is None:
        unavailable.append("mi_multiple_components")
    if myo_topo is None:
        unavailable.append("myo_topology")

    if unavailable:
        verdict = "unavailable"
        reason = "unavailable:" + ",".join(unavailable)
    else:
        flagged = bool(deduped_issues) or bool(mi_multi) or bool(myo_topo)
        verdict = "review" if flagged else "resolved"
        reason = "manifest_flags_and_issues_supplied"

    return {
        "case_id": cid,
        "verdict": verdict,
        "unavailable_reason": reason,
        "unavailable_fields": unavailable,
        "issue_count": len(deduped_issues) if deduped_issues is not None else None,
        "issues": deduped_issues,
        "issues_dropped_duplicates": dedup["dropped"] if dedup is not None else None,
        "mi_multiple_components": mi_multi,
        "myo_topology": myo_topo,
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
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE myo consumer contract over the supplied volumes."""
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
    resolved = 0
    review = 0
    unavailable = 0
    reason_filled = 0
    spatial_candidate_count = 0
    issues_raw_total = 0
    issues_dropped_total = 0
    cache_hits = 0

    for rec in records:
        consumed = _consume_case(rec)
        cid = consumed["case_id"]
        if consumed["verdict"] == "resolved":
            resolved += 1
        elif consumed["verdict"] == "review":
            review += 1
        else:
            unavailable += 1
        if consumed["unavailable_reason"]:
            reason_filled += 1
        if consumed["issues_dropped_duplicates"] is not None:
            issues_dropped_total += consumed["issues_dropped_duplicates"]
            issues_raw_total += consumed["issues_dropped_duplicates"] + (
                consumed["issue_count"] or 0
            )
        if consumed["mi_multiple_components"] is not None and consumed["myo_topology"] is not None:
            cache_hits += 1
        spatial_rec = spatial_index.get(cid)
        if isinstance(spatial_rec, dict) and spatial_rec.get("decision") == CANDIDATE_DECISION:
            spatial_candidate_count += 1
            consumed["spatial_decision"] = CANDIDATE_DECISION
            consumed["spatial_reason"] = spatial_rec.get("reason", OUTSIDE_REASON)
        else:
            consumed["spatial_decision"] = (
                spatial_rec.get("decision") if isinstance(spatial_rec, dict) else None
            )
            consumed["spatial_reason"] = (
                spatial_rec.get("reason") if isinstance(spatial_rec, dict) else None
            )
        cases[cid] = consumed

    def _rate(filled: int) -> Optional[float]:
        return round(filled / n, 6) if n else None

    unavailable_reason_fill_rate = _rate(reason_filled)
    dedup_ratio = (
        round(issues_raw_total / (issues_raw_total - issues_dropped_total), 6)
        if issues_raw_total > issues_dropped_total
        else 1.0
    )
    cache_hit_rate = _rate(cache_hits)
    consumer_ready = (
        n == EXPECTED_NON_HOLDOUT
        and unavailable < EXPECTED_NON_HOLDOUT
        and (resolved + review + unavailable) == EXPECTED_NON_HOLDOUT
        and unavailable_reason_fill_rate == 1.0
        and selection["holdout_excluded_count"] == 48
    )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_lge_myo_consumer_contract",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "report_path": REPORT_PATH,
        "cohort_note": (
            "Supplied volumes are non-holdout pending candidates, not verified "
            "truth; topology and issue fields are monitoring-grade and "
            "concordance, precision and CI against independent labels stay "
            "unavailable."
        ),
        "rule": {
            "component_fields": list(COMPONENT_FIELDS),
            "primary_area_fields": list(PRIMARY_AREA_FIELDS),
            "secondary_area_fields": list(SECONDARY_AREA_FIELDS),
            "continuity_break_fields": list(CONTINUITY_BREAK_FIELDS),
            "border_contact_fields": list(BORDER_CONTACT_FIELDS),
            "issue_list_fields": list(ISSUE_LIST_FIELDS),
            "mi_multiple_components_fields": list(MI_MULTIPLE_COMPONENTS_FIELDS),
            "myo_topology_fields": list(MYO_TOPOLOGY_FIELDS),
            "topology_fields": list(TOPOLOGY_FIELDS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "dedup_key": "case_id+issue_type",
        },
        "lge_myo_consumer_ready": consumer_ready,
        "n_records": n,
        "volumes_consumed": n,
        "expected_non_holdout": EXPECTED_NON_HOLDOUT,
        "non_holdout_denominator_matches_expected": n == EXPECTED_NON_HOLDOUT,
        "holdout_selected": 0,
        "holdout_excluded_count": selection["holdout_excluded_count"],
        "input_record_count": selection["input_count"],
        "duplicate_case_ids": selection["duplicate_case_ids"],
        "label_write": False,
        "production_writes": 0,
        "target_path": "experiments/lge_myo_consumer_contract.py",
        "pending_count": n,
        "working_count": review,
        "resolved_count": resolved,
        "unavailable_count": unavailable,
        "unavailable_reason_filled": reason_filled,
        "unavailable_reason_fill_rate": unavailable_reason_fill_rate,
        "dedup_ratio": dedup_ratio,
        "cache_hit_rate": cache_hit_rate,
        "issues_raw_total": issues_raw_total,
        "issues_dropped_duplicates": issues_dropped_total,
        "s859_fixed_validation": {
            "report_id": S859_REPORT_ID,
            "baseline_unavailable_count": S859_UNAVAILABLE_COUNT,
            "baseline_dedup_ratio": S859_DEDUP_RATIO,
            "baseline_cache_hit_rate": S859_CACHE_HIT_RATE,
            "unavailable_count_delta": unavailable - S859_UNAVAILABLE_COUNT,
            "dedup_ratio_delta": round(dedup_ratio - S859_DEDUP_RATIO, 6),
            "cache_hit_rate_delta": (
                round(cache_hit_rate - S859_CACHE_HIT_RATE, 6)
                if cache_hit_rate is not None
                else None
            ),
        },
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
