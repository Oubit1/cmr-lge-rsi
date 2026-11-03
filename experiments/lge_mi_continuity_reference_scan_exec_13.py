"""Read-only LGE MI cross-slice continuity reference-signal scan (13 non-holdout).

Scope (shadow_triage_only, zero production writes):
  * select the 13 non-holdout LGE volumes by excluding holdout_case_ids;
  * recompute cross_slice_continuity_agreement against a cross-slice
    reference signal (reference_area vs before_area ratio band) instead of
    the intensity proxy used by S839;
  * emit a per-case suppression reason for every case so
    no_reason_count == 0;
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

MODULE_VERSION = "lge_mi_continuity_reference_scan_exec_13/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "non_holdout_pending_lge_mi_continuity_reference"

EXPECTED_NON_HOLDOUT = 13

AGREEMENT_MIN = 0.8

# Cross-slice reference signal: the reference_area / before_area ratio band.
# A case is "continuous" when its ratio falls inside the band, i.e. the
# candidate volume is consistent with the cross-slice reference signal.
REFERENCE_RATIO_LOW = 0.5
REFERENCE_RATIO_HIGH = 2.0

# Suppression reasons (always populated, never empty).
REASON_CONTINUOUS = "continuous_with_reference_signal"
REASON_UNDERSEGMENTED = "undersegmented_vs_reference_signal"
REASON_OVERSEGMENTED = "oversegmented_vs_reference_signal"
REASON_NO_REFERENCE = "reference_signal_unavailable"
REASON_NO_AREA = "candidate_area_unavailable"

NO_WRITE_CLAUSE = (
    "Read-only LGE MI cross-slice continuity reference-signal scan. No "
    "production labels, code, datasets or runtime databases are written or "
    "mutated. Shadow triage only. app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/lge_mi_continuity_reference_scan_exec_13.py "
    "tests/test_lge_mi_continuity_reference_scan_exec_13.py"
)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
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


def _reference_ratio(rec: Dict[str, Any]) -> Optional[float]:
    """Cross-slice reference signal: reference_area / before_area."""
    ref = _as_float(rec.get("reference_area"))
    before = _as_float(rec.get("before_area"))
    if ref is None or before is None or before <= 0.0:
        return None
    return round(ref / before, 6)


def _scan_case(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute continuity agreement for one volume vs the reference signal."""
    cid = rec.get("case_id")
    ratio = _reference_ratio(rec)
    area_ratio = _area_ratio(rec)

    if ratio is None:
        continuous = False
        reason = REASON_NO_REFERENCE
    elif ratio < REFERENCE_RATIO_LOW:
        continuous = False
        reason = REASON_OVERSEGMENTED
    elif ratio > REFERENCE_RATIO_HIGH:
        continuous = False
        reason = REASON_UNDERSEGMENTED
    else:
        continuous = True
        reason = REASON_CONTINUOUS

    return {
        "case_id": cid,
        "reference_area": rec.get("reference_area"),
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
        "area_ratio": area_ratio,
        "reference_ratio": ratio,
        "continuous_with_reference": continuous,
        "suppression_reason": reason,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE MI continuity reference-signal scan."""
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
    continuous_count = 0
    no_reason_count = 0
    reason_counts: Dict[str, int] = {}
    spatial_candidate_count = 0

    for rec in records:
        scanned = _scan_case(rec)
        cid = scanned["case_id"]
        if scanned["continuous_with_reference"]:
            continuous_count += 1
        reason = scanned.get("suppression_reason")
        if not reason:
            no_reason_count += 1
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
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

    agreement: Optional[float] = None
    if n:
        agreement = round(continuous_count / n, 6)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_lge_mi_continuity_reference_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are non-holdout pending candidates, not verified "
            "truth; continuity agreement is measured against a cross-slice "
            "reference signal, and concordance, precision and CI against "
            "independent labels stay unavailable."
        ),
        "rule": {
            "reference_ratio_low": REFERENCE_RATIO_LOW,
            "reference_ratio_high": REFERENCE_RATIO_HIGH,
            "topology_fields": list(TOPOLOGY_FIELDS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
            "agreement_min": AGREEMENT_MIN,
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
        "target_path": "experiments/lge_mi_continuity_reference_scan_exec_13.py",
        "continuous_count": continuous_count,
        "cross_slice_continuity_agreement": agreement,
        "cross_slice_continuity_agreement_threshold": AGREEMENT_MIN,
        "cross_slice_continuity_agreement_meets_threshold": (
            agreement is not None and agreement >= AGREEMENT_MIN
        ),
        "no_reason_count": no_reason_count,
        "suppression_reason_counts": reason_counts,
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
