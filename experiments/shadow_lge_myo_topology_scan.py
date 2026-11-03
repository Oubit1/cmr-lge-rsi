"""Read-only shadow scanner: per-case LGE myocardium topology for the
non-holdout pending CMR volumes, selected explicitly by case_id.

Scope (shadow_triage_only, zero production writes):
  * select exactly the non-holdout pending volumes by case_id, excluding the
    frozen holdout (holdout_selected==0) and deduplicating case_ids;
  * reuse the existing area/cavity subscore helpers, S783 cluster classifier,
    S825 secondary-component fields and the anonymous spatial rank/core gate
    (no second harness);
  * per case emit component_count, hole_count, border_contact and a
    cross-slice continuity proxy, each with a populated reason;
  * reconcile the cross-slice continuity proxy against the mask-derived
    layer-jump measurement per case and record the difference;
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
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "shadow_lge_myo_topology_scan/1.1"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_NON_HOLDOUT = 13

FILL_RATE_MIN = 0.9

# Cross-slice continuity proxy: a volume is treated as continuous when the
# supplied topology flags do not report a layer jump / fragmentation signal.
CONTINUITY_FLAG = "layer_jump"

# Mask-derived layer-jump measurement keys, when supplied by the caller.
MASK_LAYER_JUMP_KEYS = (
    "mask_layer_jump",
    "mask_layer_jump_count",
    "layer_jump_measurement",
)

# Border-contact proxy: reuse the existing border-contact review flag.
BORDER_CONTACT_FLAG = "border_contact_review"

# Contrast-support proxy: reuse the existing image_support subscore as the
# contrast-support signal when supplied; never imputed.
CONTRAST_SUPPORT_SUBSCORE = "image_support"

NO_WRITE_CLAUSE = (
    "Read-only shadow LGE myo topology scan. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_lge_myo_topology_scan.py "
    "tests/test_shadow_lge_myo_topology_scan.py"
)


def _select_non_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Select non-holdout volumes by case_id, deduplicating case_ids."""
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


def _component_count(rec: Dict[str, Any]) -> Optional[int]:
    """Connected-component count for the LGE myo volume, when supplied."""
    for key in ("component_count", "components", "n_components"):
        val = rec.get(key)
        if isinstance(val, int):
            return val
    secondary = _secondary_component_fields(rec)
    sec_count = secondary.get("secondary_component_count")
    if isinstance(sec_count, int):
        return sec_count + 1
    return None


def _hole_count(rec: Dict[str, Any]) -> Optional[int]:
    """Hole count for the LGE myo volume, when supplied; never imputed."""
    for key in ("hole_count", "holes", "n_holes"):
        val = rec.get(key)
        if isinstance(val, int):
            return val
    return None


def _cross_slice_continuity(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-slice continuity proxy from supplied topology flags."""
    flags = rec.get("quality_flags_before") or []
    if not isinstance(flags, list):
        flags = []
    if CONTINUITY_FLAG in flags:
        return {
            "continuous": False,
            "reason": "layer_jump_flag_present",
            "source": "quality_flags_before",
        }
    return {
        "continuous": True,
        "reason": "no_layer_jump_flag",
        "source": "quality_flags_before",
    }


def _mask_layer_jump(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Mask-derived layer-jump measurement, when supplied; never imputed."""
    for key in MASK_LAYER_JUMP_KEYS:
        val = rec.get(key)
        if isinstance(val, bool):
            return {
                "available": True,
                "layer_jump": val,
                "source": key,
                "reason": "mask_layer_jump_supplied",
            }
        if isinstance(val, (int, float)):
            return {
                "available": True,
                "layer_jump": float(val) > 0.0,
                "layer_jump_count": float(val),
                "source": key,
                "reason": "mask_layer_jump_supplied",
            }
    return {
        "available": False,
        "layer_jump": None,
        "source": None,
        "reason": "missing_mask_layer_jump",
    }


def _reconcile_continuity(
    proxy: Dict[str, Any], mask: Dict[str, Any]
) -> Dict[str, Any]:
    """Reconcile the continuity proxy against the mask-derived layer jump.

    ``proxy_continuous`` is True when the proxy reports no layer jump.
    ``mask_continuous`` is True when the mask measurement reports no layer
    jump. ``agrees`` is True when both agree; ``difference`` records the
    per-case disagreement (proxy_continuous XOR mask_continuous) so the
    reconciliation is auditable even when the mask measurement is missing.
    """
    proxy_continuous = proxy.get("continuous")
    if not mask.get("available"):
        return {
            "proxy_continuous": proxy_continuous,
            "mask_continuous": None,
            "agrees": None,
            "difference": None,
            "status": "unavailable_no_mask_layer_jump",
            "reason": mask.get("reason", "missing_mask_layer_jump"),
        }
    mask_continuous = not bool(mask.get("layer_jump"))
    agrees = bool(proxy_continuous) == mask_continuous
    return {
        "proxy_continuous": proxy_continuous,
        "mask_continuous": mask_continuous,
        "agrees": agrees,
        "difference": 0 if agrees else 1,
        "status": "reconciled",
        "reason": "proxy_matches_mask" if agrees else "proxy_mask_disagreement",
    }


def _border_contact(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Border-contact proxy from supplied quality flags."""
    flags = rec.get("quality_flags_before") or []
    if not isinstance(flags, list):
        flags = []
    touches = BORDER_CONTACT_FLAG in flags or "border_clip_candidate" in flags
    return {
        "touches_border": touches,
        "reason": "border_contact_flag_present" if touches else "no_border_contact_flag",
        "source": "quality_flags_before",
    }


def _contrast_support(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Contrast-support proxy from the supplied image_support subscore."""
    subs = rec.get("quality_subscores_before") or {}
    if not isinstance(subs, dict):
        subs = {}
    val = subs.get(CONTRAST_SUPPORT_SUBSCORE)
    if isinstance(val, (int, float)):
        return {
            "supported": float(val) > 0.0,
            "image_support": float(val),
            "reason": "image_support_subscore_supplied",
        }
    return {
        "supported": None,
        "image_support": None,
        "reason": "missing_measurement",
    }


def _scan_case(
    rec: Dict[str, Any],
    spatial_index: Dict[Any, Dict[str, Any]],
    observed: set,
) -> Dict[str, Any]:
    """Per-case LGE myo topology scan with a populated reason for every case."""
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    cavity = rec.get("before_cavity")
    cluster = _classify(area_ratio, cavity)
    topology = _topology_flags(rec)
    secondary = _secondary_component_fields(rec)
    spatial_out = _spatial_decision(cid, spatial_index, observed)
    spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION

    component_count = _component_count(rec)
    hole_count = _hole_count(rec)
    continuity = _cross_slice_continuity(rec)
    mask_jump = _mask_layer_jump(rec)
    reconciliation = _reconcile_continuity(continuity, mask_jump)
    border = _border_contact(rec)
    contrast = _contrast_support(rec)

    if area_ratio is None or cavity is None:
        reason = "missing_measurement"
    elif _is_unrouted(cluster, cavity):
        reason = "unrouted"
    elif spatial_hit:
        reason = "spatial_gate_suppressed"
    elif component_count is None:
        reason = "missing_component_count"
    elif hole_count is None:
        reason = "missing_hole_count"
    elif not continuity["continuous"]:
        reason = "cross_slice_discontinuity"
    elif border["touches_border"]:
        reason = "border_contact_candidate"
    elif contrast["supported"] is False:
        reason = "contrast_support_absent"
    else:
        reason = ROUTED_REASON

    suppression_candidate = reason != ROUTED_REASON

    return {
        "case_id": cid,
        "area_ratio": area_ratio,
        "cavity": cavity,
        "cluster": cluster,
        "topology": topology,
        "secondary_components": secondary,
        "spatial_rule": spatial_out,
        "spatial_gate_suppressed": spatial_hit,
        "component_count": component_count,
        "hole_count": hole_count,
        "cross_slice_continuity": continuity,
        "mask_layer_jump": mask_jump,
        "continuity_reconciliation": reconciliation,
        "border_contact": border,
        "contrast_support": contrast,
        "suppression_candidate": suppression_candidate,
        "suppression_reason": reason,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE myo topology scan over the supplied pending volumes."""
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

    component_count_distribution: Dict[str, int] = {}
    hole_count_distribution: Dict[str, int] = {}
    component_count_filled = 0
    hole_count_filled = 0
    continuity_filled = 0
    border_filled = 0
    contrast_filled = 0
    mask_jump_filled = 0
    reconciliation_filled = 0
    reconciliation_agreements = 0
    reconciliation_disagreements = 0
    reconciliation_differences: List[Dict[str, Any]] = []
    suppression_reasons: Dict[str, int] = {ROUTED_REASON: 0}
    suppression_candidates: List[Dict[str, Any]] = []
    no_reason_count = 0
    cases: Dict[str, Any] = {}

    for rec in records:
        scanned = _scan_case(rec, spatial_index, observed)
        cid = scanned["case_id"]

        cc = scanned["component_count"]
        if cc is not None:
            component_count_filled += 1
            key = str(cc)
            component_count_distribution[key] = component_count_distribution.get(key, 0) + 1
        hc = scanned["hole_count"]
        if hc is not None:
            hole_count_filled += 1
            key = str(hc)
            hole_count_distribution[key] = hole_count_distribution.get(key, 0) + 1
        if scanned["cross_slice_continuity"].get("continuous") is not None:
            continuity_filled += 1
        if scanned["border_contact"].get("touches_border") is not None:
            border_filled += 1
        if scanned["contrast_support"].get("supported") is not None:
            contrast_filled += 1
        if scanned["mask_layer_jump"].get("available"):
            mask_jump_filled += 1
        recon = scanned["continuity_reconciliation"]
        if recon.get("status") == "reconciled":
            reconciliation_filled += 1
            if recon.get("agrees"):
                reconciliation_agreements += 1
            else:
                reconciliation_disagreements += 1
                reconciliation_differences.append({
                    "case_id": cid,
                    "proxy_continuous": recon.get("proxy_continuous"),
                    "mask_continuous": recon.get("mask_continuous"),
                    "difference": recon.get("difference"),
                    "reason": recon.get("reason"),
                })

        reason = scanned["suppression_reason"]
        suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1
        if scanned["suppression_candidate"]:
            if not reason:
                no_reason_count += 1
            suppression_candidates.append({
                "case_id": cid,
                "reason": reason,
                "component_count": cc,
                "hole_count": hc,
            })

        cases[cid] = scanned

    def _rate(filled: int) -> Optional[float]:
        if not n:
            return None
        return round(filled / n, 6)

    component_count_fill_rate = _rate(component_count_filled)
    hole_count_fill_rate = _rate(hole_count_filled)
    continuity_fill_rate = _rate(continuity_filled)
    border_fill_rate = _rate(border_filled)
    contrast_fill_rate = _rate(contrast_filled)
    mask_jump_fill_rate = _rate(mask_jump_filled)
    reconciliation_fill_rate = _rate(reconciliation_filled)
    reconciliation_agreement_rate: Optional[float] = None
    if reconciliation_filled:
        reconciliation_agreement_rate = round(
            reconciliation_agreements / reconciliation_filled, 6
        )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_lge_myo_topology_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "per-case topology is monitoring-grade and concordance, precision "
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
            "topology_fields": list(TOPOLOGY_FIELDS),
            "continuity_flag": CONTINUITY_FLAG,
            "border_contact_flag": BORDER_CONTACT_FLAG,
            "contrast_support_subscore": CONTRAST_SUPPORT_SUBSCORE,
            "mask_layer_jump_keys": list(MASK_LAYER_JUMP_KEYS),
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
        "target_path": "experiments/shadow_lge_myo_topology_scan.py",
        "component_count_distribution": component_count_distribution,
        "hole_count_distribution": hole_count_distribution,
        "component_count_filled": component_count_filled,
        "component_count_fill_rate": component_count_fill_rate,
        "hole_count_filled": hole_count_filled,
        "hole_count_fill_rate": hole_count_fill_rate,
        "cross_slice_continuity_filled": continuity_filled,
        "cross_slice_continuity_fill_rate": continuity_fill_rate,
        "border_contact_filled": border_filled,
        "border_contact_fill_rate": border_fill_rate,
        "contrast_support_filled": contrast_filled,
        "contrast_support_fill_rate": contrast_fill_rate,
        "mask_layer_jump_filled": mask_jump_filled,
        "mask_layer_jump_fill_rate": mask_jump_fill_rate,
        "continuity_reconciliation_filled": reconciliation_filled,
        "continuity_reconciliation_fill_rate": reconciliation_fill_rate,
        "continuity_reconciliation_agreements": reconciliation_agreements,
        "continuity_reconciliation_disagreements": reconciliation_disagreements,
        "continuity_reconciliation_agreement_rate": reconciliation_agreement_rate,
        "continuity_reconciliation_differences": reconciliation_differences,
        "fill_rate_threshold": FILL_RATE_MIN,
        "component_count_fill_rate_meets_threshold": (
            component_count_fill_rate is not None
            and component_count_fill_rate > FILL_RATE_MIN
        ),
        "hole_count_fill_rate_meets_threshold": (
            hole_count_fill_rate is not None
            and hole_count_fill_rate > FILL_RATE_MIN
        ),
        "cross_slice_continuity_fill_rate_meets_threshold": (
            continuity_fill_rate is not None
            and continuity_fill_rate > FILL_RATE_MIN
        ),
        "suppression_candidate_count": len(suppression_candidates),
        "suppression_candidates": suppression_candidates,
        "suppression_reasons": suppression_reasons,
        "no_reason_count": no_reason_count,
        "no_reason_count_is_zero": no_reason_count == 0,
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
