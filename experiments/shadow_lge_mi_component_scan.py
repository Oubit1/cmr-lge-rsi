"""Read-only shadow scanner: per-case LGE MI component topology for the
non-holdout pending CMR volumes, selected explicitly by case_id.

Scope (shadow_triage_only, zero production writes):
  * select exactly the non-holdout pending volumes by case_id, excluding the
    frozen holdout (holdout_selected==0) and deduplicating case_ids;
  * reuse the existing area/cavity subscore helpers, S783 cluster classifier,
    S825 secondary-component fields and the anonymous spatial rank/core gate
    (no second harness);
  * per case emit component_count_distribution, secondary_to_primary_area_ratio,
    cross-slice continuity proxy, border-contact proxy and contrast-support
    proxy, each with a populated reason for every suppression candidate;
  * add an intensity-proxy cross-slice continuity / border-contact estimator
    derived from the supplied topology subscore and quality flags (no masks, no
    imputation) and report cross_slice_continuity_agreement against the
    existing flag-based continuity signal;
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

MODULE_VERSION = "shadow_lge_mi_component_scan/1.2"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_NON_HOLDOUT = 13

FILL_RATE_MIN = 0.9

# Cross-slice continuity proxy: a component is treated as continuous when the
# supplied topology flags do not report a layer jump / fragmentation signal.
CONTINUITY_FLAG = "layer_jump"

# Border-contact proxy: reuse the existing border-contact review flag.
BORDER_CONTACT_FLAG = "border_contact_review"

# Contrast-support proxy: reuse the existing image_support subscore as the
# contrast-support signal when supplied; never imputed.
CONTRAST_SUPPORT_SUBSCORE = "image_support"

# Intensity-proxy continuity: the supplied topology subscore is the only
# intensity-derived signal available without masks. A low topology subscore is
# treated as an intensity-proxy discontinuity signal; the floor is the same
# 80.0 baseline used by the existing topology subscore reporting.
TOPOLOGY_SUBSCORE = "topology"

INTENSITY_PROXY_TOPOLOGY_FLOOR = 80.0

# Acceptance threshold for the intensity-proxy vs flag-based continuity
# agreement over the non-holdout cohort.
CONTINUITY_AGREEMENT_MIN = 0.8

# Secondary-component area field aliases accepted when deriving the
# secondary/primary area ratio directly from the record.
SECONDARY_AREA_KEYS = (
    "secondary_component_area",
    "secondary_area",
    "secondary_comp_area",
)

NO_WRITE_CLAUSE = (
    "Read-only shadow LGE MI component scan. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/shadow_lge_mi_component_scan.py "
    "tests/test_shadow_lge_mi_component_scan.py"
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
    """Connected-component count for the LGE MI volume, when supplied.

    Explicit component fields win; otherwise the count is derived from the
    existing S825 secondary-component fields (secondary count + primary).
    """
    for key in ("component_count", "components", "n_components"):
        val = rec.get(key)
        if isinstance(val, int):
            return val
    secondary = _secondary_component_fields(rec)
    sec_count = secondary.get("secondary_component_count")
    if isinstance(sec_count, int):
        return sec_count + 1
    return None


def _secondary_area(rec: Dict[str, Any]) -> Optional[float]:
    """Secondary-component area, when supplied; never imputed.

    Explicit record fields win; otherwise the S825 secondary-component helper
    is consulted. Returns None when no measurement is available.
    """
    for key in SECONDARY_AREA_KEYS:
        val = rec.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    secondary = _secondary_component_fields(rec)
    sec_area = secondary.get("secondary_component_area")
    if isinstance(sec_area, (int, float)):
        return float(sec_area)
    return None


def _secondary_to_primary_ratio(rec: Dict[str, Any]) -> Optional[float]:
    """Secondary/primary area ratio, when supplied; never imputed.

    Explicit ratio fields win; otherwise the ratio is derived from the
    secondary-component area and the supplied before_area.
    """
    for key in (
        "secondary_to_primary_area_ratio",
        "secondary_primary_area_ratio",
    ):
        val = rec.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    sec_area = _secondary_area(rec)
    prim_area = rec.get("before_area")
    if isinstance(sec_area, (int, float)) and isinstance(prim_area, (int, float)) and prim_area:
        return round(float(sec_area) / float(prim_area), 6)
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


def _intensity_proxy_continuity(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Intensity-proxy cross-slice continuity from the supplied topology subscore.

    No masks or frame geometry are supplied, so the only intensity-derived
    signal available is the topology subscore. A subscore at or above the
    baseline floor is treated as continuous; below the floor is treated as a
    discontinuity proxy. When the subscore is missing the proxy is unavailable
    and never imputed.
    """
    subs = rec.get("quality_subscores_before") or {}
    if not isinstance(subs, dict):
        subs = {}
    val = subs.get(TOPOLOGY_SUBSCORE)
    if not isinstance(val, (int, float)):
        return {
            "continuous": None,
            "topology_subscore": None,
            "floor": INTENSITY_PROXY_TOPOLOGY_FLOOR,
            "reason": "missing_measurement",
            "source": "quality_subscores_before.topology",
        }
    continuous = float(val) >= INTENSITY_PROXY_TOPOLOGY_FLOOR
    return {
        "continuous": continuous,
        "topology_subscore": float(val),
        "floor": INTENSITY_PROXY_TOPOLOGY_FLOOR,
        "reason": (
            "topology_subscore_at_or_above_floor"
            if continuous
            else "topology_subscore_below_floor"
        ),
        "source": "quality_subscores_before.topology",
    }


def _intensity_proxy_border_contact(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Intensity-proxy border-contact signal from supplied quality flags.

    Reuses the existing border-contact / border-clip flags as the intensity
    proxy; when neither flag is present the proxy reports no contact. The
    proxy is a monitoring signal only and is never promoted to a label.
    """
    flags = rec.get("quality_flags_before") or []
    if not isinstance(flags, list):
        flags = []
    touches = BORDER_CONTACT_FLAG in flags or "border_clip_candidate" in flags
    return {
        "touches_border": touches,
        "reason": (
            "border_contact_flag_present" if touches else "no_border_contact_flag"
        ),
        "source": "quality_flags_before",
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
    """Per-case LGE MI component scan with a populated reason for every candidate."""
    cid = rec.get("case_id")
    area_ratio = _area_ratio(rec)
    cavity = rec.get("before_cavity")
    cluster = _classify(area_ratio, cavity)
    topology = _topology_flags(rec)
    secondary = _secondary_component_fields(rec)
    spatial_out = _spatial_decision(cid, spatial_index, observed)
    spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION

    component_count = _component_count(rec)
    sec_ratio = _secondary_to_primary_ratio(rec)
    continuity = _cross_slice_continuity(rec)
    intensity_continuity = _intensity_proxy_continuity(rec)
    border = _border_contact(rec)
    intensity_border = _intensity_proxy_border_contact(rec)
    contrast = _contrast_support(rec)

    if area_ratio is None or cavity is None:
        reason = "missing_measurement"
    elif _is_unrouted(cluster, cavity):
        reason = "unrouted"
    elif spatial_hit:
        reason = "spatial_gate_suppressed"
    elif component_count is None:
        reason = "missing_component_count"
    elif sec_ratio is None:
        reason = "missing_secondary_ratio"
    elif sec_ratio >= SECONDARY_AREA_RATIO_THRESHOLD:
        reason = "secondary_component_dominant"
    elif not continuity["continuous"]:
        reason = "cross_slice_discontinuity"
    elif intensity_continuity["continuous"] is False:
        reason = "intensity_proxy_discontinuity"
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
        "secondary_to_primary_area_ratio": sec_ratio,
        "cross_slice_continuity": continuity,
        "intensity_proxy_continuity": intensity_continuity,
        "border_contact": border,
        "intensity_proxy_border_contact": intensity_border,
        "contrast_support": contrast,
        "suppression_candidate": suppression_candidate,
        "suppression_reason": reason,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only LGE MI component scan over the supplied pending volumes."""
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
    component_count_filled = 0
    sec_ratio_filled = 0
    continuity_filled = 0
    intensity_continuity_filled = 0
    border_filled = 0
    intensity_border_filled = 0
    contrast_filled = 0
    suppression_reasons: Dict[str, int] = {ROUTED_REASON: 0}
    suppression_candidates: List[Dict[str, Any]] = []
    no_reason_count = 0
    cases: Dict[str, Any] = {}

    continuity_agreement_denominator = 0
    continuity_agreement_numerator = 0

    for rec in records:
        scanned = _scan_case(rec, spatial_index, observed)
        cid = scanned["case_id"]

        cc = scanned["component_count"]
        if cc is not None:
            component_count_filled += 1
            key = str(cc)
            component_count_distribution[key] = component_count_distribution.get(key, 0) + 1
        if scanned["secondary_to_primary_area_ratio"] is not None:
            sec_ratio_filled += 1
        if scanned["cross_slice_continuity"].get("continuous") is not None:
            continuity_filled += 1
        if scanned["intensity_proxy_continuity"].get("continuous") is not None:
            intensity_continuity_filled += 1
        if scanned["border_contact"].get("touches_border") is not None:
            border_filled += 1
        if scanned["intensity_proxy_border_contact"].get("touches_border") is not None:
            intensity_border_filled += 1
        if scanned["contrast_support"].get("supported") is not None:
            contrast_filled += 1

        flag_cont = scanned["cross_slice_continuity"].get("continuous")
        proxy_cont = scanned["intensity_proxy_continuity"].get("continuous")
        if flag_cont is not None and proxy_cont is not None:
            continuity_agreement_denominator += 1
            if bool(flag_cont) == bool(proxy_cont):
                continuity_agreement_numerator += 1

        reason = scanned["suppression_reason"]
        suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1
        if scanned["suppression_candidate"]:
            if not reason:
                no_reason_count += 1
            suppression_candidates.append({
                "case_id": cid,
                "reason": reason,
                "component_count": cc,
                "secondary_to_primary_area_ratio": scanned["secondary_to_primary_area_ratio"],
                "intensity_proxy_continuity": proxy_cont,
                "intensity_proxy_border_contact": scanned["intensity_proxy_border_contact"].get("touches_border"),
            })

        cases[cid] = scanned

    def _rate(filled: int) -> Optional[float]:
        if not n:
            return None
        return round(filled / n, 6)

    component_count_fill_rate = _rate(component_count_filled)
    sec_ratio_fill_rate = _rate(sec_ratio_filled)
    continuity_fill_rate = _rate(continuity_filled)
    intensity_continuity_fill_rate = _rate(intensity_continuity_filled)
    border_fill_rate = _rate(border_filled)
    intensity_border_fill_rate = _rate(intensity_border_filled)
    contrast_fill_rate = _rate(contrast_filled)

    continuity_agreement: Optional[float] = None
    if continuity_agreement_denominator:
        continuity_agreement = round(
            continuity_agreement_numerator / continuity_agreement_denominator, 6
        )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_shadow_lge_mi_component_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "per-case component topology is monitoring-grade and concordance, "
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
            "continuity_flag": CONTINUITY_FLAG,
            "border_contact_flag": BORDER_CONTACT_FLAG,
            "contrast_support_subscore": CONTRAST_SUPPORT_SUBSCORE,
            "intensity_proxy_topology_subscore": TOPOLOGY_SUBSCORE,
            "intensity_proxy_topology_floor": INTENSITY_PROXY_TOPOLOGY_FLOOR,
            "continuity_agreement_min": CONTINUITY_AGREEMENT_MIN,
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
        "target_path": "experiments/shadow_lge_mi_component_scan.py",
        "component_count_distribution": component_count_distribution,
        "component_count_filled": component_count_filled,
        "component_count_fill_rate": component_count_fill_rate,
        "secondary_to_primary_area_ratio_filled": sec_ratio_filled,
        "secondary_to_primary_area_ratio_fill_rate": sec_ratio_fill_rate,
        "cross_slice_continuity_filled": continuity_filled,
        "cross_slice_continuity_fill_rate": continuity_fill_rate,
        "intensity_proxy_continuity_filled": intensity_continuity_filled,
        "intensity_proxy_continuity_fill_rate": intensity_continuity_fill_rate,
        "border_contact_filled": border_filled,
        "border_contact_fill_rate": border_fill_rate,
        "intensity_proxy_border_contact_filled": intensity_border_filled,
        "intensity_proxy_border_contact_fill_rate": intensity_border_fill_rate,
        "contrast_support_filled": contrast_filled,
        "contrast_support_fill_rate": contrast_fill_rate,
        "fill_rate_threshold": FILL_RATE_MIN,
        "component_count_fill_rate_meets_threshold": (
            component_count_fill_rate is not None
            and component_count_fill_rate > FILL_RATE_MIN
        ),
        "secondary_to_primary_area_ratio_fill_rate_meets_threshold": (
            sec_ratio_fill_rate is not None
            and sec_ratio_fill_rate > FILL_RATE_MIN
        ),
        "cross_slice_continuity_agreement": continuity_agreement,
        "cross_slice_continuity_agreement_denominator": continuity_agreement_denominator,
        "cross_slice_continuity_agreement_numerator": continuity_agreement_numerator,
        "cross_slice_continuity_agreement_threshold": CONTINUITY_AGREEMENT_MIN,
        "cross_slice_continuity_agreement_meets_threshold": (
            continuity_agreement is not None
            and continuity_agreement > CONTINUITY_AGREEMENT_MIN
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
