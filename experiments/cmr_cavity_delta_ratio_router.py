"""Read-only router: CMR cavity-vs-myo area-delta ratio (S758).

New measurement dimension over S757: for each pending CMR volume we compare
cavity delta against myocardial delta and route the three cavity-context
clusters (adequate_area_low_cavity, low_area_adequate_cavity,
low_area_low_cavity) to SHADOW TRIAGE only.

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).

Read-only: writes no production labels, code, datasets or runtime databases.
Concordance, precision and CI are reported unavailable (no independent
reference labels). Suppression coverage is unavailable (no masks/geometry).
"""

from __future__ import annotations

import hashlib
import inspect
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

MODULE_VERSION = "cmr_cavity_delta_ratio_router/1.1"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

TARGET_CLUSTER = "adequate_area_low_cavity"

NO_WRITE_CLAUSE = (
    "Read-only router. No production labels, code, datasets or runtime "
    "databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_cavity_delta_ratio_router.py "
    "tests/test_cmr_cavity_delta_ratio_router.py"
)


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(module_hash))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def audit_diff(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
    if previous is None:
        return {
            "baseline": "none",
            "changed_cases": [],
            "added_cases": sorted(current.get("cases", {}).keys()),
            "removed_cases": [],
            "production_writes": 0,
        }
    prev_cases = previous.get("cases", {})
    cur_cases = current.get("cases", {})
    changed = [c for c in cur_cases if c in prev_cases and prev_cases[c] != cur_cases[c]]
    added = [c for c in cur_cases if c not in prev_cases]
    removed = [c for c in prev_cases if c not in cur_cases]
    return {
        "baseline": previous.get("module_hash"),
        "changed_cases": changed,
        "added_cases": added,
        "removed_cases": removed,
        "production_writes": 0,
    }


def _area_ratio(rec: Dict[str, Any]) -> Optional[float]:
    ref = rec.get("reference_area")
    area = rec.get("before_area")
    if ref in (None, 0) or area is None:
        return None
    return area / ref


def _delta_ratio(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Cavity delta vs myo delta from supplied before/after measurements.

    Deltas are taken from the supplied before/after area and cavity fields.
    Missing measurements are reported as None, never imputed as zero.
    """
    ba, aa = rec.get("before_area"), rec.get("after_area")
    bc, ac = rec.get("before_cavity"), rec.get("after_cavity")
    myo_delta = None if (ba is None or aa is None) else (aa - ba)
    cavity_delta = None if (bc is None or ac is None) else (ac - bc)
    ratio = None
    if myo_delta is not None and cavity_delta is not None and myo_delta != 0:
        ratio = cavity_delta / myo_delta
    cavity_gt_myo = None
    if myo_delta is not None and cavity_delta is not None:
        cavity_gt_myo = cavity_delta > myo_delta
    return {
        "myo_delta": myo_delta,
        "cavity_delta": cavity_delta,
        "cavity_myo_delta_ratio": ratio,
        "cavity_delta_gt_myo_delta": cavity_gt_myo,
    }


def _route_transition(
    cluster: Optional[str],
    base_transition: str,
    delta: Dict[str, Any],
) -> str:
    """Delta-ratio routing: target cluster with cavity delta > myo delta -> trash.

    The reused _simulate_transition provides the baseline transition. The new
    measurement dimension promotes the target cluster to a simulated trash
    transition when the cavity delta exceeds the myocardial delta. This is a
    shadow-triage signal only; no production routing is implied.
    """
    if cluster == TARGET_CLUSTER and delta.get("cavity_delta_gt_myo_delta"):
        return "trash"
    return base_transition


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only per-cluster counts, delta-ratio routing, simulated transitions."""
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    cases: Dict[str, Any] = {}
    unclassified: List[str] = []
    missing_measurement: List[str] = []
    target_cavity_gt_myo: List[str] = []
    target_trash = 0

    for rec in records:
        cid = rec.get("case_id")
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        cluster = _classify(area_ratio, cavity)
        if area_ratio is None or cavity is None:
            missing_measurement.append(cid)
        if cluster is None:
            unclassified.append(cid)
        else:
            counts[cluster] += 1
        delta = _delta_ratio(rec)
        base_transition = _simulate_transition(cluster, area_ratio, cavity)
        transition = _route_transition(cluster, base_transition, delta)
        if cluster is not None:
            transitions[cluster][transition] += 1
        if cluster == TARGET_CLUSTER and delta["cavity_delta_gt_myo_delta"]:
            target_cavity_gt_myo.append(cid)
            if transition == "trash":
                target_trash += 1
        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "routing": ROUTING_MODE if cluster is not None else "unrouted",
            "simulated_transition": transition,
            "delta_ratio": delta,
        }

    cluster_counts = {c: {"count": counts[c], "denominator": n} for c in CLUSTERS}

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_delta_ratio_router",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Pending-queue sample is not an independent held-out cohort; "
            "routing conclusions stay provisional."
        ),
        "thresholds": {"area_ratio_min": AREA_RATIO_MIN, "cavity_max": CAVITY_MAX},
        "n_records": n,
        "cluster_counts": cluster_counts,
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
        "delta_ratio_routing": {
            "target_cluster": TARGET_CLUSTER,
            "cavity_delta_gt_myo_delta_count": len(target_cavity_gt_myo),
            "cavity_delta_gt_myo_delta_cases": target_cavity_gt_myo,
            "simulated_trash_transitions": target_trash,
            "denominator": counts[TARGET_CLUSTER],
        },
        "suppression_coverage": {
            cond: {
                "count": None,
                "denominator": n,
                "status": "unavailable_no_masks_or_geometry",
            }
            for cond in SUPPRESSION_CONDITIONS
        },
        "unclassified_cases": unclassified,
        "missing_measurement_cases": missing_measurement,
        "unavailable": {
            "suppression_coverage": "no masks, frame geometry or centroid series supplied",
            "concordance": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
        },
        "cases": cases,
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
