"""Read-only pending-volume cluster check (S779).

Scope (shadow_triage_only, zero production writes):
  * classify each pending CMR volume into the three accepted clusters
    (adequate_area_low_cavity / low_area_adequate_cavity / low_area_low_cavity)
    using the shared classifier from cmr_cavity_cluster_holdout_transition_check;
  * emit per-case cluster flags, suppression reasons and simulated
    checked/trash transition counts with denominators;
  * report missing fields as unavailable, never imputed;
  * emit module hash, rollback command and an empty zero-write audit diff.

Reuses existing helpers (no second harness). Read-only: writes no production
labels, code, datasets or runtime databases. Concordance, precision and CI
against independent reference labels are reported unavailable, never imputed.
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

MODULE_VERSION = "cmr_pending_volume_cluster_check/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

EXPECTED_TOTAL = 156

NO_WRITE_CLAUSE = (
    "Read-only pending-volume cluster check. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_pending_volume_cluster_check.py "
    "tests/test_cmr_pending_volume_cluster_check.py"
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


def _suppression_reasons(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Per-case suppression flags from supplied fields.

    Each condition is read from an explicit field; when absent it is reported
    unavailable (None), never assumed false.
    """
    out: Dict[str, Any] = {}
    for cond in SUPPRESSION_CONDITIONS:
        val = rec.get(cond)
        out[cond] = {
            "value": val,
            "status": "ok" if val is not None else "unavailable_no_masks_or_geometry",
        }
    return out


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only pending-volume cluster check over the supplied CMR volumes."""
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    suppression_counts: Dict[str, int] = {cond: 0 for cond in SUPPRESSION_CONDITIONS}
    suppression_available: Dict[str, bool] = {cond: False for cond in SUPPRESSION_CONDITIONS}
    cases: Dict[str, Any] = {}
    unclassified: List[str] = []
    missing_measurement: List[str] = []
    suppressed_cases: List[str] = []

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

        reasons = _suppression_reasons(rec)
        suppressed = False
        for cond, info in reasons.items():
            if info["value"] is None:
                continue
            suppression_available[cond] = True
            if info["value"]:
                suppression_counts[cond] += 1
                suppressed = True
        if suppressed:
            suppressed_cases.append(cid)

        if cluster is not None and not suppressed:
            transition = _simulate_transition(cluster, area_ratio, cavity)
            transitions[cluster][transition] += 1
            routing = ROUTING_MODE
        elif cluster is not None and suppressed:
            transition = "no_transition"
            routing = "stack_level_review"
        else:
            transition = "no_transition"
            routing = "unrouted"

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "suppression_reasons": reasons,
            "suppressed": suppressed,
            "routing": routing,
            "simulated_transition": transition,
        }

    classified_total = sum(counts.values())
    simulated_checked = sum(transitions[c]["checked"] for c in CLUSTERS)
    simulated_trash = sum(transitions[c]["trash"] for c in CLUSTERS)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_pending_volume_cluster_check",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are pending candidates, not verified truth; "
            "concordance, precision and CI against independent labels stay "
            "unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
        },
        "n_records": n,
        "expected_total": EXPECTED_TOTAL,
        "cluster_counts": {
            c: {"count": counts[c], "denominator": n} for c in CLUSTERS
        },
        "classified_total": classified_total,
        "classified_total_matches_expected": classified_total == EXPECTED_TOTAL,
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
        "simulated_checked_total": simulated_checked,
        "simulated_trash_total": simulated_trash,
        "suppression_reason_distribution": {
            cond: {
                "count": suppression_counts[cond] if suppression_available[cond] else None,
                "denominator": n,
                "status": (
                    "ok" if suppression_available[cond]
                    else "unavailable_no_masks_or_geometry"
                ),
            }
            for cond in SUPPRESSION_CONDITIONS
        },
        "suppressed_cases": suppressed_cases,
        "unclassified_cases": unclassified,
        "missing_measurement_cases": missing_measurement,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "suppression_reasons": "no masks, frame geometry or centroid series supplied",
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
