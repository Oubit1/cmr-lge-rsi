"""Read-only cavity-context cluster transition evaluation (CMR, S757).

Runs the three-cluster cavity-context shadow triage on the current pending
CMR queue and reports:

  * per-cluster counts with denominators for adequate_area_low_cavity,
    low_area_adequate_cavity, low_area_low_cavity;
  * per-cluster simulated checked / trash / no_transition counts with
    denominators;
  * suppression coverage counts with denominators for empty, border-touch,
    temporal-hole and centroid-drift (reported unavailable: no masks or
    frame geometry supplied);
  * module hash, audit diff and rollback command proving zero production
    label / code / dataset writes.

The cohort is the current pending queue, NOT an independent held-out
cohort; routing conclusions stay provisional. Concordance, precision and
CI are reported unavailable (no independent reference labels).

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).
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

MODULE_VERSION = "cmr_cavity_cluster_transition_holdout/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

NO_WRITE_CLAUSE = (
    "Read-only evaluation. No production labels, code, datasets or runtime "
    "databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_cavity_cluster_transition_holdout.py "
    "tests/test_cmr_cavity_cluster_transition_holdout.py"
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


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only per-cluster counts, simulated transitions, suppression coverage."""
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    cases: Dict[str, Any] = {}
    unclassified: List[str] = []
    missing_measurement: List[str] = []

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
        transition = _simulate_transition(cluster, area_ratio, cavity)
        if cluster is not None:
            transitions[cluster][transition] += 1
        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "routing": ROUTING_MODE if cluster is not None else "unrouted",
            "simulated_transition": transition,
        }

    cluster_counts = {c: {"count": counts[c], "denominator": n} for c in CLUSTERS}

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_cluster_transition_evaluation",
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
            "additive_edit_invariants": "no masks or frame geometry supplied",
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
