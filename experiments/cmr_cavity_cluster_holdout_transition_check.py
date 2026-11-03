"""Read-only cavity-context cluster holdout transition check for CMR volumes.

Classifies each supplied per-volume area/cavity measurement into one of three
clusters using reference_area as the area denominator and a cavity threshold
(reusing the thresholds from cmr_cavity_cluster_shadow_counts):

  adequate_area_low_cavity   : area_ratio >= AREA_RATIO_MIN and cavity <  CAVITY_MAX
  low_area_adequate_cavity   : area_ratio <  AREA_RATIO_MIN and cavity >= CAVITY_MAX
  low_area_low_cavity        : area_ratio <  AREA_RATIO_MIN and cavity <  CAVITY_MAX

For each cluster it reports the per-case count with denominator and simulated
checked/trash transition counts. Suppression coverage for empty / border-touch /
temporal-hole / centroid-drift is reported as unavailable because no masks,
frame geometry or centroid series are supplied. Missing measurements are never
imputed as zero.

This module is read-only: it writes no labels, code or datasets. Concordance,
precision and CI are reported unavailable (no independent reference labels).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, List, Optional

MODULE_VERSION = "cmr_cavity_cluster_holdout_transition_check/1.0"
AREA_RATIO_MIN = 0.5
CAVITY_MAX = 0.5

CLUSTERS = (
    "adequate_area_low_cavity",
    "low_area_adequate_cavity",
    "low_area_low_cavity",
)

SUPPRESSION_CONDITIONS = (
    "empty",
    "border_touch",
    "temporal_hole",
    "centroid_drift",
)

NO_WRITE_CLAUSE = (
    "Read-only holdout transition check. No production labels, code, datasets "
    "or runtime databases are written or mutated."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_cavity_cluster_holdout_transition_check.py "
    "tests/test_cmr_cavity_cluster_holdout_transition_check.py"
)


def _classify(area_ratio: Optional[float], cavity: Optional[float]) -> Optional[str]:
    """Return cluster name, or None when a required measurement is missing."""
    if area_ratio is None or cavity is None:
        return None
    low_area = area_ratio < AREA_RATIO_MIN
    low_cavity = cavity < CAVITY_MAX
    if not low_area and low_cavity:
        return "adequate_area_low_cavity"
    if low_area and not low_cavity:
        return "low_area_adequate_cavity"
    if low_area and low_cavity:
        return "low_area_low_cavity"
    return None  # adequate area, adequate cavity -> outside the three clusters


def _simulate_transition(cluster: Optional[str], area_ratio: Optional[float], cavity: Optional[float]) -> str:
    """Deterministic simulated reviewer transition for a classified case.

    low_area_low_cavity is the most severe residual cluster and is simulated as
    trash; the other two clusters are simulated as checked. Unclassified or
    missing-measurement cases are simulated as no_transition.
    """
    if cluster is None:
        return "no_transition"
    if cluster == "low_area_low_cavity":
        return "trash"
    return "checked"


def module_hash() -> str:
    src = inspect.getsource(inspect.getmodule(_classify))
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


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only cluster counts and simulated transitions over supplied records."""
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
        ref = rec.get("reference_area")
        area = rec.get("before_area")
        cavity = rec.get("before_cavity")
        area_ratio = (area / ref) if (ref not in (None, 0) and area is not None) else None
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
            "simulated_transition": transition,
        }

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_holdout_transition_check",
        "thresholds": {"area_ratio_min": AREA_RATIO_MIN, "cavity_max": CAVITY_MAX},
        "n_records": n,
        "cluster_counts": {c: {"count": counts[c], "denominator": n} for c in CLUSTERS},
        "simulated_transitions": {
            c: {
                "checked": transitions[c]["checked"],
                "trash": transitions[c]["trash"],
                "no_transition": transitions[c]["no_transition"],
                "denominator": counts[c],
            }
            for c in CLUSTERS
        },
        "unclassified_cases": unclassified,
        "missing_measurement_cases": missing_measurement,
        "suppression_coverage": {
            cond: {"count": None, "denominator": n, "status": "unavailable_no_masks_or_geometry"}
            for cond in SUPPRESSION_CONDITIONS
        },
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
