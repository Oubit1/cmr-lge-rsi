"""Read-only pseudo-holdout concordance check for the S759 myo/cavity area-ratio rule (S760).

Applies the S759 three-cluster myo/cavity area-ratio routing to the supplied
checked CMR volumes and reports, per cluster:
  * flag counts with denominators;
  * flag-vs-checked/trash concordance (PSEUDO-HOLDOUT, not verified truth);
  * simulated checked/trash transition counts and deltas vs the current ledger;
  * suppression coverage (unavailable: no masks/geometry);
  * zero-write audit (module hash, audit diff, rollback command).

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).

Read-only: writes no production labels, code, datasets or runtime databases.
Concordance/precision/CI against independent labels are reported unavailable.
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

MODULE_VERSION = "cmr_area_ratio_pseudo_holdout_check/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "pseudo_holdout_checked_volumes_not_truth"

LEDGER_BUCKETS = ("pending", "work", "checked", "trash")

NO_WRITE_CLAUSE = (
    "Read-only pseudo-holdout check. No production labels, code, datasets or "
    "runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_area_ratio_pseudo_holdout_check.py "
    "tests/test_cmr_area_ratio_pseudo_holdout_check.py"
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


def _myo_cavity_area_ratio(rec: Dict[str, Any]) -> Optional[float]:
    """Myocardial/cavity area ratio from supplied measurements.

    Missing measurements are reported as None, never imputed as zero.
    """
    area = rec.get("before_area")
    cavity = rec.get("before_cavity")
    if area is None or cavity in (None, 0):
        return None
    return area / cavity


def _ledger_deltas(
    transitions: Dict[str, Dict[str, int]],
    ledger: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-cluster simulated checked/trash counts vs current ledger buckets.

    The ledger is supplied by the caller; when absent, deltas are reported as
    unavailable rather than fabricated.
    """
    if not isinstance(ledger, dict):
        return {
            "status": "unavailable_no_ledger_supplied",
            "buckets": list(LEDGER_BUCKETS),
        }
    out: Dict[str, Any] = {"status": "ok", "buckets": list(LEDGER_BUCKETS)}
    for c in CLUSTERS:
        sim_checked = transitions[c]["checked"]
        sim_trash = transitions[c]["trash"]
        cur = ledger.get(c, {}) if isinstance(ledger.get(c), dict) else {}
        cur_checked = cur.get("checked")
        cur_trash = cur.get("trash")
        out[c] = {
            "simulated_checked": sim_checked,
            "simulated_trash": sim_trash,
            "ledger_checked": cur_checked,
            "ledger_trash": cur_trash,
            "checked_delta": None if cur_checked is None else sim_checked - cur_checked,
            "trash_delta": None if cur_trash is None else sim_trash - cur_trash,
        }
    return out


def _concordance(
    cases: Dict[str, Any],
    statuses: Dict[str, str],
) -> Dict[str, Any]:
    """Flag-vs-checked/trash concordance on the pseudo-holdout cohort.

    A flagged case (cluster is not None) is concordant when its supplied
    checked/trash status is 'trash'; an unflagged case is concordant when its
    status is 'checked'. Statuses are pseudo-holdout, not verified truth.
    """
    per_cluster: Dict[str, Any] = {}
    total_flagged = 0
    total_flagged_concordant = 0
    total_unflagged = 0
    total_unflagged_concordant = 0
    unknown_status: List[str] = []

    for c in CLUSTERS:
        per_cluster[c] = {
            "flagged": 0,
            "flagged_concordant": 0,
            "unflagged": 0,
            "unflagged_concordant": 0,
            "denominator": 0,
        }

    for cid, rec in cases.items():
        status = statuses.get(cid)
        if status not in ("checked", "trash"):
            unknown_status.append(cid)
            continue
        cluster = rec.get("cluster")
        if cluster is None:
            total_unflagged += 1
            if status == "checked":
                total_unflagged_concordant += 1
            continue
        bucket = per_cluster[cluster]
        bucket["flagged"] += 1
        bucket["denominator"] += 1
        total_flagged += 1
        if status == "trash":
            bucket["flagged_concordant"] += 1
            total_flagged_concordant += 1

    for c in CLUSTERS:
        b = per_cluster[c]
        b["concordance"] = (
            None if b["denominator"] == 0
            else b["flagged_concordant"] / b["denominator"]
        )

    return {
        "cohort": COHORT_LABEL,
        "note": (
            "checked/trash is a pseudo-holdout status, not verified truth; "
            "concordance may mislead if over-read."
        ),
        "per_cluster": per_cluster,
        "flagged_total": total_flagged,
        "flagged_concordant_total": total_flagged_concordant,
        "flagged_concordance": (
            None if total_flagged == 0
            else total_flagged_concordant / total_flagged
        ),
        "unflagged_total": total_unflagged,
        "unflagged_concordant_total": total_unflagged_concordant,
        "unflagged_concordance": (
            None if total_unflagged == 0
            else total_unflagged_concordant / total_unflagged
        ),
        "unknown_status_cases": unknown_status,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only per-cluster counts, area-ratio routing, concordance, transitions."""
    records = data.get("records", []) if isinstance(data, dict) else []
    ledger = data.get("ledger") if isinstance(data, dict) else None
    statuses = data.get("statuses") if isinstance(data, dict) else None
    if not isinstance(statuses, dict):
        statuses = {}
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
        myo_cavity_ratio = _myo_cavity_area_ratio(rec)
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
            "myo_cavity_area_ratio": myo_cavity_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "routing": ROUTING_MODE if cluster is not None else "unrouted",
            "simulated_transition": transition,
        }

    cluster_counts = {c: {"count": counts[c], "denominator": n} for c in CLUSTERS}

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_area_ratio_pseudo_holdout_check",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "158 checked volumes are a pseudo-holdout, not verified truth; "
            "concordance conclusions stay provisional."
        ),
        "thresholds": {"area_ratio_min": AREA_RATIO_MIN, "cavity_max": CAVITY_MAX},
        "n_records": n,
        "cluster_counts": cluster_counts,
        "concordance": _concordance(cases, statuses),
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
        "ledger_deltas": _ledger_deltas(transitions, ledger),
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
            "topology_invariants": "no masks or frame geometry supplied",
            "concordance_vs_independent_labels": "no independent reference labels supplied",
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
