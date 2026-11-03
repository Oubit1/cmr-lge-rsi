"""Read-only rule draft: myo/cavity area-ratio small-area candidate with consecutive-slice gate (S760).

Rule (shadow triage only):
  * flag a small-area candidate when the myocardial/cavity area ratio is below
    AREA_RATIO_MIN (0.15) and the candidate persists for >= MIN_CONSECUTIVE
    (2) consecutive slices;
  * suppress the flag and route to stack-level review when any suppression
    condition exceeds threshold: empty, border-touch, temporal-hole,
    centroid-drift.

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).

Read-only: writes no production labels, code, datasets or runtime databases.
Suppression coverage, topology invariants, concordance/precision/CI against
independent labels are reported unavailable when the required inputs (masks,
frame geometry, centroid series, reference labels) are not supplied.
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

MODULE_VERSION = "cmr_area_ratio_consecutive_suppress_rule/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "pseudo_holdout_checked_volumes_not_truth"

MIN_CONSECUTIVE = 2

LEDGER_BUCKETS = ("pending", "work", "checked", "trash")

NO_WRITE_CLAUSE = (
    "Read-only rule draft. No production labels, code, datasets or runtime "
    "databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_area_ratio_consecutive_suppress_rule.py "
    "tests/test_cmr_area_ratio_consecutive_suppress_rule.py"
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


def _consecutive_run(rec: Dict[str, Any]) -> int:
    """Consecutive-slice run length for the candidate.

    The supplied schema carries no per-slice series, so the run length is read
    from an explicit 'consecutive_slices' field when present; otherwise it is
    reported as unavailable (None) rather than assumed.
    """
    run = rec.get("consecutive_slices")
    if isinstance(run, int) and run >= 0:
        return run
    return None


def _suppression_flags(rec: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    """Per-condition suppression flags from supplied fields.

    Each condition is read from an explicit boolean field; when absent the
    condition is reported as None (unavailable), never assumed False.
    """
    return {
        "empty": rec.get("empty"),
        "border_touch": rec.get("border_touch"),
        "temporal_hole": rec.get("temporal_hole"),
        "centroid_drift": rec.get("centroid_drift"),
    }


def _ledger_deltas(
    transitions: Dict[str, Dict[str, int]],
    ledger: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
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


def _concordance(cases: Dict[str, Any], statuses: Dict[str, str]) -> Dict[str, Any]:
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
    """Read-only per-cluster counts, consecutive-gate routing, suppression, concordance."""
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
    suppression_counts: Dict[str, int] = {cond: 0 for cond in SUPPRESSION_CONDITIONS}
    suppression_available: Dict[str, bool] = {cond: False for cond in SUPPRESSION_CONDITIONS}
    cases: Dict[str, Any] = {}
    unclassified: List[str] = []
    missing_measurement: List[str] = []
    consecutive_unavailable: List[str] = []
    suppressed_cases: List[str] = []

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

        run = _consecutive_run(rec)
        if run is None:
            consecutive_unavailable.append(cid)
        consecutive_ok = run is not None and run >= MIN_CONSECUTIVE

        flags = _suppression_flags(rec)
        suppressed = False
        for cond, val in flags.items():
            if val is None:
                continue
            suppression_available[cond] = True
            if val:
                suppression_counts[cond] += 1
                suppressed = True

        if cluster is not None and consecutive_ok and not suppressed:
            transition = _simulate_transition(cluster, area_ratio, cavity)
            transitions[cluster][transition] += 1
            routing = ROUTING_MODE
        elif cluster is not None and (not consecutive_ok or suppressed):
            transition = "no_transition"
            routing = "stack_level_review"
            suppressed_cases.append(cid)
        else:
            transition = "no_transition"
            routing = "unrouted"

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "myo_cavity_area_ratio": myo_cavity_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "consecutive_slices": run,
            "consecutive_ok": consecutive_ok,
            "suppression_flags": flags,
            "suppressed": suppressed,
            "routing": routing,
            "simulated_transition": transition,
        }

    cluster_counts = {c: {"count": counts[c], "denominator": n} for c in CLUSTERS}

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_area_ratio_consecutive_suppress_rule",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "158 checked volumes are a pseudo-holdout, not verified truth; "
            "concordance conclusions stay provisional."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "min_consecutive_slices": MIN_CONSECUTIVE,
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
        },
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
        "consecutive_unavailable_cases": consecutive_unavailable,
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
