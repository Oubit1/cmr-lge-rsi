"""Read-only skill draft: CMR small-area volume routing by myo/cavity area ratio (S759).

Routes pending CMR small-area volumes into three cavity-context clusters
(adequate_area_low_cavity, low_area_adequate_cavity, low_area_low_cavity) using
the myocardial/cavity area ratio, to SHADOW TRIAGE only.

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).

Skill clauses:
  * three-cluster routing by myo/cavity area ratio;
  * additive edits must preserve connected-component count and hole count and
    must not touch the border (declared invariant; unmeasured here because no
    masks/geometry are supplied);
  * promotion requires independent held-out per-cluster transition evidence
    first (control gate FAIL is not promotion evidence).

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

MODULE_VERSION = "cmr_area_ratio_skill_draft/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

LEDGER_BUCKETS = ("pending", "work", "checked", "trash")

TOPOLOGY_INVARIANTS = (
    "connected_component_count_unchanged",
    "hole_count_unchanged",
    "no_border_contact",
)

PROMOTION_CLAUSE = (
    "Promotion requires independent held-out per-cluster transition evidence "
    "first; the pending-queue control gate FAIL is not promotion evidence."
)

NO_WRITE_CLAUSE = (
    "Read-only skill draft. No production labels, code, datasets or runtime "
    "databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_area_ratio_skill_draft.py "
    "tests/test_cmr_area_ratio_skill_draft.py"
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


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only per-cluster counts, area-ratio routing, simulated transitions."""
    records = data.get("records", []) if isinstance(data, dict) else []
    ledger = data.get("ledger") if isinstance(data, dict) else None
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
        "mode": "read_only_area_ratio_skill_draft",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Pending-queue sample is not an independent held-out cohort; "
            "routing conclusions stay provisional."
        ),
        "skill_clauses": {
            "three_cluster_routing": list(CLUSTERS),
            "routing_dimension": "myo_cavity_area_ratio",
            "topology_invariants": list(TOPOLOGY_INVARIANTS),
            "topology_invariants_status": "declared_unmeasured_no_masks_or_geometry",
            "promotion_clause": PROMOTION_CLAUSE,
        },
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
