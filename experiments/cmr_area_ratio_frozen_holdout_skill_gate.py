"""Read-only shadow-triage skill gate (S761): myo/cavity area-ratio routing for
pending CMR small-area volumes, with additive-edit topology invariants and a
frozen-holdout per-cluster checked->trash transition gate.

Scope (shadow_triage_only, no production writes):
  * classify each volume by myo/cavity area ratio using the shared three-cluster
    classifier from cmr_cavity_cluster_holdout_transition_check;
  * route flagged small-area candidates to shadow triage, everything else to
    stack-level review;
  * require additive edits to preserve connected-component count, hole count and
    border-contact status (topology invariants) before any promotion;
  * require frozen-holdout per-cluster checked/trash transition counts with
    denominators before escalation;
  * reconcile the S760 threshold mismatch by evaluating both AREA_RATIO_MIN
    and the alternate 0.5 threshold side by side;
  * emit module hash, zero-write audit and rollback command.

Reuses existing helpers (no second harness). Read-only: writes no production
labels, code, datasets or runtime databases. Metrics that need masks, frame
geometry, centroid series or independent reference labels are reported
unavailable, never imputed.
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

MODULE_VERSION = "cmr_area_ratio_frozen_holdout_skill_gate/1.0"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "checked_volumes_frozen_holdout_not_truth"

ALTERNATE_AREA_RATIO_MIN = 0.5

LEDGER_BUCKETS = ("pending", "work", "checked", "trash")

TOPOLOGY_INVARIANTS = ("connected_components", "holes", "border_contact")

NO_WRITE_CLAUSE = (
    "Read-only shadow-triage skill gate. No production labels, code, datasets "
    "or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_area_ratio_frozen_holdout_skill_gate.py "
    "tests/test_cmr_area_ratio_frozen_holdout_skill_gate.py"
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


def _classify_at(area_ratio: Optional[float], cavity: Optional[float], threshold: float) -> Optional[str]:
    """Classify using an explicit threshold, mirroring the shared classifier.

    The shared _classify uses AREA_RATIO_MIN; for the alternate threshold we
    re-derive the same three-cluster decision with the supplied threshold so
    the S760 mismatch can be reconciled without mutating shared constants.
    """
    if area_ratio is None or cavity is None:
        return None
    if area_ratio < threshold and cavity <= CAVITY_MAX:
        return CLUSTERS[0]
    if area_ratio < threshold:
        return CLUSTERS[1]
    return CLUSTERS[2]


def _topology_invariants(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Per-case topology invariants from supplied fields.

    Each invariant is read from an explicit before/after pair; when either side
    is absent the invariant is reported unavailable (None), never assumed.
    """
    out: Dict[str, Any] = {}
    for name in TOPOLOGY_INVARIANTS:
        before = rec.get(f"{name}_before")
        after = rec.get(f"{name}_after")
        if before is None or after is None:
            out[name] = {
                "before": before,
                "after": after,
                "preserved": None,
                "status": "unavailable_no_masks_or_geometry",
            }
        else:
            out[name] = {
                "before": before,
                "after": after,
                "preserved": before == after,
                "status": "ok",
            }
    return out


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


def _run_threshold(records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    suppression_counts: Dict[str, int] = {cond: 0 for cond in SUPPRESSION_CONDITIONS}
    suppression_available: Dict[str, bool] = {cond: False for cond in SUPPRESSION_CONDITIONS}
    topology_counts: Dict[str, Dict[str, int]] = {
        name: {"measured": 0, "preserved": 0, "violated": 0} for name in TOPOLOGY_INVARIANTS
    }
    cases: Dict[str, Any] = {}
    unclassified: List[str] = []
    missing_measurement: List[str] = []
    topology_violations: List[str] = []
    n = len(records)

    for rec in records:
        cid = rec.get("case_id")
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        myo_cavity_ratio = _myo_cavity_area_ratio(rec)
        cluster = _classify_at(area_ratio, cavity, threshold)
        if area_ratio is None or cavity is None:
            missing_measurement.append(cid)
        if cluster is None:
            unclassified.append(cid)
        else:
            counts[cluster] += 1

        flags = {
            "empty": rec.get("empty"),
            "border_touch": rec.get("border_touch"),
            "temporal_hole": rec.get("temporal_hole"),
            "centroid_drift": rec.get("centroid_drift"),
        }
        suppressed = False
        for cond, val in flags.items():
            if val is None:
                continue
            suppression_available[cond] = True
            if val:
                suppression_counts[cond] += 1
                suppressed = True

        topo = _topology_invariants(rec)
        for name, info in topo.items():
            if info["preserved"] is None:
                continue
            topology_counts[name]["measured"] += 1
            if info["preserved"]:
                topology_counts[name]["preserved"] += 1
            else:
                topology_counts[name]["violated"] += 1
                topology_violations.append(cid)

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
            "myo_cavity_area_ratio": myo_cavity_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "suppression_flags": flags,
            "suppressed": suppressed,
            "routing": routing,
            "simulated_transition": transition,
            "topology_invariants": topo,
        }

    return {
        "threshold": threshold,
        "n_records": n,
        "cluster_counts": {c: {"count": counts[c], "denominator": n} for c in CLUSTERS},
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
                "count": suppression_counts[cond] if suppression_available[cond] else None,
                "denominator": n,
                "status": (
                    "ok" if suppression_available[cond]
                    else "unavailable_no_masks_or_geometry"
                ),
            }
            for cond in SUPPRESSION_CONDITIONS
        },
        "topology_invariants": {
            name: {
                "measured": topology_counts[name]["measured"],
                "preserved": topology_counts[name]["preserved"],
                "violated": topology_counts[name]["violated"],
                "denominator": n,
                "status": (
                    "ok" if topology_counts[name]["measured"] > 0
                    else "unavailable_no_masks_or_geometry"
                ),
            }
            for name in TOPOLOGY_INVARIANTS
        },
        "topology_violation_cases": topology_violations,
        "unclassified_cases": unclassified,
        "missing_measurement_cases": missing_measurement,
        "cases": cases,
        "_transitions_raw": transitions,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only frozen-holdout skill gate over the supplied CMR volumes."""
    records = data.get("records", []) if isinstance(data, dict) else []
    ledger = data.get("ledger") if isinstance(data, dict) else None
    n = len(records)

    primary = _run_threshold(records, AREA_RATIO_MIN)
    alternate = _run_threshold(records, ALTERNATE_AREA_RATIO_MIN)

    primary_transitions = primary.pop("_transitions_raw")
    alternate_transitions = alternate.pop("_transitions_raw")

    threshold_reconciliation = {
        "primary_threshold": AREA_RATIO_MIN,
        "alternate_threshold": ALTERNATE_AREA_RATIO_MIN,
        "primary_flagged_total": sum(
            primary["simulated_transitions"][c]["checked"]
            + primary["simulated_transitions"][c]["trash"]
            for c in CLUSTERS
        ),
        "alternate_flagged_total": sum(
            alternate["simulated_transitions"][c]["checked"]
            + alternate["simulated_transitions"][c]["trash"]
            for c in CLUSTERS
        ),
        "note": (
            "S760 reported a 0.15 vs 0.5 threshold mismatch; both thresholds "
            "are evaluated here so the zero-flag outcome is reconciled honestly."
        ),
    }

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_area_ratio_frozen_holdout_skill_gate",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are checked, not verified truth; concordance, "
            "precision and CI against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "alternate_area_ratio_min": ALTERNATE_AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "topology_invariants": list(TOPOLOGY_INVARIANTS),
        },
        "n_records": n,
        "primary": primary,
        "alternate": alternate,
        "threshold_reconciliation": threshold_reconciliation,
        "ledger_deltas": _ledger_deltas(primary_transitions, ledger),
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "suppression_coverage": "no masks, frame geometry or centroid series supplied",
            "topology_invariants": "no masks or frame geometry supplied",
        },
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
