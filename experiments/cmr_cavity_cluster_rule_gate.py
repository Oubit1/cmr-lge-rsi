"""Read-only rule draft: CMR cavity-context cluster shadow-triage gate (S757).

Three testable clauses:

  R1  The three cavity-context clusters (adequate_area_low_cavity,
      low_area_adequate_cavity, low_area_low_cavity) are routed to SHADOW
      TRIAGE only. No production routing or label promotion is implied.
  R2  Any additive edit must preserve the connected-component count and the
      hole count, and must not touch the border. Declared here; measurement
      is reported unavailable because no masks or frame geometry are supplied.
  R3  Cluster-level escalation requires per-cluster transition counts with
      denominators first. Escalation is blocked while counts are absent.

Classification and thresholds are reused from
cmr_cavity_cluster_holdout_transition_check (no second harness).

Read-only: writes no production labels, code, datasets or runtime databases.
Concordance, precision and CI are reported unavailable (no independent
reference labels).
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

MODULE_VERSION = "cmr_cavity_cluster_rule_gate/1.1"

ROUTING_MODE = "shadow_triage_only"

COHORT_LABEL = "current_pending_queue_not_independent_holdout"

ADDITIVE_EDIT_INVARIANTS = (
    "connected_component_count_unchanged",
    "hole_count_unchanged",
    "no_border_contact",
)

NO_WRITE_CLAUSE = (
    "Read-only rule draft. No production labels, code, datasets or runtime "
    "databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_cavity_cluster_rule_gate.py "
    "tests/test_cmr_cavity_cluster_rule_gate.py"
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


def rule_text() -> Dict[str, Any]:
    """Static rule text: three clauses, each with a measurable predicate."""
    return {
        "R1_cluster_routing": {
            "clusters": list(CLUSTERS),
            "routing_mode": ROUTING_MODE,
            "predicate": "every classified record routes to shadow_triage_only",
        },
        "R2_additive_edit_invariants": {
            "invariants": list(ADDITIVE_EDIT_INVARIANTS),
            "predicate": (
                "component_count and hole_count unchanged and no border contact; "
                "measurement unavailable without masks or frame geometry"
            ),
        },
        "R3_escalation_gate": {
            "predicate": (
                "cluster-level escalation allowed only when per-cluster transition "
                "counts with denominators are present"
            ),
        },
    }


def _transition_denominator(
    cluster_counts: Dict[str, Dict[str, int]],
    transitions: Dict[str, Dict[str, int]],
    cluster: str,
) -> int:
    """Per-cluster transition denominator = classified count for that cluster."""
    return cluster_counts[cluster]["count"]


def _escalation_gate(
    cluster_counts: Dict[str, Dict[str, int]],
    transitions: Dict[str, Dict[str, int]],
    n: int,
) -> Dict[str, Any]:
    """R3: escalation blocked until per-cluster transition counts exist.

    The transition accumulator holds only checked/trash/no_transition counts;
    its denominator is the per-cluster classified count, so the gate derives
    the denominator instead of reading a key that is not present.
    """
    have_counts = n > 0 and all(cluster_counts[c]["denominator"] == n for c in CLUSTERS)
    have_transitions = all(
        transitions[c]["checked"]
        + transitions[c]["trash"]
        + transitions[c]["no_transition"]
        == _transition_denominator(cluster_counts, transitions, c)
        for c in CLUSTERS
    )
    return {
        "requires_per_cluster_transition_counts": True,
        "per_cluster_counts_present": have_counts,
        "per_cluster_transition_counts_present": have_transitions,
        "escalation_allowed": False,
        "reason": (
            "per-cluster transition counts with denominators required before "
            "cluster-level escalation; shadow triage only"
        ),
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
            "additive_edit_invariants": {
                inv: "unmeasured_no_masks_or_geometry" for inv in ADDITIVE_EDIT_INVARIANTS
            },
        }

    cluster_counts = {c: {"count": counts[c], "denominator": n} for c in CLUSTERS}

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_rule_gate",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Pending-queue sample is not an independent held-out cohort; "
            "routing conclusions stay provisional."
        ),
        "rule_text": rule_text(),
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
        "additive_edit_invariants": {
            inv: {"status": "declared_unmeasured_no_masks_or_geometry"}
            for inv in ADDITIVE_EDIT_INVARIANTS
        },
        "escalation_gate": _escalation_gate(cluster_counts, transitions, n),
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
