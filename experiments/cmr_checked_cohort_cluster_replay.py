"""Read-only S792 checked-cohort cluster replay over the 180 checked CMR volumes.

Scope (shadow_triage_only, zero production writes):
  * reuse the existing S788 checked-cluster replay helpers
    (cmr_checked_cluster_replay) plus the S785 per-cluster threshold replay,
    S784 discriminator scan, S782/S783 cluster classifier and S777 spatial
    gate that S788 already imports;
  * apply the S759/S785 per-cluster area_ratio threshold and the S777 spatial
    rank/core gate to each supplied checked volume and emit per-cluster
    simulated checked/trash/no_transition counts whose total equals the checked
    denominator (180);
  * reconcile the three-cluster simulated total against the 96 pending
    baseline and the S783 108-case no_transition baseline, or mark
    reconciliation unavailable when the denominator does not match;
  * report agreement of the simulated transition against the existing checked
    label as monitoring-grade only;
  * report concordance, precision, CI and topology flags as unavailable when
    no independent verdicts or masks are supplied, and never impute them;
  * emit module hash, rollback command and an empty zero-write audit diff.

Reuses existing helpers (no second harness). Read-only: writes no production
labels, code, datasets or runtime databases.
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
from experiments.cmr_checked_cluster_replay import (
    _apply_threshold,
    audit_diff,
    module_hash,
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
from experiments.cmr_low_cavity_threshold_replay import (
    ADEQUATE_CLUSTER,
    LOW_AREA_CLUSTER,
    _cluster_area_ratios,
    _derive_threshold,
)
from experiments.shadow_spatial_boundary import (
    CANDIDATE_DECISION,
    OUTSIDE_REASON,
    _checked_rank_core_set,
)

MODULE_VERSION = "cmr_checked_cohort_cluster_replay/1.0"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "checked_cohort_not_independent_holdout"

EXPECTED_CHECKED = 180

PENDING_BASELINE = 96

NO_WRITE_CLAUSE = (
    "Read-only S792 checked-cohort cluster replay. No production labels, code, "
    "datasets or runtime databases are written or mutated. Shadow triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_checked_cohort_cluster_replay.py "
    "tests/test_cmr_checked_cohort_cluster_replay.py"
)


def _reconciliation_table(
    n: int,
    simulated_checked: int,
    simulated_trash: int,
    simulated_no_transition: int,
) -> Dict[str, Any]:
    """Reconcile the three-cluster simulated total against the 96 pending
    baseline and the S783 108-case no_transition baseline.

    Reconciliation is only reported when the checked denominator matches the
    expected 180; otherwise it is marked unavailable and never imputed.
    """
    denominator_ok = n == EXPECTED_CHECKED
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    if denominator_ok:
        pending_delta: Optional[int] = simulated_no_transition - PENDING_BASELINE
        no_transition_delta: Optional[int] = (
            simulated_no_transition - NO_TRANSITION_BASELINE_S783
        )
        status = "ok"
    else:
        pending_delta = None
        no_transition_delta = None
        status = "unavailable_checked_denominator_mismatch"
    return {
        "checked_denominator": n,
        "expected_checked_denominator": EXPECTED_CHECKED,
        "checked_denominator_matches_expected": denominator_ok,
        "simulated_checked": simulated_checked,
        "simulated_trash": simulated_trash,
        "simulated_no_transition": simulated_no_transition,
        "simulated_total": simulated_total,
        "simulated_total_matches_denominator": simulated_total == n,
        "pending_baseline": PENDING_BASELINE,
        "no_transition_delta_vs_pending_baseline": pending_delta,
        "s783_no_transition_baseline": NO_TRANSITION_BASELINE_S783,
        "no_transition_delta_vs_s783": no_transition_delta,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "status": status,
        "note": (
            "S783/S784 baselines and the 96 pending baseline are aggregate "
            "cohort counts, not per-record truth; deltas are monitoring-grade "
            "reconciliation only."
        ),
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only S792 checked-cohort cluster replay."""
    records = data.get("records", []) if isinstance(data, dict) else []
    n = len(records)

    spatial = data.get("spatial_control") if isinstance(data, dict) else None
    spatial_records: List[Dict[str, Any]] = []
    if isinstance(spatial, dict):
        spatial_records = spatial.get("records", []) or []
    observed = _checked_rank_core_set(spatial_records)
    spatial_index: Dict[Any, Dict[str, Any]] = {
        r.get("case_id"): r for r in spatial_records
    }

    cluster_ratios = _cluster_area_ratios(records)
    threshold = _derive_threshold(
        cluster_ratios.get(ADEQUATE_CLUSTER, []),
        cluster_ratios.get(LOW_AREA_CLUSTER, []),
    )

    counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
    transitions: Dict[str, Dict[str, int]] = {
        c: {"checked": 0, "trash": 0, "no_transition": 0} for c in CLUSTERS
    }
    suppression_reasons: Dict[str, int] = {
        ROUTED_REASON: 0,
        "unrouted": 0,
        "spatial_gate_suppressed": 0,
    }
    topology_unavailable: Dict[str, int] = {f: 0 for f in TOPOLOGY_FIELDS}
    cases: Dict[str, Any] = {}
    unrouted_ids: List[str] = []
    missing_measurement: List[str] = []
    spatial_candidate_count = 0
    spatial_suppressed_count = 0

    simulated_checked = 0
    simulated_trash = 0
    simulated_no_transition = 0
    agreement_count = 0
    agreement_denominator = 0

    for rec in records:
        cid = rec.get("case_id")
        area_ratio = _area_ratio(rec)
        cavity = rec.get("before_cavity")
        cluster = _classify(area_ratio, cavity)
        if area_ratio is None or cavity is None:
            missing_measurement.append(cid)

        topology = _topology_flags(rec)
        for field, info in topology.items():
            if info["value"] is None:
                topology_unavailable[field] += 1

        spatial_out = _spatial_decision(cid, spatial_index, observed)
        spatial_hit = spatial_out.get("decision") == CANDIDATE_DECISION
        if spatial_hit:
            spatial_candidate_count += 1

        if _is_unrouted(cluster, cavity):
            unrouted_ids.append(cid)
            transition = _simulate_transition(cluster, area_ratio, cavity)
            routing = "unrouted"
        elif spatial_hit:
            spatial_suppressed_count += 1
            transition = "no_transition"
            routing = "spatial_gate_suppressed"
        else:
            counts[cluster] += 1
            transition = _apply_threshold(cluster, area_ratio, threshold)
            transitions[cluster][transition] += 1
            routing = ROUTED_REASON

        if transition == "checked":
            simulated_checked += 1
        elif transition == "trash":
            simulated_trash += 1
        else:
            simulated_no_transition += 1

        suppression_reasons[routing] = suppression_reasons.get(routing, 0) + 1

        # Monitoring-grade agreement against the existing checked label. The
        # checked label is not an independent verdict, so this is not
        # concordance and is never treated as precision.
        checked_label = rec.get("checked_label")
        if checked_label is not None:
            agreement_denominator += 1
            if checked_label == transition:
                agreement_count += 1

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "topology": topology,
            "spatial_rule": spatial_out,
            "spatial_gate_suppressed": spatial_hit,
            "routing": routing,
            "simulated_transition": transition,
            "checked_label": checked_label,
        }

    classified_total = sum(counts.values())
    simulated_total = simulated_checked + simulated_trash + simulated_no_transition
    unrouted_count = len(unrouted_ids)

    if agreement_denominator > 0:
        agreement_rate: Optional[float] = agreement_count / agreement_denominator
        agreement_status = "monitoring_grade_only"
    else:
        agreement_rate = None
        agreement_status = "unavailable_no_checked_labels"

    reconciliation = _reconciliation_table(
        n, simulated_checked, simulated_trash, simulated_no_transition
    )

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_checked_cohort_cluster_replay",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are checked candidates, not verified truth; "
            "agreement is monitoring-grade and concordance, precision and CI "
            "against independent labels stay unavailable."
        ),
        "rule": {
            "area_ratio_min": AREA_RATIO_MIN,
            "cavity_max": CAVITY_MAX,
            "clusters": list(CLUSTERS),
            "suppression_conditions": list(SUPPRESSION_CONDITIONS),
            "spatial_candidate_decision": CANDIDATE_DECISION,
            "spatial_outside_reason": OUTSIDE_REASON,
        },
        "n_records": n,
        "expected_checked": EXPECTED_CHECKED,
        "checked_denominator_matches_expected": n == EXPECTED_CHECKED,
        "cluster_counts": {
            c: {"count": counts[c], "denominator": n} for c in CLUSTERS
        },
        "classified_total": classified_total,
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
        "simulated_no_transition_total": simulated_no_transition,
        "simulated_total": simulated_total,
        "simulated_total_matches_checked": simulated_total == n,
        "suppression_reasons": suppression_reasons,
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
        "unrouted_count": unrouted_count,
        "unrouted_cases": unrouted_ids,
        "threshold": {
            "adequate_cluster": ADEQUATE_CLUSTER,
            "low_area_cluster": LOW_AREA_CLUSTER,
            "axis": "area_ratio",
            "value": threshold["threshold"],
            "margin": threshold["margin"],
            "status": threshold["status"],
            "adequate_min": threshold["adequate_min"],
            "low_area_max": threshold["low_area_max"],
        },
        "agreement_vs_checked_labels": {
            "agreement_count": agreement_count,
            "denominator": agreement_denominator,
            "rate": agreement_rate,
            "status": agreement_status,
            "note": (
                "Checked labels are not independent verdicts; this is "
                "monitoring-grade agreement only."
            ),
        },
        "reconciliation": reconciliation,
        "pending_baseline": PENDING_BASELINE,
        "no_transition_reconciliation_delta_vs_pending_baseline": reconciliation[
            "no_transition_delta_vs_pending_baseline"
        ],
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
        "no_transition_reconciliation_delta_vs_s783": reconciliation[
            "no_transition_delta_vs_s783"
        ],
        "no_transition_reconciliation_status": reconciliation["status"],
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "topology_unavailable_counts": {
            f: {"count": topology_unavailable[f], "denominator": n}
            for f in TOPOLOGY_FIELDS
        },
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
        "missing_measurement_cases": missing_measurement,
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
            "topology_flags": "no masks, frame geometry or centroid series supplied",
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
