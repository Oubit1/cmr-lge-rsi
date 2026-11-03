"""Read-only S789 per-cluster topology-safe expansion candidate scan.

Scope (shadow_triage_only, zero production writes):
  * reuse the existing S788 checked-cluster replay helpers
    (cmr_checked_cluster_replay) plus the S784 discriminator scan, S785
    threshold replay, S782/S783 cluster classifier and S777 spatial gate that
    S788 already imports;
  * scan the 20 stratified pending CMR volumes and, per cluster, count
    topology-safe expand_undersegmented_region candidates: routed volumes that
    are not spatial-gate suppressed, not unrouted, and carry a usable
    area_ratio/cavity measurement;
  * reconcile the three-cluster candidate total against the 20-volume
    stratified denominator and the 106 pending baseline;
  * report component/hole/border deltas as unavailable when no masks are
    supplied, and never impute concordance, precision or CI;
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

MODULE_VERSION = "cmr_small_area_expansion_gate_scan/1.0"

ROUTING_MODE = "shadow_triage_only"

ROUTED_REASON = "routed"

COHORT_LABEL = "stratified_pending_volumes_not_independent_holdout"

EXPECTED_PENDING = 20

PENDING_BASELINE = 106

NO_WRITE_CLAUSE = (
    "Read-only S789 small-area expansion gate scan. No production labels, "
    "code, datasets or runtime databases are written or mutated. Shadow "
    "triage only."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_small_area_expansion_gate_scan.py "
    "tests/test_cmr_small_area_expansion_gate_scan.py"
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


def _reconciliation_table(
    n: int,
    candidate_total: int,
    per_cluster: Dict[str, int],
) -> Dict[str, Any]:
    """Reconcile the three-cluster candidate total against the 20-volume
    stratified denominator and the 106 pending baseline.

    Reconciliation is only reported when the stratified denominator matches the
    expected 20; otherwise it is marked unavailable and never imputed.
    """
    denominator_ok = n == EXPECTED_PENDING
    if denominator_ok:
        delta: Optional[int] = candidate_total - PENDING_BASELINE
        status = "ok"
    else:
        delta = None
        status = "unavailable_stratified_denominator_mismatch"
    return {
        "stratified_denominator": n,
        "expected_stratified_denominator": EXPECTED_PENDING,
        "stratified_denominator_matches_expected": denominator_ok,
        "candidate_total": candidate_total,
        "per_cluster_candidate_counts": dict(per_cluster),
        "pending_baseline": PENDING_BASELINE,
        "candidate_delta_vs_pending_baseline": delta,
        "s783_no_transition_baseline": NO_TRANSITION_BASELINE_S783,
        "s784_adequate_area_low_cavity_denominator": S784_EXPECTED_PENDING,
        "s784_cohort": S784_COHORT_LABEL,
        "status": status,
        "note": (
            "Pending queue is not an independent holdout; the 106 pending "
            "baseline and S783/S784 counts are aggregate cohort numbers, not "
            "per-record truth. Deltas are monitoring-grade reconciliation only."
        ),
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only S789 topology-safe expansion candidate scan."""
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
    candidate_counts: Dict[str, int] = {c: 0 for c in CLUSTERS}
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
    candidate_total = 0

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
            routing = "unrouted"
            candidate = False
        elif spatial_hit:
            spatial_suppressed_count += 1
            routing = "spatial_gate_suppressed"
            candidate = False
        else:
            counts[cluster] += 1
            routing = ROUTED_REASON
            candidate = area_ratio is not None and cavity is not None
            if candidate:
                candidate_counts[cluster] += 1
                candidate_total += 1

        suppression_reasons[routing] = suppression_reasons.get(routing, 0) + 1

        cases[cid] = {
            "case_id": cid,
            "area_ratio": area_ratio,
            "cavity": cavity,
            "cluster": cluster,
            "topology": topology,
            "spatial_rule": spatial_out,
            "spatial_gate_suppressed": spatial_hit,
            "routing": routing,
            "expansion_candidate": candidate,
        }

    classified_total = sum(counts.values())
    reconciliation = _reconciliation_table(n, candidate_total, candidate_counts)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_small_area_expansion_gate_scan",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Supplied volumes are stratified pending candidates, not verified "
            "truth; candidate counts are monitoring-grade and concordance, "
            "precision and CI against independent labels stay unavailable."
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
        "expected_pending": EXPECTED_PENDING,
        "stratified_denominator_matches_expected": n == EXPECTED_PENDING,
        "cluster_counts": {
            c: {"count": counts[c], "denominator": n} for c in CLUSTERS
        },
        "classified_total": classified_total,
        "expansion_candidate_counts": {
            c: {"count": candidate_counts[c], "denominator": counts[c]} for c in CLUSTERS
        },
        "expansion_candidate_total": candidate_total,
        "expansion_candidate_total_matches_denominator": candidate_total == n,
        "suppression_reasons": suppression_reasons,
        "spatial_gate": {
            "candidate_count": spatial_candidate_count,
            "suppressed_count": spatial_suppressed_count,
            "denominator": n,
        },
        "unrouted_count": len(unrouted_ids),
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
        "reconciliation": reconciliation,
        "pending_baseline": PENDING_BASELINE,
        "candidate_reconciliation_delta_vs_pending_baseline": reconciliation[
            "candidate_delta_vs_pending_baseline"
        ],
        "candidate_reconciliation_status": reconciliation["status"],
        "no_transition_baseline_s783": NO_TRANSITION_BASELINE_S783,
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
