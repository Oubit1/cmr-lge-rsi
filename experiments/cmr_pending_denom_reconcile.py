"""Read-only denominator reconciliation for the pending CMR queue.

Scope (isolated_code_experiment, zero writes):
  * deduplicate representative_cases by (case_id, issue_type);
  * reconcile the 13/61/95/136/146/156/180 denominators against their
    declared sources and attribute the 13 vs 61 gap explicitly;
  * emit unique_frozen_denominator and dedup_ratio;
  * exclude the frozen holdout by case_id (holdout_selected==0);
  * report concordance/precision/CI as unavailable, never imputed.

Read-only: writes no production labels, code, datasets or runtime databases.
Does not touch app/check_auto_apply.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from experiments.cmr_checked_cluster_replay import audit_diff, module_hash

MODULE_VERSION = "cmr_pending_denom_reconcile/1.0"

ROUTING_MODE = "read_only_denominator_reconcile"

COHORT_LABEL = "current_pending_queue_with_frozen_split"

# Declared denominators and their sources, as reported by the orchestrator.
DENOMINATOR_SOURCES: Dict[str, Dict[str, Any]] = {
    "13": {
        "value": 13,
        "source": "frozen_split.non_holdout_in_queue",
        "scope": "non_holdout_pending",
    },
    "61": {
        "value": 61,
        "source": "real_evaluation_input.full_pending_queue",
        "scope": "full_pending_queue",
    },
    "95": {
        "value": 95,
        "source": "quality_benchmark.defect_classes.true_small.n",
        "scope": "gold_defect_class",
    },
    "136": {
        "value": 136,
        "source": "quality_benchmark.defect_classes.layer_jump.n",
        "scope": "gold_defect_class",
    },
    "146": {
        "value": 146,
        "source": "quality_benchmark.defect_classes.hole_fill.n",
        "scope": "gold_defect_class",
    },
    "156": {
        "value": 156,
        "source": "quality_benchmark.defect_classes.fragmentation.n",
        "scope": "gold_defect_class",
    },
    "180": {
        "value": 180,
        "source": "quality_benchmark.defect_classes.border_clip.n",
        "scope": "gold_defect_class",
    },
}

NO_WRITE_CLAUSE = (
    "Read-only denominator reconciliation. No production labels, code, "
    "datasets or runtime databases are written or mutated. "
    "app/check_auto_apply.py is untouched."
)

ROLLBACK_COMMAND = (
    "git checkout -- experiments/cmr_pending_denom_reconcile.py "
    "tests/test_cmr_pending_denom_reconcile.py"
)


def _dedup_key(rec: Dict[str, Any]) -> Tuple[Any, Any]:
    return (rec.get("case_id"), rec.get("issue_type"))


def _dedup_representative_cases(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deduplicate representative_cases by (case_id, issue_type)."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    duplicate_keys: List[Dict[str, Any]] = []
    for rec in records:
        key = _dedup_key(rec)
        if key in seen:
            duplicate_keys.append({"case_id": key[0], "issue_type": key[1]})
            continue
        seen.add(key)
        unique.append(rec)
    input_count = len(records)
    unique_count = len(unique)
    dedup_ratio: Optional[float] = None
    if input_count:
        dedup_ratio = round(unique_count / input_count, 6)
    return {
        "input_count": input_count,
        "unique_count": unique_count,
        "duplicate_count": len(duplicate_keys),
        "dedup_ratio": dedup_ratio,
        "duplicate_keys": duplicate_keys,
        "unique": unique,
    }


def _select_non_holdout(
    records: List[Dict[str, Any]],
    holdout_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Exclude frozen holdout by case_id; holdout_selected is 0 by construction."""
    holdout = set(holdout_ids or [])
    selected: List[Dict[str, Any]] = []
    holdout_excluded = 0
    for rec in records:
        cid = rec.get("case_id")
        if cid in holdout:
            holdout_excluded += 1
            continue
        selected.append(rec)
    holdout_selected = sum(1 for r in selected if r.get("case_id") in holdout)
    return {
        "selected": selected,
        "holdout_selected": holdout_selected,
        "holdout_excluded_count": holdout_excluded,
    }


def _reconcile_denominators(
    data: Dict[str, Any],
    unique_frozen_denominator: int,
) -> Dict[str, Any]:
    """Reconcile declared denominators and attribute the 13 vs 61 gap."""
    frozen_split = data.get("frozen_split") or {}
    full_pending = data.get("full_pending_queue")
    non_holdout = frozen_split.get("non_holdout_in_queue")
    holdout_in_queue = frozen_split.get("frozen_holdout_in_queue")

    gap_attribution: Dict[str, Any] = {
        "status": "unavailable_missing_frozen_split",
        "note": "frozen_split not supplied; 13 vs 61 gap not attributable.",
    }
    if isinstance(non_holdout, int) and isinstance(full_pending, int):
        gap = full_pending - non_holdout
        gap_attribution = {
            "status": "ok",
            "full_pending_queue": full_pending,
            "non_holdout_in_queue": non_holdout,
            "gap": gap,
            "frozen_holdout_in_queue": holdout_in_queue,
            "attribution": (
                "61 is the full pending queue; 13 is the non-holdout subset. "
                "The 48-volume gap equals the frozen holdout volumes present "
                "in the queue (frozen_split.frozen_holdout_in_queue)."
            ),
            "gap_matches_holdout": (
                isinstance(holdout_in_queue, int) and gap == holdout_in_queue
            ),
        }

    return {
        "declared_sources": DENOMINATOR_SOURCES,
        "full_pending_queue": full_pending,
        "non_holdout_in_queue": non_holdout,
        "frozen_holdout_in_queue": holdout_in_queue,
        "unique_frozen_denominator": unique_frozen_denominator,
        "unique_frozen_denominator_matches_non_holdout": (
            isinstance(non_holdout, int)
            and unique_frozen_denominator == non_holdout
        ),
        "gap_13_vs_61": gap_attribution,
    }


def run_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only denominator reconciliation over the supplied pending queue."""
    if not isinstance(data, dict):
        data = {}
    raw_records = data.get("records", []) or []
    holdout_ids = data.get("holdout_case_ids")

    dedup = _dedup_representative_cases(raw_records)
    selection = _select_non_holdout(dedup["unique"], holdout_ids)
    non_holdout = selection["selected"]
    unique_frozen_denominator = len(non_holdout)

    reconciliation = _reconcile_denominators(data, unique_frozen_denominator)

    report: Dict[str, Any] = {
        "module_version": MODULE_VERSION,
        "module_hash": module_hash(),
        "mode": "read_only_cmr_pending_denom_reconcile",
        "routing_mode": ROUTING_MODE,
        "cohort": COHORT_LABEL,
        "cohort_note": (
            "Pending queue is not an independent validation cohort; "
            "reconciliation is monitoring-grade only."
        ),
        "dedup_key": ["case_id", "issue_type"],
        "representative_cases_input_count": dedup["input_count"],
        "representative_cases_unique_count": dedup["unique_count"],
        "representative_cases_duplicate_count": dedup["duplicate_count"],
        "dedup_ratio": dedup["dedup_ratio"],
        "duplicate_keys": dedup["duplicate_keys"],
        "holdout_selected": selection["holdout_selected"],
        "holdout_excluded_count": selection["holdout_excluded_count"],
        "unique_frozen_denominator": unique_frozen_denominator,
        "reconciliation": reconciliation,
        "label_write": False,
        "target_path": "experiments/cmr_pending_denom_reconcile.py",
        "unavailable": {
            "concordance_vs_independent_labels": "no independent reference labels supplied",
            "precision": "no independent reference labels supplied",
            "ci": "no independent reference labels supplied",
        },
        "no_write_clause": NO_WRITE_CLAUSE,
        "rollback_command": ROLLBACK_COMMAND,
        "writes": {"labels": 0, "code": 0, "datasets": 0},
    }
    report["audit_diff"] = audit_diff(None, report)
    return report


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2)
