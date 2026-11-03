"""
Evidence Rollup & Hash-Gated Cache-Reuse Guard
Enforces deduplicated evidence rollup and prevents redundant extractor constructions.
Read-only module; no production writes or label edits.
"""

from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
VALID_DIGESTS_RUNS = {"run-4", "run-5", "run-9", "run-13", "run-14", "run-15", "run-16"}
BLOCKED_CLASSES_SET = {"run-3", "run-4", "run-5", "run-6", "run-9"}


def _compute_severity_tier(severities: List[str]) -> str:
    if not severities:
        return "NONE"
    max_val = max(SEVERITY_ORDER.get(s.upper(), 0) for s in severities)
    for tier, val in SEVERITY_ORDER.items():
        if val == max_val:
            return tier
    return "UNKNOWN"


def rollup_evidence(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emits exactly one record per (case_id, modality) with merged issue_type set
    and per-issue slice-index lists. Rolls up high-volume flags.
    """
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for rec in records:
        key = (rec["case_id"], rec["modality"])
        groups[key].append(rec)

    results = []
    for (case_id, modality), recs in groups.items():
        merged_issues: Set[str] = set()
        slice_indices_by_issue: Dict[str, List[int]] = defaultdict(list)
        all_severities: List[str] = []
        all_slice_indices: List[int] = []

        for r in recs:
            issue = r.get("issue_type", "UNKNOWN")
            slice_idx = r.get("slice_index")
            severity = r.get("severity", "LOW")

            merged_issues.add(issue)
            if slice_idx is not None:
                slice_indices_by_issue[issue].append(slice_idx)
                all_slice_indices.append(slice_idx)
            all_severities.append(severity)

        worst_slice = max(all_slice_indices) if all_slice_indices else None
        severity_tier = _compute_severity_tier(all_severities)
        count = len(recs)
        unique_slices = len(set(all_slice_indices))
        rate = round(count / max(1, unique_slices), 2)

        results.append({
            "case_id": case_id,
            "modality": modality,
            "merged_issue_types": sorted(list(merged_issues)),
            "slice_indices_by_issue": {k: sorted(v) for k, v in slice_indices_by_issue.items()},
            "rollup_metrics": {
                "count": count,
                "rate": rate,
                "worst_slice_index": worst_slice,
                "severity_tier": severity_tier
            }
        })
    return results


def validate_extractor_guard(proposed_digest: str, valid_digests: Set[str], proposed_class: str) -> Dict[str, Any]:
    """
    Blocks new extractor construction when a valid run-4/5/9/13/14/15/16 digest exists.
    Flags re-proposal of run-3/4/5/6/9 experiment classes.
    """
    is_blocked = False
    mismatch_report = ""
    reproposal_flagged = False

    if proposed_digest in valid_digests:
        is_blocked = True
        mismatch_report = f"Cache-reuse guard triggered: valid digest '{proposed_digest}' found. New extractor construction blocked."

    if proposed_class in BLOCKED_CLASSES_SET:
        reproposal_flagged = True
        if mismatch_report:
            mismatch_report += f" Additionally, re-proposal of experiment class '{proposed_class}' is flagged."
        else:
            mismatch_report = f"Re-proposal guard triggered: experiment class '{proposed_class}' is in the deprecated/redundant set."

    return {
        "guard_status": "BLOCKED" if is_blocked else "ALLOWED",
        "mismatch_report": mismatch_report,
        "reproposal_flagged": reproposal_flagged,
        "proposed_digest": proposed_digest,
        "proposed_class": proposed_class
    }


def process_qc_payload(raw_records: List[Dict[str, Any]], valid_digests: Set[str], proposed_digest: str, proposed_class: str) -> Dict[str, Any]:
    """
    Orchestrates rollup and guard checks. Returns unified payload.
    """
    rolled_up = rollup_evidence(raw_records)
    guard_result = validate_extractor_guard(proposed_digest, valid_digests, proposed_class)

    return {
        "rolled_up_records": rolled_up,
        "guard_check": guard_result
    }
