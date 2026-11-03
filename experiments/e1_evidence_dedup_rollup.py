"""
E1 Evidence Dedup Rollup & Cache Reuse Guard
Implements run-9 E1 / run-13 A3 deduplicated evidence rollup.
Read-only enforcement module for CMR/LGE QC orchestrator.
"""
from typing import List, Dict, Any, Set, Tuple

# Configuration constants
HIGH_VOLUME_MODALITIES = frozenset({"CMR", "LGE"})
GUARDED_RUN_CLASSES = frozenset({"run-3", "run-4", "run-5", "run-6", "run-9"})
VALID_DIGEST_RUNS = frozenset({"run-4", "run-5", "run-9"})
SEVERITY_TIER_THRESHOLDS = {"low": 0, "medium": 7, "high": 15, "critical": 30}


def deduplicate_evidence(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emits exactly one record per (case_id, modality).
    Merges issue_type into a sorted set.
    Collects per-issue slice-index lists, sorted.
    """
    if not records:
        return []

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rec in records:
        case_id = rec.get("case_id")
        modality = rec.get("modality")
        if case_id is None or modality is None:
            continue

        key = (case_id, modality)
        if key not in grouped:
            grouped[key] = {
                "case_id": case_id,
                "modality": modality,
                "issue_types": set(),
                "slice_indices_by_issue": {},
            }

        entry = grouped[key]
        issue = rec.get("issue_type", "unknown")
        entry["issue_types"].add(issue)

        slice_idx = rec.get("slice_index")
        if slice_idx is not None:
            entry["slice_indices_by_issue"].setdefault(issue, []).append(slice_idx)

    result = []
    for data in grouped.values():
        # Ensure sorted slice indices per issue
        sorted_slices = {k: sorted(v) for k, v in data["slice_indices_by_issue"].items()}
        result.append({
            "case_id": data["case_id"],
            "modality": data["modality"],
            "issue_types": sorted(data["issue_types"]),
            "slice_indices_by_issue": sorted_slices,
            "_raw_record_count": sum(len(v) for v in sorted_slices.values()),
        })

    return result


def roll_up_high_volume_flags(rolled_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rolls up high-volume CMR/LGE flags per case.
    Adds: count, rate, worst-slice index, severity tier.
    Keeps every flagged slice index addressable via 'addressable_slices'.
    """
    for rec in rolled_records:
        if rec["modality"] not in HIGH_VOLUME_MODALITIES:
            continue

        all_slices = []
        for slices in rec["slice_indices_by_issue"].values():
            all_slices.extend(slices)

        all_slices.sort()
        worst_slice = all_slices[-1] if all_slices else None
        count = len(all_slices)

        # Rate calculation: normalized count per 10 slices (configurable baseline)
        rate = count / 10.0

        # Determine severity tier based on count thresholds
        tier = "low"
        for t in ["low", "medium", "high", "critical"]:
            if count >= SEVERITY_TIER_THRESHOLDS[t]:
                tier = t

        rec["flag_count"] = count
        rec["rollup_rate"] = round(rate, 2)
        rec["worst_slice_index"] = worst_slice
        rec["severity_tier"] = tier
        rec["addressable_slices"] = all_slices

    return rolled_records


class CacheReuseGuard:
    """
    Hash-gated cache-reuse guard.
    Blocks new extractor construction when a valid run-4/5/9 digest exists.
    Flags re-proposals of run-3/4/5/6/9 experiment classes.
    """

    def __init__(self):
        self._digest_registry: Dict[str, str] = {}
        self._flagged_proposals: Set[str] = set()

    def register_valid_digest(self, run_id: str, digest_hash: str) -> None:
        if run_id in VALID_DIGEST_RUNS:
            self._digest_registry[run_id] = digest_hash

    def check_and_block_extractor(self, run_id: str, proposed_digest: str) -> bool:
        """
        Returns True if construction is BLOCKED (cache hit).
        Returns False if construction is ALLOWED.
        """
        if run_id in VALID_DIGEST_RUNS:
            existing = self._digest_registry.get(run_id)
            if existing is not None and existing == proposed_digest:
                return True  # Blocked
        return False

    def evaluate_proposal(self, experiment_class: str) -> Dict[str, Any]:
        """
        Evaluates an experiment class proposal against guarded runs.
        Returns status dict with 'blocked', 'flagged', 'reason'.
        """
        is_guarded = experiment_class in GUARDED_RUN_CLASSES
        status = {
            "experiment_class": experiment_class,
            "blocked": False,
            "flagged": False,
            "reason": "",
        }

        if is_guarded:
            status["flagged"] = True
            self._flagged_proposals.add(experiment_class)
            status["reason"] = f"Re-proposal of guarded run-class {experiment_class} detected."

        return status
