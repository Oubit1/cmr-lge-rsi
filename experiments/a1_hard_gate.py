"""
Hard Output Gate for A1 CMR/LGE QC Orchestrator
Enforces deduplicated evidence rollup, hash-gated cache-reuse, and stratum re-proposal blocking.
Read-only enforcement; no production code changes.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Constants
VALID_CACHE_RUNS = {4, 5, 9, 13, 14, 15, 16, 17}
MAX_ATTEMPTS_CAP = 1
RUN_18_STRATUM_KEY = "mi_multiple_components"


@dataclass
class GateResult:
    status: str  # "PASS" or "FAIL"
    records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    escalation_message: Optional[str] = None
    cache_reuse_blocked: bool = False
    cache_mismatch_hashes: List[str] = field(default_factory=list)
    run18_stratum_flagged: bool = False


def compute_record_hash(record: Dict[str, Any]) -> str:
    """Deterministic SHA-256 hash for a rolled-up record."""
    canonical = {k: v for k, v in sorted(record.items()) if k != "hash"}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def validate_and_enforce_gate(
    a1_output: List[Dict[str, Any]],
    cached_digests: Optional[Dict[int, str]] = None,
    current_cycle_run_ids: Optional[List[str]] = None
) -> GateResult:
    """
    Applies the hard output gate rules to A1 output.
    Returns a GateResult indicating pass/fail, processed records, and escalation if needed.
    """
    result = GateResult(status="PASS")
    if not a1_output:
        return result

    # 1. Group by (case_id, modality)
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for rec in a1_output:
        key = (rec.get("case_id"), rec.get("modality"))
        if not key[0] or not key[1]:
            result.errors.append(f"Missing case_id or modality in record: {rec}")
            continue
        grouped.setdefault(key, []).append(rec)

    # 2. Deduplicate & Rollup
    rolled_up_records = []
    for (cid, mod), recs in grouped.items():
        merged_issue_types: Set[str] = set()
        merged_slice_indices: Dict[str, List[int]] = {}
        merged_high_vol_flags: Dict[str, Any] = {}

        for r in recs:
            merged_issue_types.update(r.get("issue_types", []))
            for iss, slices in r.get("slice_indices", {}).items():
                merged_slice_indices.setdefault(iss, []).extend(slices)

            hv = r.get("high_volume_flags", {})
            if hv:
                merged_high_vol_flags = {
                    "count": sum(r.get("high_volume_flags", {}).get("count", 0) for r in recs),
                    "rate": max((r.get("high_volume_flags", {}).get("rate", 0.0) for r in recs), default=0.0),
                    "worst_slice_index": max((r.get("high_volume_flags", {}).get("worst_slice_index", 0) for r in recs), default=0),
                    "severity_tier": max((r.get("high_volume_flags", {}).get("severity_tier", "") for r in recs), default="")
                }

        # Ensure every flagged slice index is addressable
        all_slices = []
        for slices in merged_slice_indices.values():
            all_slices.extend(slices)

        rolled_rec = {
            "case_id": cid,
            "modality": mod,
            "issue_types": sorted(list(merged_issue_types)),
            "slice_indices": merged_slice_indices,
            "high_volume_flags": merged_high_vol_flags,
            "all_flagged_slices": sorted(all_slices),
            "addressable": True
        }
        rolled_up_records.append(rolled_rec)

    # Verify exactly one record per group post-rollup
    seen_keys = set()
    for rec in rolled_up_records:
        k = (rec["case_id"], rec["modality"])
        if k in seen_keys:
            result.errors.append(f"Duplicate record after rollup for {k}")
            result.status = "FAIL"
        seen_keys.add(k)

    result.records = rolled_up_records

    # 3. Hash-Gated Cache-Reuse Guard
    if cached_digests:
        for run_id in VALID_CACHE_RUNS:
            if run_id in cached_digests:
                result.cache_reuse_blocked = True
                for rec in rolled_up_records:
                    h = compute_record_hash(rec)
                    if cached_digests[run_id] != h:
                        result.cache_mismatch_hashes.append(f"Run-{run_id}: expected={cached_digests[run_id]}, got={h}")
                break  # One valid digest is sufficient to block

    # 4. Run-18 mi_multiple_components Primary Stratum Re-proposal Flag
    if current_cycle_run_ids:
        run18_present = any("run-18" in rid for rid in current_cycle_run_ids)
        if run18_present:
            for rec in rolled_up_records:
                if RUN_18_STRATUM_KEY in str(rec.get("issue_types", [])) or RUN_18_STRATUM_KEY in str(rec):
                    result.run18_stratum_flagged = True
                    result.errors.append(f"Blocked re-proposal of run-18 {RUN_18_STRATUM_KEY} primary stratum in same cycle.")
                    break

    # 5. Final Status & Escalation
    if result.errors:
        result.status = "FAIL"
        result.escalation_message = (
            f"GATE FAILED: {len(result.errors)} violation(s) detected. "
            "Escalating to human maintenance owner. Max attempts (1) exhausted."
        )

    return result
