"""Bounded read-only paired shadow audit for cmr_myo_small_area.

Shadow-only audit: pairs positive cases with CMR controls by slice/phase
metadata, reuses cached myocardial-area features from runs 4/5/9/13-18,
falls back to the run-9 B1 extractor on cache-hash mismatch (each fallback
is registered in a ledger), and reports the acceptance metrics:

  * audit completion rate (target >= 0.95)
  * difference in the fraction with >=1 support flag (positives - controls)
  * separation of cached myocardial-area medians (positives - controls)

Read-only contract: never mutates labels, rules, thresholds, caches, or
patient data; all inputs are treated as immutable.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ALLOWED_CACHE_RUNS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)
FALLBACK_RUN: int = 9
FALLBACK_EXTRACTOR: str = "B1"
MIN_COMPLETION_RATE: float = 0.95
POSITIVE: str = "positive"
CONTROL: str = "control"


@dataclass(frozen=True)
class CaseRecord:
    """Immutable per-case audit input (one slice/phase row)."""

    case_id: str
    group: str  # POSITIVE or CONTROL
    slice_index: int
    phase: str
    source_run: int
    cached_area: Optional[float]
    cache_hash: str
    support_flags: Tuple[str, ...] = ()


def expected_cache_hash(case_id: str, source_run: int, area: float) -> str:
    """Deterministic provenance hash for a cached area value."""
    payload = "{}|run{}|{:.6f}".format(case_id, source_run, area).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_by_metadata(
    positives: Sequence[CaseRecord], controls: Sequence[CaseRecord]
) -> List[Dict[str, object]]:
    """Greedy one-to-one pairing on (slice_index, phase), then slice only."""
    pairs: List[Dict[str, object]] = []
    used: set = set()
    for ctrl in controls:
        match: Optional[int] = None
        for idx, pos in enumerate(positives):
            if idx in used:
                continue
            if (pos.slice_index, pos.phase) == (ctrl.slice_index, ctrl.phase):
                match = idx
                break
        if match is None:
            for idx, pos in enumerate(positives):
                if idx not in used and pos.slice_index == ctrl.slice_index:
                    match = idx
                    break
        if match is not None:
            used.add(match)
            pairs.append(
                {
                    "control_id": ctrl.case_id,
                    "positive_id": positives[match].case_id,
                    "slice_index": ctrl.slice_index,
                    "phase": ctrl.phase,
                }
            )
    return pairs


def _resolve_area(
    case: CaseRecord,
    fallback_extractor: Optional[Callable[[CaseRecord], float]],
    ledger: List[Dict[str, object]],
) -> Tuple[Optional[float], str]:
    """Return (area, provenance); register any fallback in the ledger."""
    reason: Optional[str] = None
    if case.cached_area is None:
        reason = "missing_cache"
    elif case.source_run not in ALLOWED_CACHE_RUNS:
        reason = "run_not_allowed"
    elif case.cache_hash != expected_cache_hash(
        case.case_id, case.source_run, case.cached_area
    ):
        reason = "hash_mismatch"
    if reason is None:
        return case.cached_area, "cache:run{}".format(case.source_run)
    if fallback_extractor is None:
        ledger.append(
            {
                "case_id": case.case_id,
                "source_run": case.source_run,
                "reason": reason,
                "fallback": "unavailable",
            }
        )
        return None, "failed"
    area = float(fallback_extractor(case))
    ledger.append(
        {
            "case_id": case.case_id,
            "source_run": case.source_run,
            "reason": reason,
            "fallback": "run{}/{}".format(FALLBACK_RUN, FALLBACK_EXTRACTOR),
        }
    )
    return area, "fallback:run{}/{}".format(FALLBACK_RUN, FALLBACK_EXTRACTOR)


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def run_audit(
    positives: Sequence[CaseRecord],
    controls: Sequence[CaseRecord],
    fallback_extractor: Optional[Callable[[CaseRecord], float]] = None,
) -> Dict[str, object]:
    """Execute the bounded read-only shadow audit and return a report.

    fallback_extractor: the run-9 B1 extractor, injected by the caller; it
    is the only permitted re-extraction path and is invoked solely on cache
    hash mismatch, missing cache, or a disallowed source run.
    """
    ledger: List[Dict[str, object]] = []
    areas: Dict[str, List[float]] = {POSITIVE: [], CONTROL: []}
    flag_hits: Dict[str, int] = {POSITIVE: 0, CONTROL: 0}
    counts: Dict[str, int] = {POSITIVE: 0, CONTROL: 0}
    audited = 0
    total = len(positives) + len(controls)
    for group, cases in ((POSITIVE, positives), (CONTROL, controls)):
        for case in cases:
            counts[group] += 1
            area, _provenance = _resolve_area(case, fallback_extractor, ledger)
            if area is None:
                continue
            audited += 1
            areas[group].append(area)
            if case.support_flags:
                flag_hits[group] += 1
    completion = (audited / total) if total else 0.0
    pos_frac = flag_hits[POSITIVE] / counts[POSITIVE] if counts[POSITIVE] else 0.0
    ctl_frac = flag_hits[CONTROL] / counts[CONTROL] if counts[CONTROL] else 0.0
    pos_med = _median(areas[POSITIVE])
    ctl_med = _median(areas[CONTROL])
    separation: Optional[float] = None
    if pos_med is not None and ctl_med is not None:
        separation = pos_med - ctl_med
    pairs = pair_by_metadata(positives, controls)
    return {
        "audit": "cmr_myo_small_area_paired_shadow",
        "read_only": True,
        "total_cases": total,
        "audited_cases": audited,
        "completion_rate": round(completion, 6),
        "completion_target": MIN_COMPLETION_RATE,
        "completion_target_met": completion >= MIN_COMPLETION_RATE,
        "pairs": pairs,
        "pair_count": len(pairs),
        "positive_support_flag_fraction": round(pos_frac, 6),
        "control_support_flag_fraction": round(ctl_frac, 6),
        "support_flag_fraction_difference": round(pos_frac - ctl_frac, 6),
        "positive_area_median": pos_med,
        "control_area_median": ctl_med,
        "area_median_separation": separation,
        "fallback_count": sum(1 for e in ledger if e["fallback"] != "unavailable"),
        "ledger": ledger,
    }
