"""Bounded read-only shadow audit for the LGE myo_topology pending class.

This module implements one bounded audit slice: exactly 50 detector-positive
candidates and 10 LGE negative controls.  Detector outputs are read from
cached artifacts produced by runs 4/5/9/13-18.  A cached entry is accepted
only when its run is in the allow-list and its SHA256 digest matches the
expected digest registry.  Cases without a valid cached entry fall back to
the run-9 B1 extractor and are reported.  The module never mutates labels,
rules or thresholds and never reads patient images; all inputs are opaque
case identifiers and cached detector predictions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

AUDIT_POSITIVE_N = 50
AUDIT_CONTROL_N = 10
CACHE_RUN_IDS: Tuple[str, ...] = (
    "run-4",
    "run-5",
    "run-9",
    "run-13",
    "run-14",
    "run-15",
    "run-16",
    "run-17",
    "run-18",
)
FALLBACK_RUN_ID = "run-9"
FALLBACK_EXTRACTOR_ID = "B1"
MIN_PPV = 0.70
MAX_CONTROL_FALSE_POSITIVES = 1
MIN_CACHE_HIT_RATE = 0.80
MAX_FALLBACK_EXTRACTIONS = 5


@dataclass(frozen=True)
class CacheEntry:
    """One cached detector output for a single case from one run."""

    case_id: str
    run_id: str
    artifact_sha256: str
    prediction: bool


@dataclass(frozen=True)
class ShadowAuditReport:
    """Immutable outcome of one bounded read-only shadow audit."""

    positive_n: int
    control_n: int
    ppv: float
    control_false_positives: int
    cache_hits: int
    cache_misses: int
    fallback_extractions: int
    hash_mismatch_case_ids: Tuple[str, ...]
    fallback_case_ids: Tuple[str, ...]
    fallback_run_id: str = FALLBACK_RUN_ID
    fallback_extractor_id: str = FALLBACK_EXTRACTOR_ID

    @property
    def cache_hit_rate(self) -> float:
        total = self.positive_n + self.control_n
        return (self.cache_hits / total) if total else 0.0

    @property
    def passes_ppv(self) -> bool:
        return self.ppv >= MIN_PPV

    @property
    def passes_control_false_positive_budget(self) -> bool:
        return self.control_false_positives <= MAX_CONTROL_FALSE_POSITIVES

    @property
    def passes_cache_hit_rate(self) -> bool:
        return self.cache_hit_rate >= MIN_CACHE_HIT_RATE

    @property
    def passes_fallback_budget(self) -> bool:
        return self.fallback_extractions <= MAX_FALLBACK_EXTRACTIONS

    def passes(self) -> bool:
        return all(
            (
                self.passes_ppv,
                self.passes_control_false_positive_budget,
                self.passes_cache_hit_rate,
                self.passes_fallback_budget,
            )
        )

    def to_dict(self) -> dict:
        return {
            "positive_n": self.positive_n,
            "control_n": self.control_n,
            "ppv": self.ppv,
            "control_false_positives": self.control_false_positives,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "fallback_extractions": self.fallback_extractions,
            "hash_mismatch_case_ids": list(self.hash_mismatch_case_ids),
            "fallback_case_ids": list(self.fallback_case_ids),
            "fallback_run_id": self.fallback_run_id,
            "fallback_extractor_id": self.fallback_extractor_id,
            "passes_ppv": self.passes_ppv,
            "passes_control_false_positive_budget": self.passes_control_false_positive_budget,
            "passes_cache_hit_rate": self.passes_cache_hit_rate,
            "passes_fallback_budget": self.passes_fallback_budget,
            "passes": self.passes(),
        }


def _stable_key(seed: str, case_id: str) -> str:
    payload = f"{seed}\x1f{case_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bounded_select(case_ids: Iterable[str], n: int, *, seed: str) -> Tuple[str, ...]:
    """Deterministically pick ``n`` unique case ids from ``case_ids``.

    Ordering is by SHA256(seed + case_id), then by case_id, so repeated runs
    over the same inputs produce the same bounded sample without random state.
    """
    ordered = sorted(set(case_ids), key=lambda c: (_stable_key(seed, c), c))
    return tuple(ordered[:n])


def _accepted_cached_prediction(
    case_id: str,
    *,
    cache_lookup: Callable[[str], Sequence[CacheEntry]],
    expected_hashes: Optional[Mapping[str, str]],
    cache_run_ids: Tuple[str, ...],
    hash_mismatch: list,
) -> Optional[bool]:
    for entry in cache_lookup(case_id) or ():
        if entry.case_id != case_id:
            continue
        if entry.run_id not in cache_run_ids:
            continue
        if expected_hashes is not None:
            expected = expected_hashes.get(entry.run_id)
            if expected is None or expected != entry.artifact_sha256:
                hash_mismatch.append(case_id)
                continue
        return entry.prediction
    return None


def run_bounded_shadow_audit(
    *,
    positive_case_ids: Sequence[str],
    control_case_ids: Sequence[str],
    ground_truth: Mapping[str, bool],
    cache_lookup: Callable[[str], Sequence[CacheEntry]],
    fallback_extractor: Callable[[str], bool],
    expected_hashes: Optional[Mapping[str, str]] = None,
    cache_run_ids: Tuple[str, ...] = CACHE_RUN_IDS,
    fallback_run_id: str = FALLBACK_RUN_ID,
    fallback_extractor_id: str = FALLBACK_EXTRACTOR_ID,
) -> ShadowAuditReport:
    """Run one bounded, read-only shadow audit slice.

    Parameters are injected so the same logic can run offline in tests and in
    the orchestrator against real cache/extractor adapters.  The function never
    mutates labels, rules or thresholds; it only reads cached predictions,
    verifies artifact SHA256 digests, and falls back to the run-9 B1 extractor
    for cases without a valid cached entry.
    """
    if len(positive_case_ids) != AUDIT_POSITIVE_N:
        raise ValueError(
            f"expected {AUDIT_POSITIVE_N} positive cases, got {len(positive_case_ids)}"
        )
    if len(control_case_ids) != AUDIT_CONTROL_N:
        raise ValueError(
            f"expected {AUDIT_CONTROL_N} control cases, got {len(control_case_ids)}"
        )
    if len(set(positive_case_ids)) != AUDIT_POSITIVE_N:
        raise ValueError("positive_case_ids must be unique")
    if len(set(control_case_ids)) != AUDIT_CONTROL_N:
        raise ValueError("control_case_ids must be unique")
    overlap = set(positive_case_ids) & set(control_case_ids)
    if overlap:
        raise ValueError(
            f"positive and control case ids must be disjoint: {sorted(overlap)}"
        )

    hash_mismatch: list = []
    fallback_case_ids: list = []
    predictions: dict = {}

    def resolve(case_id: str) -> bool:
        cached = _accepted_cached_prediction(
            case_id,
            cache_lookup=cache_lookup,
            expected_hashes=expected_hashes,
            cache_run_ids=cache_run_ids,
            hash_mismatch=hash_mismatch,
        )
        if cached is not None:
            return bool(cached)
        fallback_case_ids.append(case_id)
        return bool(fallback_extractor(case_id))

    for case_id in positive_case_ids:
        predictions[case_id] = resolve(case_id)
    for case_id in control_case_ids:
        predictions[case_id] = resolve(case_id)

    # PPV among sampled positives: every sampled positive was already flagged
    # positive by the rule under audit, so PPV is the fraction that the
    # reference labels confirm as truly myo_topology positive.
    true_positives = sum(
        1 for cid in positive_case_ids if ground_truth.get(cid, False)
    )
    ppv = true_positives / AUDIT_POSITIVE_N

    # Controls are reference-negative; any positive detector prediction is a FP.
    control_false_positives = sum(
        1 for cid in control_case_ids if predictions[cid]
    )

    cache_hits = (AUDIT_POSITIVE_N + AUDIT_CONTROL_N) - len(fallback_case_ids)
    cache_misses = len(fallback_case_ids)

    # Preserve first-seen order while de-duplicating mismatch reports.
    seen_mismatch = set()
    dedup_mismatch = []
    for cid in hash_mismatch:
        if cid not in seen_mismatch:
            seen_mismatch.add(cid)
            dedup_mismatch.append(cid)

    return ShadowAuditReport(
        positive_n=AUDIT_POSITIVE_N,
        control_n=AUDIT_CONTROL_N,
        ppv=ppv,
        control_false_positives=control_false_positives,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        fallback_extractions=len(fallback_case_ids),
        hash_mismatch_case_ids=tuple(dedup_mismatch),
        fallback_case_ids=tuple(fallback_case_ids),
        fallback_run_id=fallback_run_id,
        fallback_extractor_id=fallback_extractor_id,
    )
