"""Bounded, read-only shadow audit for ``cmr_centroid_drift`` co-occurrence strata.

Scope
-----
This module is intentionally a *bookkeeping* layer.  It consumes already
extracted, already labelled per-case slice counts and never touches pixel
data, patient identifiers, production labels, review rules or thresholds.
Nothing is written to disk: the module returns an immutable report object.

Cache policy
------------
Only ``run-4/5/9/13-18`` cache entries may be reused.  A case counts as a
cache hit only when its recorded hash equals the expected hash.  Any other
case -- mismatched hash, or a run outside the reusable set -- is re-extracted
with the ``run-9:B1`` extractor and the fallback is recorded in the mismatch
ledger so the deviation stays auditable.

Acceptance mirrors the approved maintenance action:

* slice-level positive concordance lift >= ``MIN_POSITIVE_CONCORDANCE_LIFT``
  relative to the baseline,
* control slice-level false-positive rate <=
  ``MAX_CONTROL_FALSE_POSITIVE_RATE``,
* cache hit rate >= ``MIN_CACHE_HIT_RATE`` with a complete mismatch ledger,
* the approved cohort shape (``EXPECTED_POSITIVE_CASES`` /
  ``EXPECTED_CONTROL_CASES``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "COOCCURRENCE_FLAGS",
    "REUSABLE_RUNS",
    "FALLBACK_EXTRACTOR_RUN",
    "FALLBACK_EXTRACTOR_TAG",
    "EXPECTED_POSITIVE_CASES",
    "EXPECTED_CONTROL_CASES",
    "MIN_POSITIVE_CONCORDANCE_LIFT",
    "MAX_CONTROL_FALSE_POSITIVE_RATE",
    "MIN_CACHE_HIT_RATE",
    "STRATUM_NONE",
    "AuditInputError",
    "ShadowCase",
    "CacheDecision",
    "MismatchLedgerEntry",
    "StratumSummary",
    "ShadowAuditReport",
    "cooccurrence_stratum",
    "resolve_cache",
    "audit_shadow_cohort",
]

#: Co-occurrence features the shadow audit stratifies on (declared order is
#: the canonical order used to build stratum keys).
COOCCURRENCE_FLAGS: Tuple[str, ...] = (
    "small_area",
    "excess_components",
    "border",
    "review_centroid_outlier",
)

#: Cache runs approved for reuse by the maintenance action.
REUSABLE_RUNS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)

#: The only extractor allowed when a cache entry cannot be reused.
FALLBACK_EXTRACTOR_RUN: int = 9
FALLBACK_EXTRACTOR_TAG: str = "run-9:B1"

#: Approved cohort shape for this bounded shadow audit.
EXPECTED_POSITIVE_CASES: int = 30
EXPECTED_CONTROL_CASES: int = 10

#: Acceptance thresholds (never weakened by this module).
MIN_POSITIVE_CONCORDANCE_LIFT: float = 0.10
MAX_CONTROL_FALSE_POSITIVE_RATE: float = 0.05
MIN_CACHE_HIT_RATE: float = 0.80

STRATUM_NONE: str = "none"

_POSITIVE_COHORT = "positive"
_CONTROL_COHORT = "control"
_COHORTS: Tuple[str, ...] = (_POSITIVE_COHORT, _CONTROL_COHORT)


class AuditInputError(ValueError):
    """Raised when a shadow-audit record is malformed or ambiguous."""


@dataclass(frozen=True)
class ShadowCase:
    """One read-only observation for a single case already extracted upstream.

    ``baseline_hits`` / ``shadow_hits`` are *slice* counts: for a positive
    case they are the slices concordant with the cmr_centroid_drift call; for
    a control case they are the slices spuriously flagged.  The audit never
    recomputes them and never rewrites labels.
    """

    case_id: str
    cohort: str
    flags: Mapping[str, bool]
    cache_run: int
    cache_hash: str
    expected_hash: str
    slices: int
    baseline_hits: int
    shadow_hits: int

    def __post_init__(self) -> None:  # noqa: D105 - validation only
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise AuditInputError("case_id must be a non-empty string")
        if self.cohort not in _COHORTS:
            raise AuditInputError(
                "cohort must be one of {0!r}, got {1!r}".format(
                    list(_COHORTS), self.cohort
                )
            )
        if not isinstance(self.flags, Mapping):
            raise AuditInputError("flags must be a mapping of bool values")
        unknown = sorted(set(self.flags) - set(COOCCURRENCE_FLAGS))
        if unknown:
            raise AuditInputError(
                "unknown co-occurrence flag(s): {0}".format(", ".join(unknown))
            )
        if not isinstance(self.cache_run, int) or isinstance(self.cache_run, bool):
            raise AuditInputError("cache_run must be an int")
        if self.cache_run <= 0:
            raise AuditInputError("cache_run must be positive")
        for name in ("cache_hash", "expected_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AuditInputError("{0} must be a non-empty string".format(name))
        if not isinstance(self.slices, int) or self.slices <= 0:
            raise AuditInputError("slices must be a positive int")
        for name in ("baseline_hits", "shadow_hits"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise AuditInputError("{0} must be an int".format(name))
            if value < 0 or value > self.slices:
                raise AuditInputError(
                    "{0}={1} out of range 0..{2}".format(name, value, self.slices)
                )


@dataclass(frozen=True)
class CacheDecision:
    """Outcome of the read-only cache reuse decision for one case."""

    case_id: str
    used_run: int
    extractor: str
    cache_hit: bool
    hash_mismatch: bool
    registered: bool
    reason: str


@dataclass(frozen=True)
class MismatchLedgerEntry:
    """Registered fallback to the run-9:B1 extractor."""

    case_id: str
    declared_run: int
    declared_hash: str
    expected_hash: str
    fallback_run: int
    extractor: str
    reason: str
    hash_mismatch: bool


@dataclass(frozen=True)
class StratumSummary:
    """Slice-level metrics rolled up for one co-occurrence stratum."""

    stratum: str
    positive_cases: int
    control_cases: int
    positive_slices: int
    control_slices: int
    baseline_positive_concordance: float
    shadow_positive_concordance: float
    positive_lift: Optional[float]
    control_false_positive_rate: float
    cache_hit_rate: float


@dataclass(frozen=True)
class ShadowAuditReport:
    """Immutable result of the bounded read-only shadow audit."""

    strata: Tuple[StratumSummary, ...]
    decisions: Tuple[CacheDecision, ...]
    ledger: Tuple[MismatchLedgerEntry, ...]
    positive_cases: int
    control_cases: int
    baseline_positive_concordance: float
    shadow_positive_concordance: float
    positive_lift: Optional[float]
    control_false_positive_rate: float
    cache_hit_rate: float
    acceptance: Mapping[str, bool]
    readonly: bool = True
    writes_performed: bool = False
    labels_modified: bool = False
    rules_modified: bool = False
    thresholds_modified: bool = False

    @property
    def passed(self) -> bool:
        """True only when every acceptance check holds."""

        return all(self.acceptance.values())


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def cooccurrence_stratum(flags: Mapping[str, bool]) -> str:
    """Canonical, order-independent stratum key for a co-occurrence pattern."""

    if not isinstance(flags, Mapping):
        raise AuditInputError("flags must be a mapping of bool values")
    unknown = sorted(set(flags) - set(COOCCURRENCE_FLAGS))
    if unknown:
        raise AuditInputError(
            "unknown co-occurrence flag(s): {0}".format(", ".join(unknown))
        )
    active = [name for name in COOCCURRENCE_FLAGS if flags.get(name)]
    return "|".join(active) if active else STRATUM_NONE


def resolve_cache(case: ShadowCase) -> CacheDecision:
    """Decide whether an approved cache entry may be reused for ``case``.

    Reuse requires both an approved run *and* an exact hash match.  Every
    other case is routed to the run-9:B1 extractor and marked registered.
    """

    if case.cache_run not in REUSABLE_RUNS:
        return CacheDecision(
            case_id=case.case_id,
            used_run=FALLBACK_EXTRACTOR_RUN,
            extractor=FALLBACK_EXTRACTOR_TAG,
            cache_hit=False,
            hash_mismatch=False,
            registered=True,
            reason="run-{0}-not-in-reusable-set".format(case.cache_run),
        )
    if case.cache_hash != case.expected_hash:
        return CacheDecision(
            case_id=case.case_id,
            used_run=FALLBACK_EXTRACTOR_RUN,
            extractor=FALLBACK_EXTRACTOR_TAG,
            cache_hit=False,
            hash_mismatch=True,
            registered=True,
            reason="hash-mismatch",
        )
    return CacheDecision(
        case_id=case.case_id,
        used_run=case.cache_run,
        extractor="run-{0}:cache".format(case.cache_run),
        cache_hit=True,
        hash_mismatch=False,
        registered=False,
        reason="cache-hit",
    )


def _rate(hits: int, slices: int) -> float:
    if slices <= 0:
        return 0.0
    return hits / slices


def _lift(baseline: float, shadow: float) -> Optional[float]:
    if baseline <= 0.0:
        return None
    return (shadow - baseline) / baseline


def _totals(cases: Sequence[ShadowCase]) -> Tuple[int, int, int]:
    slices = sum(case.slices for case in cases)
    baseline = sum(case.baseline_hits for case in cases)
    shadow = sum(case.shadow_hits for case in cases)
    return slices, baseline, shadow


def _stratum_summary(
    stratum: str,
    positives: Sequence[ShadowCase],
    controls: Sequence[ShadowCase],
    decisions_by_id: Mapping[str, CacheDecision],
) -> StratumSummary:
    p_slices, p_base, p_shadow = _totals(positives)
    c_slices, _c_base, c_shadow = _totals(controls)
    baseline_rate = _rate(p_base, p_slices)
    shadow_rate = _rate(p_shadow, p_slices)
    members = tuple(positives) + tuple(controls)
    cache_hits = sum(1 for case in members if decisions_by_id[case.case_id].cache_hit)
    return StratumSummary(
        stratum=stratum,
        positive_cases=len(positives),
        control_cases=len(controls),
        positive_slices=p_slices,
        control_slices=c_slices,
        baseline_positive_concordance=baseline_rate,
        shadow_positive_concordance=shadow_rate,
        positive_lift=_lift(baseline_rate, shadow_rate),
        control_false_positive_rate=_rate(c_shadow, c_slices),
        cache_hit_rate=(cache_hits / len(members)) if members else 0.0,
    )


# ---------------------------------------------------------------------------
# audit entry point
# ---------------------------------------------------------------------------
def audit_shadow_cohort(cases: Sequence[ShadowCase]) -> ShadowAuditReport:
    """Run the bounded, read-only shadow audit over ``cases``.

    The function is pure: it never mutates the input, never writes to disk
    and never alters labels, review rules or thresholds.
    """

    case_tuple = tuple(cases)
    if not case_tuple:
        raise AuditInputError("at least one case is required")

    seen = set()
    for case in case_tuple:
        if case.case_id in seen:
            raise AuditInputError("duplicate case_id: {0}".format(case.case_id))
        seen.add(case.case_id)

    decisions = tuple(resolve_cache(case) for case in case_tuple)
    decisions_by_id = {decision.case_id: decision for decision in decisions}
    case_by_id = {case.case_id: case for case in case_tuple}

    ledger = tuple(
        MismatchLedgerEntry(
            case_id=decision.case_id,
            declared_run=case_by_id[decision.case_id].cache_run,
            declared_hash=case_by_id[decision.case_id].cache_hash,
            expected_hash=case_by_id[decision.case_id].expected_hash,
            fallback_run=FALLBACK_EXTRACTOR_RUN,
            extractor=FALLBACK_EXTRACTOR_TAG,
            reason=decision.reason,
            hash_mismatch=decision.hash_mismatch,
        )
        for decision in decisions
        if decision.registered
    )

    grouped: Dict[str, Dict[str, list]] = {}
    for case in case_tuple:
        key = cooccurrence_stratum(case.flags)
        bucket = grouped.setdefault(
            key, {_POSITIVE_COHORT: [], _CONTROL_COHORT: []}
        )
        bucket[case.cohort].append(case)

    strata = tuple(
        _stratum_summary(
            key,
            tuple(grouped[key][_POSITIVE_COHORT]),
            tuple(grouped[key][_CONTROL_COHORT]),
            decisions_by_id,
        )
        for key in sorted(grouped)
    )

    positives = tuple(c for c in case_tuple if c.cohort == _POSITIVE_COHORT)
    controls = tuple(c for c in case_tuple if c.cohort == _CONTROL_COHORT)

    p_slices, p_base, p_shadow = _totals(positives)
    c_slices, _c_base, c_shadow = _totals(controls)
    baseline_rate = _rate(p_base, p_slices)
    shadow_rate = _rate(p_shadow, p_slices)
    lift = _lift(baseline_rate, shadow_rate)
    control_fp_rate = _rate(c_shadow, c_slices)
    cache_hits = sum(1 for decision in decisions if decision.cache_hit)
    cache_hit_rate = cache_hits / len(decisions)

    hash_mismatch_ids = {
        decision.case_id for decision in decisions if decision.hash_mismatch
    }
    ledger_mismatch_ids = {
        entry.case_id for entry in ledger if entry.hash_mismatch
    }
    ledger_uses_fallback_only = all(
        entry.extractor == FALLBACK_EXTRACTOR_TAG
        and entry.fallback_run == FALLBACK_EXTRACTOR_RUN
        for entry in ledger
    )

    acceptance = {
        "cohort_composition": (
            len(positives) == EXPECTED_POSITIVE_CASES
            and len(controls) == EXPECTED_CONTROL_CASES
        ),
        "positive_concordance_lift": (
            lift is not None and lift >= MIN_POSITIVE_CONCORDANCE_LIFT
        ),
        "control_false_positive_rate": (
            control_fp_rate <= MAX_CONTROL_FALSE_POSITIVE_RATE
        ),
        "cache_hit_rate": cache_hit_rate >= MIN_CACHE_HIT_RATE,
        "mismatch_ledger": (
            hash_mismatch_ids.issubset(ledger_mismatch_ids)
            and ledger_uses_fallback_only
        ),
    }

    return ShadowAuditReport(
        strata=strata,
        decisions=decisions,
        ledger=ledger,
        positive_cases=len(positives),
        control_cases=len(controls),
        baseline_positive_concordance=baseline_rate,
        shadow_positive_concordance=shadow_rate,
        positive_lift=lift,
        control_false_positive_rate=control_fp_rate,
        cache_hit_rate=cache_hit_rate,
        acceptance=acceptance,
    )
