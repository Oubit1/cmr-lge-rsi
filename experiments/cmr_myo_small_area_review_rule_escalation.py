"""Draft, review-only rule ``R-CMR-MYO-SMALL-AREA-REVIEW-01`` (default OFF).

This module is the escalation companion for the small-area myocardial
review rule.  It consumes upstream-produced booleans verbatim and never
derives a new numeric threshold:

* slice positive = modality is ``cmr`` AND ``myo_small_area`` is true AND the
  same slice carries at least one supporting flag;
* case escalated = at least one positive slice AND at least two distinct
  case-level supporting flags AND paired LGE myocardial support present;
* supporting flags are limited to border contact, excess components and
  centroid drift - any other key is ignored;
* the rule is disabled by default and emits review items only: there are zero
  production write channels.

The cardinalities used here (1 positive slice, 2 supporting flags) come from
the approved rule text.  They count structural evidence, they are not
measure thresholds and no image-derived number is computed anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

RULE_ID = "R-CMR-MYO-SMALL-AREA-REVIEW-01"
DEFAULT_ENABLED = False
MODE = "review_only"

CMR_MODALITY = "cmr"
SMALL_AREA_FLAG = "myo_small_area"
SUPPORTING_FLAGS: Tuple[str, ...] = (
    "myo_touches_border",     # border contact
    "myo_excess_components",  # excess components
    "myo_centroid_drift",     # centroid drift
)
SUPPORTING_FLAG_SET = frozenset(SUPPORTING_FLAGS)
PAIRED_LGE_SUPPORT_KEY = "paired_lge_myo_support"

SLICE_POSITIVE_MIN = 1   # structural count, not a measure threshold
CASE_SUPPORT_MIN = 2     # structural count, not a measure threshold

PRODUCTION_WRITE_CHANNELS: Tuple[str, ...] = ()


def _is_true(record: Mapping[str, Any], key: str) -> bool:
    return bool(record.get(key))


def supporting_flags_on_slice(slice_record: Mapping[str, Any]) -> frozenset:
    """Allowed supporting flags present on one slice (others ignored)."""
    return frozenset(flag for flag in SUPPORTING_FLAGS if _is_true(slice_record, flag))


def is_small_area_cmr_slice(slice_record: Mapping[str, Any]) -> bool:
    """True when the slice is cmr and carries the small-area flag."""
    return (
        slice_record.get("modality") == CMR_MODALITY
        and _is_true(slice_record, SMALL_AREA_FLAG)
    )


def slice_positive(slice_record: Mapping[str, Any]) -> bool:
    """cmr + small-area flag + >=1 supporting flag on the same slice."""
    if not is_small_area_cmr_slice(slice_record):
        return False
    return len(supporting_flags_on_slice(slice_record)) >= SLICE_POSITIVE_MIN


@dataclass(frozen=True)
class CaseReview:
    """Immutable review decision; nothing here is written to production."""

    rule_id: str
    case_id: str
    enabled: bool
    escalated: bool
    reason: str
    positive_slices: Tuple[Any, ...]
    case_support: Tuple[str, ...]
    paired_lge_myo_support: bool

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "case_id": self.case_id,
            "enabled": self.enabled,
            "escalated": self.escalated,
            "reason": self.reason,
            "positive_slices": list(self.positive_slices),
            "case_support": list(self.case_support),
            "paired_lge_myo_support": self.paired_lge_myo_support,
        }


def _slice_id(slice_record: Mapping[str, Any], index: int) -> Any:
    return slice_record.get("slice_id", index)


def evaluate_case(case: Mapping[str, Any], enabled: bool = DEFAULT_ENABLED) -> CaseReview:
    """Evaluate one case for review-only escalation candidates."""
    case_id = str(case.get("case_id", ""))
    paired = bool(case.get(PAIRED_LGE_SUPPORT_KEY, False))

    if not enabled:
        return CaseReview(
            RULE_ID, case_id, False, False, "rule_disabled", (), (), paired
        )

    slices = tuple(case.get("slices") or ())
    positive = tuple(
        _slice_id(s, i) for i, s in enumerate(slices) if slice_positive(s)
    )

    support: set = set()
    for s in slices:
        if is_small_area_cmr_slice(s):
            support |= supporting_flags_on_slice(s)
    case_support = tuple(sorted(support))

    if len(positive) < SLICE_POSITIVE_MIN:
        escalated, reason = False, "no_positive_slice"
    elif not paired:
        escalated, reason = False, "no_paired_lge_myo_support"
    elif len(case_support) < CASE_SUPPORT_MIN:
        escalated, reason = False, "insufficient_case_level_support"
    else:
        escalated, reason = True, "escalation_candidate_review_only"

    return CaseReview(
        RULE_ID, case_id, True, escalated, reason, positive, case_support, paired
    )


def filter_escalation_candidates(
    cases: Iterable[Mapping[str, Any]], enabled: bool = DEFAULT_ENABLED
) -> Tuple[CaseReview, ...]:
    """Return only the escalated reviews; emits review items, never writes."""
    return tuple(
        review
        for review in (evaluate_case(case, enabled=enabled) for case in cases)
        if review.escalated
    )


def rule_metadata() -> dict:
    return {
        "rule_id": RULE_ID,
        "mode": MODE,
        "default_enabled": DEFAULT_ENABLED,
        "advisory_only": True,
        "mutates_production": False,
        "production_write_channels": list(PRODUCTION_WRITE_CHANNELS),
        "supporting_flags": list(SUPPORTING_FLAGS),
        "slice_positive_min_support": SLICE_POSITIVE_MIN,
        "case_support_min": CASE_SUPPORT_MIN,
        "requires_paired_lge_myo_support": True,
        "new_numeric_measure_thresholds": 0,
    }


def assert_review_only() -> None:
    """Fail loudly if a production write channel is ever wired in."""
    if PRODUCTION_WRITE_CHANNELS:
        raise AssertionError(
            "%s must stay review-only, found write channels: %r"
            % (RULE_ID, PRODUCTION_WRITE_CHANNELS)
        )
