"""Draft rule ``R-CMR-CENTROID-DRIFT-REVIEW-01`` (default OFF, review-only).

Structure of the rule (as approved)
-----------------------------------
* **slice positive** := the slice belongs to a ``cmr`` case
  AND the primary drift flag is truthy on that slice
  AND the *same* slice carries at least one supporting flag.
* **case escalation** := at least one positive slice
  AND the case carries at least two supporting flags.

The rule is advisory only.  It emits a review record for human adjudication,
never writes to production, never mutates the QC payload and never relaxes an
existing hard gate.  ``DEFAULT_ENABLED`` is ``False`` so that wiring this module
into the orchestrator changes nothing until a curator flips the flag.

Zero new numeric thresholds are introduced: ``NUMERIC_THRESHOLDS`` is empty and
the only cardinalities used are the spec-mandated ones declared below.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "RULE_ID",
    "RULE_VERSION",
    "ISSUE_TYPE",
    "MODALITY",
    "MODE",
    "DEFAULT_ENABLED",
    "WRITES_PRODUCTION",
    "PRIMARY_FLAG",
    "SUPPORT_FLAG_KEYS",
    "MIN_SLICE_SUPPORT_FLAGS",
    "MIN_POSITIVE_SLICES_PER_CASE",
    "MIN_CASE_SUPPORT_FLAGS",
    "NUMERIC_THRESHOLDS",
    "WRITE_TARGETS",
    "rule_fingerprint",
    "cache_hash",
    "rule_metadata",
    "count_support_flags",
    "slice_support_flag_count",
    "slice_has_primary_flag",
    "resolve_slice_modality",
    "is_slice_positive",
    "positive_slice_indices",
    "is_case_escalated",
    "evaluate_case",
    "replay_metrics",
    "evaluate",
    "metadata",
]

# ---------------------------------------------------------------------------
# Rule identity / metadata
# ---------------------------------------------------------------------------

RULE_ID = "R-CMR-CENTROID-DRIFT-REVIEW-01"
RULE_VERSION = "0.1.0"
ISSUE_TYPE = "centroid_drift"
MODALITY = "cmr"
MODE = "review_only"
DEFAULT_ENABLED = False
WRITES_PRODUCTION = False

# ---------------------------------------------------------------------------
# Flags (no new flag is invented; supporting flags already exist upstream)
# ---------------------------------------------------------------------------

PRIMARY_FLAG = "centroid_drift_flag"

SUPPORT_FLAG_KEYS: tuple[str, ...] = (
    "centroid_drift_cooccurrence_flag",
    "centroid_drift_provenance_flag",
    "centroid_drift_bounded_flag",
)

# Spec-mandated cardinalities (structure of the rule, not tunable thresholds).
MIN_SLICE_SUPPORT_FLAGS = 1
MIN_POSITIVE_SLICES_PER_CASE = 1
MIN_CASE_SUPPORT_FLAGS = 2

# Deliberately empty: this draft introduces zero numeric thresholds.
NUMERIC_THRESHOLDS: Mapping[str, float] = {}
# Deliberately empty: this draft has no production write targets.
WRITE_TARGETS: tuple[str, ...] = ()

_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on"})


# ---------------------------------------------------------------------------
# Small, dependency-free helpers
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    """Interpret an existing QC flag value without inventing new semantics."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TOKENS
    return bool(value)


def _flag(record: Any, key: str) -> Any:
    """Look up ``key`` under ``record['flags']`` first, then at top level."""
    if not isinstance(record, Mapping):
        return None
    flags = record.get("flags")
    if isinstance(flags, Mapping) and key in flags:
        return flags[key]
    return record.get(key)


def count_support_flags(record: Any) -> int:
    """Number of *distinct* supporting flags that are truthy on ``record``."""
    return sum(1 for key in SUPPORT_FLAG_KEYS if _truthy(_flag(record, key)))


def _case_slices(case_record: Any) -> list:
    if not isinstance(case_record, Mapping):
        return []
    slices = case_record.get("slices")
    if slices is None:
        slices = case_record.get("slice_records")
    if slices is None:
        return []
    if isinstance(slices, Mapping):
        return list(slices.values())
    if isinstance(slices, Sequence) and not isinstance(slices, (str, bytes)):
        return list(slices)
    return []


def _case_modality(case_record: Any) -> str | None:
    if not isinstance(case_record, Mapping):
        return None
    modality = case_record.get("modality")
    if modality is None:
        modality = case_record.get("case_modality")
    return None if modality is None else str(modality).strip().lower()


# ---------------------------------------------------------------------------
# Rule body
# ---------------------------------------------------------------------------


def slice_support_flag_count(slice_record: Any) -> int:
    """Supporting flags present on a single slice."""
    return count_support_flags(slice_record)


def slice_has_primary_flag(slice_record: Any) -> bool:
    """True when the primary drift flag is truthy on this slice."""
    return _truthy(_flag(slice_record, PRIMARY_FLAG))


def resolve_slice_modality(slice_record: Any, case_modality: Any = None) -> str | None:
    """Slice modality, falling back to the case modality when absent."""
    modality = None
    if isinstance(slice_record, Mapping):
        modality = slice_record.get("modality")
    if modality is None:
        modality = case_modality
    return None if modality is None else str(modality).strip().lower()


def is_slice_positive(slice_record: Any, case_modality: Any = None) -> bool:
    """Slice-positive := cmr AND primary flag truthy AND >=1 supporting flag."""
    if resolve_slice_modality(slice_record, case_modality) != MODALITY:
        return False
    if not slice_has_primary_flag(slice_record):
        return False
    return slice_support_flag_count(slice_record) >= MIN_SLICE_SUPPORT_FLAGS


def positive_slice_indices(case_record: Any) -> list:
    """Indices (positional) of the positive slices inside a case record."""
    case_modality = _case_modality(case_record)
    return [
        index
        for index, slice_record in enumerate(_case_slices(case_record))
        if is_slice_positive(slice_record, case_modality)
    ]


def is_case_escalated(case_record: Any) -> bool:
    """Structural escalation gate, independent of the enable flag."""
    if _case_modality(case_record) != MODALITY:
        return False
    if len(positive_slice_indices(case_record)) < MIN_POSITIVE_SLICES_PER_CASE:
        return False
    return count_support_flags(case_record) >= MIN_CASE_SUPPORT_FLAGS


def _case_id(case_record: Any) -> Any:
    if not isinstance(case_record, Mapping):
        return None
    case_id = case_record.get("case_id")
    if case_id is None:
        case_id = case_record.get("id")
    return case_id


def evaluate_case(case_record: Any, *, enabled: bool = DEFAULT_ENABLED) -> dict:
    """Evaluate the review rule for one case.

    Always returns a record carrying the mandatory rule metadata
    (``issue_type``, ``modality``, ``cache_hash``, ``rule_version``).  The
    ``review_required`` field is only ever ``True`` when the caller opts in via
    ``enabled=True`` *and* the structural gates are satisfied.  No production
    write is performed or scheduled.
    """
    enabled = bool(enabled)
    case_modality = _case_modality(case_record)
    indices = positive_slice_indices(case_record)
    case_support = count_support_flags(case_record) if isinstance(case_record, Mapping) else 0

    modality_ok = case_modality == MODALITY
    structural = (
        modality_ok
        and len(indices) >= MIN_POSITIVE_SLICES_PER_CASE
        and case_support >= MIN_CASE_SUPPORT_FLAGS
    )

    if not modality_ok:
        reason = "case_modality_not_cmr"
    elif len(indices) < MIN_POSITIVE_SLICES_PER_CASE:
        reason = "no_positive_slice"
    elif case_support < MIN_CASE_SUPPORT_FLAGS:
        reason = "insufficient_case_support_flags"
    elif not enabled:
        reason = "rule_disabled"
    else:
        reason = "review_escalation"

    review_required = reason == "review_escalation"

    return {
        "case_id": _case_id(case_record),
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "issue_type": ISSUE_TYPE,
        "modality": MODALITY,
        "cache_hash": cache_hash(),
        "mode": MODE,
        "enabled": enabled,
        "writes_production": WRITES_PRODUCTION,
        "positive_slice_indices": indices,
        "positive_slice_count": len(indices),
        "case_support_flag_count": case_support,
        "structural_escalation": bool(structural),
        "review_required": bool(review_required),
        "reason": reason,
    }


def replay_metrics(
    labelled_cases: Sequence[Any], *, enabled: bool = DEFAULT_ENABLED
) -> dict:
    """Precision/recall of the rule over an inline labelled replay set.

    Each item is either ``{"case": <case record>, "expected_review": bool}``
    or a bare case record carrying ``expected_review``.  Pure function: no I/O.
    """
    tp = fp = fn = tn = 0
    for item in labelled_cases:
        if isinstance(item, Mapping) and "case" in item:
            case_record = item["case"]
        else:
            case_record = item
        expected = bool(item.get("expected_review")) if isinstance(item, Mapping) else False
        predicted = evaluate_case(case_record, enabled=enabled)["review_required"]
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


# ---------------------------------------------------------------------------
# Metadata / cache key (replay caches invalidate when the draft changes)
# ---------------------------------------------------------------------------


def rule_fingerprint() -> dict:
    """Canonical, JSON-serialisable description of this draft rule."""
    return {
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "issue_type": ISSUE_TYPE,
        "modality": MODALITY,
        "mode": MODE,
        "default_enabled": DEFAULT_ENABLED,
        "writes_production": WRITES_PRODUCTION,
        "primary_flag": PRIMARY_FLAG,
        "support_flag_keys": list(SUPPORT_FLAG_KEYS),
        "min_slice_support_flags": MIN_SLICE_SUPPORT_FLAGS,
        "min_positive_slices_per_case": MIN_POSITIVE_SLICES_PER_CASE,
        "min_case_support_flags": MIN_CASE_SUPPORT_FLAGS,
        "numeric_thresholds": dict(NUMERIC_THRESHOLDS),
        "write_targets": list(WRITE_TARGETS),
    }


def cache_hash(fingerprint: Mapping[str, Any] | None = None) -> str:
    """Deterministic hash used as the rule cache key (offline, stdlib only)."""
    payload = rule_fingerprint() if fingerprint is None else fingerprint
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def rule_metadata() -> dict:
    """Full rule metadata block for review handoff / coverage ledgers."""
    metadata_block = rule_fingerprint()
    metadata_block["cache_hash"] = cache_hash()
    return metadata_block


# Convenience aliases for orchestrator wiring.
evaluate = evaluate_case
metadata = rule_metadata
