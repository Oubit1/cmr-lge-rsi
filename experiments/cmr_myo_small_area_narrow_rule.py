"""Narrow small-myocardial-area rule draft for CMR/LGE short-axis stacks.

Rule (narrow scope):
    Flag small_area when at least MIN_CONSECUTIVE_SLICES (default 2)
    consecutive slices show myocardial area below AREA_RATIO_THRESHOLD
    (default 0.15) times the blood-pool area of the same slice.

Guards (rule suppression -> stack-level review):
    Temporal alignment failure, a myocardial contour touching the image
    border, or a temporal hole makes a slice unusable. Missing gate fields
    and unmeasurable areas are equally unusable. A stack containing any
    unusable slice suppresses the rule and escalates to stack-level review,
    because the consecutive-slice count cannot be trusted near that slice.
    Unknown evidence gates conservatively instead of being silently dropped;
    no QC safety check is weakened.

Pure functions only (stdlib, no I/O, no dataset/patient-image access) so the
curator can run the rule offline on de-identified derived area tables.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RULE_ID = "cmr_myo_small_area_narrow"
AREA_RATIO_THRESHOLD = 0.15
MIN_CONSECUTIVE_SLICES = 2

DECISION_FLAG = "flag_small_area"
DECISION_ESCALATE = "escalate_stack_review"
DECISION_CLEAR = "clear"

REASON_TEMPORAL_ALIGNMENT = "temporal_alignment_failed"
REASON_TOUCHES_BORDER = "touches_border"
REASON_TEMPORAL_HOLE = "temporal_hole"
REASON_EVIDENCE_INCOMPLETE = "evidence_incomplete"
REASON_AREA_UNMEASURABLE = "area_unmeasurable"

STATUS_SMALL = "small"
STATUS_NOT_SMALL = "not_small"
STATUS_GATED = "gated"
STATUS_UNMEASURABLE = "unmeasurable"

# (field, reason, gated_when_value)
_BOOL_GATES: Tuple[Tuple[str, str, bool], ...] = (
    ("temporal_aligned", REASON_TEMPORAL_ALIGNMENT, False),
    ("touches_border", REASON_TOUCHES_BORDER, True),
    ("temporal_hole", REASON_TEMPORAL_HOLE, True),
)


def _append_unique(bucket: List[str], value: str) -> None:
    if value not in bucket:
        bucket.append(value)


def _area(value: Any) -> Optional[float]:
    """Coerce a numeric area; booleans, NaN and infinities are unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def slice_area_ratio(record: Mapping[str, Any]) -> Optional[float]:
    """Return myo_area / blood_pool_area, or None when not measurable."""
    myo = _area(record.get("myo_area"))
    pool = _area(record.get("blood_pool_area"))
    if myo is None or pool is None or pool <= 0.0:
        return None
    return myo / pool


def gate_reasons(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Suppression reasons for one slice, deduplicated and order-stable."""
    reasons: List[str] = []
    for field, reason, gated_when in _BOOL_GATES:
        if field not in record or record[field] is None:
            _append_unique(reasons, REASON_EVIDENCE_INCOMPLETE)
            continue
        flag = record[field]
        if not isinstance(flag, bool):
            _append_unique(reasons, REASON_EVIDENCE_INCOMPLETE)
            continue
        if flag is gated_when:
            _append_unique(reasons, reason)
    return tuple(reasons)


def slice_status(
    record: Mapping[str, Any], threshold: float = AREA_RATIO_THRESHOLD
) -> str:
    """Classify one slice: small / not_small / gated / unmeasurable."""
    if gate_reasons(record):
        return STATUS_GATED
    ratio = slice_area_ratio(record)
    if ratio is None:
        return STATUS_UNMEASURABLE
    return STATUS_SMALL if ratio < threshold else STATUS_NOT_SMALL


def longest_small_run(statuses: Sequence[str]) -> int:
    """Length of the longest run of consecutive STATUS_SMALL slices."""
    best = 0
    current = 0
    for status in statuses:
        if status == STATUS_SMALL:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def evaluate_stack(
    slices: Iterable[Mapping[str, Any]],
    threshold: float = AREA_RATIO_THRESHOLD,
    min_consecutive: int = MIN_CONSECUTIVE_SLICES,
) -> Dict[str, Any]:
    """Decide flag / escalate / clear for one ordered stack of slices.

    A stack is escalated whenever any slice is unusable, because the
    consecutive-slice evidence cannot be trusted there. Otherwise the stack
    is flagged only when the longest run of small slices reaches
    ``min_consecutive``; otherwise it is cleared.
    """
    records = list(slices)
    statuses: List[str] = []
    reasons: List[str] = []

    for record in records:
        slice_reasons = gate_reasons(record)
        if slice_reasons:
            statuses.append(STATUS_GATED)
            for reason in slice_reasons:
                _append_unique(reasons, reason)
            continue
        ratio = slice_area_ratio(record)
        if ratio is None:
            statuses.append(STATUS_UNMEASURABLE)
            _append_unique(reasons, REASON_AREA_UNMEASURABLE)
            continue
        statuses.append(STATUS_SMALL if ratio < threshold else STATUS_NOT_SMALL)

    run = longest_small_run(statuses)
    if reasons:
        decision = DECISION_ESCALATE
    elif run >= min_consecutive:
        decision = DECISION_FLAG
    else:
        decision = DECISION_CLEAR

    return {
        "rule_id": RULE_ID,
        "decision": decision,
        "suppressed": bool(reasons),
        "reasons": tuple(reasons),
        "longest_small_run": run,
        "slice_statuses": tuple(statuses),
        "threshold": float(threshold),
        "min_consecutive": int(min_consecutive),
    }


def tally_decisions(results: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Count rule decisions so the curator can report firing rates."""
    tally = {DECISION_FLAG: 0, DECISION_ESCALATE: 0, DECISION_CLEAR: 0}
    total = 0
    for result in results:
        decision = result.get("decision")
        if decision in tally:
            tally[decision] += 1
            total += 1
    tally["total"] = total
    return tally


def agreement_rate(
    pairs: Iterable[Tuple[str, str]], include_deferred: bool = False
) -> Optional[float]:
    """Rule-vs-expert agreement; deferred (escalated) cases are skipped.

    Returns None when no comparable pair exists, so callers never divide by
    zero on an all-deferred sample.
    """
    comparable = 0
    matches = 0
    for rule_decision, expert_decision in pairs:
        if rule_decision == DECISION_ESCALATE and not include_deferred:
            continue
        comparable += 1
        if rule_decision == expert_decision:
            matches += 1
    if comparable == 0:
        return None
    return matches / comparable
