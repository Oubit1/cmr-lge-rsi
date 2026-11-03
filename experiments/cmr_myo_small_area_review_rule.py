'''Review-only draft rule R-CMR-MYO-SMALL-AREA-REVIEW-01.

Default disabled. The rule is review-only: it never writes to production state
and never modifies labels. It introduces no numeric QC threshold.

Rule:
- Slice positive: cmr is true AND cmr_myo_small_area is true AND at least one
  support flag is true on the same slice.
- Case escalation: at least one positive slice AND at least two case-level
  support flags are true.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

RULE_ID = 'R-CMR-MYO-SMALL-AREA-REVIEW-01'

DEFAULT_SUPPORT_FLAG_KEYS = (
    'myo_small_area_support',
    'myo_small_area_shadow_support',
    'myo_small_area_import_support',
    'myo_small_area_bounded_support',
    'myo_border_support',
    'myo_topology_support',
    'centroid_provenance_support',
    'stack_continuity_support',
)

# Explicitly empty: this draft adds no numeric QC threshold.
NUMERIC_QC_THRESHOLDS: tuple[float, ...] = ()


@dataclass(frozen=True)
class RuleConfig:
    enabled: bool = False
    review_only: bool = True
    support_flag_keys: tuple[str, ...] = DEFAULT_SUPPORT_FLAG_KEYS

    def __post_init__(self) -> None:
        if not self.review_only:
            raise ValueError('R-CMR-MYO-SMALL-AREA-REVIEW-01 is review-only')


@dataclass(frozen=True)
class SliceDecision:
    slice_id: str
    positive: bool
    reasons: tuple[str, ...]
    support_flags: tuple[str, ...]
    action: str = 'review'
    write_allowed: bool = False
    label_modification_allowed: bool = False


@dataclass(frozen=True)
class CaseDecision:
    case_id: str
    escalate: bool
    positive_slice_ids: tuple[str, ...]
    case_support_flags: tuple[str, ...]
    reasons: tuple[str, ...]
    action: str = 'review'
    write_allowed: bool = False
    label_modification_allowed: bool = False


@dataclass(frozen=True)
class PrecisionRecord:
    true_positive: int
    false_positive: int
    precision: float | None


@dataclass(frozen=True)
class ReplayReport:
    slice: PrecisionRecord
    case: PrecisionRecord


@dataclass(frozen=True)
class SliceReplaySample:
    slice_record: Any
    expected_positive: bool


@dataclass(frozen=True)
class CaseReplaySample:
    case_record: Any
    slice_records: Sequence[Any]
    expected_escalation: bool


def default_config() -> RuleConfig:
    return RuleConfig()


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _is_true(value: Any) -> bool:
    return value is True


def _record_id(record: Any, key: str, fallback: str) -> str:
    value = _get(record, key, fallback)
    return str(value)


def _collect_support_flags(
    record: Any, support_flag_keys: Sequence[str]
) -> tuple[str, ...]:
    found: list[str] = []
    support_flags = _get(record, 'support_flags', None)
    if isinstance(support_flags, Mapping):
        for key, value in support_flags.items():
            if _is_true(value):
                found.append(str(key))
    for key in support_flag_keys:
        if _is_true(_get(record, key, False)):
            found.append(key)

    seen: set[str] = set()
    unique: list[str] = []
    for key in found:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return tuple(unique)


def decide_slice(
    slice_record: Any, config: RuleConfig | None = None
) -> SliceDecision:
    config = config or default_config()
    slice_id = _record_id(slice_record, 'slice_id', 'unknown-slice')
    if not config.enabled:
        return SliceDecision(
            slice_id=slice_id,
            positive=False,
            reasons=('rule_disabled',),
            support_flags=(),
        )

    support_flags = _collect_support_flags(slice_record, config.support_flag_keys)
    reasons: list[str] = []

    cmr = _is_true(_get(slice_record, 'cmr', False))
    small_area = _is_true(_get(slice_record, 'cmr_myo_small_area', False))

    if not cmr:
        reasons.append('not_cmr')
    if not small_area:
        reasons.append('cmr_myo_small_area_not_true')
    if not support_flags:
        reasons.append('no_slice_support_flag')

    positive = cmr and small_area and bool(support_flags)
    if positive:
        reasons.append('slice_positive')

    return SliceDecision(
        slice_id=slice_id,
        positive=positive,
        reasons=tuple(reasons),
        support_flags=support_flags,
    )


def decide_case(
    case_record: Any,
    slice_records: Sequence[Any],
    config: RuleConfig | None = None,
) -> CaseDecision:
    config = config or default_config()
    case_id = _record_id(case_record, 'case_id', 'unknown-case')
    if not config.enabled:
        return CaseDecision(
            case_id=case_id,
            escalate=False,
            positive_slice_ids=(),
            case_support_flags=(),
            reasons=('rule_disabled',),
        )

    slice_decisions = tuple(decide_slice(s, config) for s in slice_records)
    positive_slice_ids = tuple(d.slice_id for d in slice_decisions if d.positive)
    case_support_flags = _collect_support_flags(
        case_record, config.support_flag_keys
    )

    reasons: list[str] = []
    if not positive_slice_ids:
        reasons.append('no_positive_slice')
    if len(case_support_flags) < 2:
        reasons.append('case_support_flags_lt_2')

    escalate = bool(positive_slice_ids) and len(case_support_flags) >= 2
    if escalate:
        reasons.append('case_escalate_review')

    return CaseDecision(
        case_id=case_id,
        escalate=escalate,
        positive_slice_ids=positive_slice_ids,
        case_support_flags=case_support_flags,
        reasons=tuple(reasons),
    )


def _precision_record(true_positive: int, false_positive: int) -> PrecisionRecord:
    denominator = true_positive + false_positive
    precision = None if denominator == 0 else true_positive / denominator
    return PrecisionRecord(
        true_positive=true_positive,
        false_positive=false_positive,
        precision=precision,
    )


def record_replay_metrics(
    slice_samples: Iterable[SliceReplaySample],
    case_samples: Iterable[CaseReplaySample],
    config: RuleConfig | None = None,
) -> ReplayReport:
    '''Record offline slice-level and case-level precision for review.

    Replay is an offline analysis activity. The production default remains
    RuleConfig(enabled=False); callers must explicitly enable review.
    '''

    config = config or RuleConfig(enabled=True, review_only=True)

    slice_true_positive = 0
    slice_false_positive = 0
    for sample in slice_samples:
        decision = decide_slice(sample.slice_record, config)
        if decision.positive and sample.expected_positive:
            slice_true_positive += 1
        elif decision.positive and not sample.expected_positive:
            slice_false_positive += 1

    case_true_positive = 0
    case_false_positive = 0
    for sample in case_samples:
        decision = decide_case(sample.case_record, sample.slice_records, config)
        if decision.escalate and sample.expected_escalation:
            case_true_positive += 1
        elif decision.escalate and not sample.expected_escalation:
            case_false_positive += 1

    return ReplayReport(
        slice=_precision_record(slice_true_positive, slice_false_positive),
        case=_precision_record(case_true_positive, case_false_positive),
    )
