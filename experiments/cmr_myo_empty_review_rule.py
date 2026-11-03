'''R-CMR-MYO-EMPTY-REVIEW-01 draft rule.

Review-only, default-off rule for CMR myocardial empty flag corroboration.
The module is intentionally isolated under experiments/ and performs no
production writes. It introduces no new numeric thresholds beyond the
approved count semantics in the maintenance objective.
'''

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RULE_ID = 'R-CMR-MYO-EMPTY-REVIEW-01'
RULE_VERSION = '0.1.0-draft'
RULE_ENABLED_BY_DEFAULT = False
PRIMARY_FLAG = 'myo_empty'
CMR_MODALITY = 'cmr'

# Approved count semantics from the maintenance objective. These are not
# newly invented thresholds; they encode the explicit >=1 and >=2 counts
# already present in the approved action.
MIN_SLICE_SUPPORTING_FLAGS = 1
MIN_CASE_SUPPORTING_FLAGS = 2

RULE_SPEC: dict[str, Any] = {
    'rule_id': RULE_ID,
    'version': RULE_VERSION,
    'default_enabled': RULE_ENABLED_BY_DEFAULT,
    'mode': 'review_only',
    'production_writes': False,
    'slice_positive': {
        'all_of': [
            'modality == cmr',
            'flags.myo_empty is True',
            'same_slice_supporting_flag_count >= 1',
            'paired_lge_myocardium_present is True',
        ]
    },
    'case_escalation': {
        'all_of': [
            'positive_slice_count >= 1',
            'case_supporting_flag_count >= 2',
        ]
    },
    'numeric_thresholds': {},
}


def canonical_rule_spec() -> str:
    '''Return deterministic serialization for cache-hash provenance.'''
    return json.dumps(RULE_SPEC, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def rule_cache_hash() -> str:
    '''Return the cache hash for the draft rule specification.'''
    return hashlib.sha256(canonical_rule_spec().encode('utf-8')).hexdigest()


RULE_CACHE_HASH = rule_cache_hash()


@dataclass(frozen=True)
class SliceDecision:
    positive: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CaseDecision:
    enabled: bool
    review_only: bool
    escalated: bool
    rule_id: str
    rule_version: str
    rule_cache_hash: str
    positive_slice_count: int
    case_supporting_flag_count: int
    slice_decisions: tuple[SliceDecision, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _truthy_flag(flags: Any, name: str) -> bool:
    return _as_mapping(flags).get(name) is True


def _supporting_flags_from_slice(slice_record: Mapping[str, Any]) -> tuple[str, ...]:
    raw = slice_record.get('supporting_flags')
    if raw is None:
        raw = slice_record.get('same_slice_supporting_flags')
    names: list[str] = []
    for item in _as_sequence(raw):
        if isinstance(item, str) and item:
            names.append(item)
    return tuple(sorted(set(names)))


def evaluate_slice(slice_record: Mapping[str, Any], primary_flag: str = PRIMARY_FLAG) -> SliceDecision:
    '''Evaluate one slice under the default-off review rule.

    A slice is positive only when all approved conditions hold:
    CMR modality, primary flag true, at least one same-slice supporting
    flag, and paired LGE myocardium present.
    '''
    record = _as_mapping(slice_record)
    reasons: list[str] = []

    if str(record.get('modality', '')).lower() != CMR_MODALITY:
        reasons.append('modality_not_cmr')
    if not _truthy_flag(record.get('flags'), primary_flag):
        reasons.append('primary_flag_false')
    if len(_supporting_flags_from_slice(record)) < MIN_SLICE_SUPPORTING_FLAGS:
        reasons.append('no_same_slice_supporting_flag')
    if record.get('paired_lge_myocardium_present') is not True:
        reasons.append('paired_lge_myocardium_absent')

    return SliceDecision(positive=not reasons, reasons=tuple(reasons))


def _case_supporting_flags(case_record: Mapping[str, Any], slices: Sequence[Any]) -> tuple[str, ...]:
    raw = case_record.get('case_supporting_flags')
    if raw is None:
        raw = case_record.get('supporting_flags')

    names: list[str] = []
    if raw is not None:
        for item in _as_sequence(raw):
            if isinstance(item, str) and item:
                names.append(item)
    else:
        for item in slices:
            names.extend(_supporting_flags_from_slice(_as_mapping(item)))
    return tuple(sorted(set(names)))


def evaluate_case(
    case_record: Mapping[str, Any],
    *,
    enabled: bool = RULE_ENABLED_BY_DEFAULT,
    primary_flag: str = PRIMARY_FLAG,
) -> CaseDecision:
    '''Evaluate case escalation for the review-only draft rule.

    Default ``enabled=False`` keeps the rule dark. When explicitly enabled
    in isolation, escalation requires at least one positive slice and at
    least two case-level supporting flags.
    '''
    record = _as_mapping(case_record)
    slices = tuple(_as_sequence(record.get('slices')))
    slice_decisions = tuple(evaluate_slice(item, primary_flag=primary_flag) for item in slices)
    positive_slice_count = sum(1 for decision in slice_decisions if decision.positive)
    case_supporting = _case_supporting_flags(record, slices)
    case_supporting_flag_count = len(case_supporting)

    if enabled is not True:
        return CaseDecision(
            enabled=False,
            review_only=True,
            escalated=False,
            rule_id=RULE_ID,
            rule_version=RULE_VERSION,
            rule_cache_hash=RULE_CACHE_HASH,
            positive_slice_count=positive_slice_count,
            case_supporting_flag_count=case_supporting_flag_count,
            slice_decisions=slice_decisions,
            reasons=('default_disabled',),
        )

    reasons: list[str] = []
    if positive_slice_count < 1:
        reasons.append('no_positive_slice')
    if case_supporting_flag_count < MIN_CASE_SUPPORTING_FLAGS:
        reasons.append('insufficient_case_supporting_flags')

    return CaseDecision(
        enabled=True,
        review_only=True,
        escalated=not reasons,
        rule_id=RULE_ID,
        rule_version=RULE_VERSION,
        rule_cache_hash=RULE_CACHE_HASH,
        positive_slice_count=positive_slice_count,
        case_supporting_flag_count=case_supporting_flag_count,
        slice_decisions=slice_decisions,
        reasons=tuple(reasons),
    )


def rule_trace() -> dict[str, Any]:
    '''Return trace metadata without touching production stores.'''
    return {
        'rule_id': RULE_ID,
        'rule_version': RULE_VERSION,
        'rule_cache_hash': RULE_CACHE_HASH,
        'default_enabled': RULE_ENABLED_BY_DEFAULT,
        'mode': 'review_only',
        'production_writes': False,
        'numeric_thresholds': dict(RULE_SPEC['numeric_thresholds']),
    }
