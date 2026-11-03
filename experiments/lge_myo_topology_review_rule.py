'''Review-only draft rule R-LGE-MYO-TOPOLOGY-REVIEW-01.

The rule is intentionally default-disabled and read-only. It computes only
review markers; it does not alter production thresholds, labels, writes, or
datasets.
'''
from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from typing import Any, Dict, Optional, Tuple

RULE_ID = 'R-LGE-MYO-TOPOLOGY-REVIEW-01'
DEFAULT_ENABLED = False
REVIEW_ONLY = True
PRODUCTION_THRESHOLDS_CHANGED = 0
LABELS_CHANGED = 0

# Structural cardinalities specified by the approved rule wording. These are
# not image-derived numeric thresholds and must not be tuned against data.
REQUIRED_SLICE_SUPPORT_FLAG_COUNT = len(('same_slice_support',))
REQUIRED_POSITIVE_SLICE_COUNT = len(('positive_slice',))
REQUIRED_CASE_SUPPORT_FLAG_COUNT = len(('case_support_one', 'case_support_two'))

SUPPORT_FLAG_PREFIX = 'support_'
SUPPORT_FLAG_KEYS = frozenset(
    {
        'myo_topology_support',
        'lge_myo_topology_support',
        'topology_support',
        'review_support',
        'provenance_support',
        'cooccurrence_support',
        'fragmentation_support',
        'multicomponent_support',
        'border_contact_support',
        'temporal_hole_support',
        'small_area_support',
        'excess_components_support',
        'support_flag',
    }
)


def _looks_like_support_flag(key: str) -> bool:
    lowered = key.lower()
    if lowered in SUPPORT_FLAG_KEYS:
        return True
    if lowered.startswith(SUPPORT_FLAG_PREFIX):
        return True
    if lowered.endswith('_support') or lowered.endswith('_support_flag'):
        return True
    if 'support_flag' in lowered:
        return True
    return False


def _explicit_support_flags(record: MappingABC[str, Any]) -> set:
    raw = record.get('support_flags')
    flags = set()
    if raw is None:
        return flags
    if isinstance(raw, str):
        raw = (raw,)
    if isinstance(raw, IterableABC):
        for item in raw:
            if isinstance(item, str) and item.strip():
                flags.add(item.strip())
    return flags


def _iter_support_flags(record: MappingABC[str, Any]) -> Tuple[str, ...]:
    if not isinstance(record, MappingABC):
        return ()
    flags = set(_explicit_support_flags(record))
    for key, value in record.items():
        if not isinstance(key, str) or key == 'support_flags':
            continue
        if not _looks_like_support_flag(key):
            continue
        if value is True:
            flags.add(key)
        elif isinstance(value, str) and value.strip():
            flags.add(key)
    return tuple(sorted(flags))


def evaluate_slice(slice_record: MappingABC[str, Any]) -> Dict[str, Any]:
    '''Evaluate one slice for the review-only topology rule.'''
    if not isinstance(slice_record, MappingABC):
        slice_record = {}
    lge = slice_record.get('lge') is True
    myo_topology = slice_record.get('myo_topology') is True
    support_flags = _iter_support_flags(slice_record)
    slice_positive = bool(
        lge
        and myo_topology
        and len(support_flags) >= REQUIRED_SLICE_SUPPORT_FLAG_COUNT
    )
    return {
        'slice_id': str(slice_record.get('slice_id') or slice_record.get('id') or ''),
        'lge': lge,
        'myo_topology': myo_topology,
        'support_flags': support_flags,
        'support_flag_count': len(support_flags),
        'slice_positive': slice_positive,
    }


def evaluate_case(
    case_record: MappingABC[str, Any], *, enabled: Optional[bool] = None
) -> Dict[str, Any]:
    '''Evaluate a case without mutating it or changing production state.'''
    if not isinstance(case_record, MappingABC):
        case_record = {}
    if enabled is None:
        enabled = DEFAULT_ENABLED
    enabled = bool(enabled)
    result: Dict[str, Any] = {
        'rule_id': RULE_ID,
        'enabled': enabled,
        'review_only': REVIEW_ONLY,
        'review_required': False,
        'reason': 'rule_disabled' if not enabled else '',
    }
    if not enabled:
        return result

    raw_slices = case_record.get('slices') or ()
    if isinstance(raw_slices, MappingABC) or isinstance(raw_slices, (str, bytes)):
        raw_slices = ()
    slice_results = tuple(
        evaluate_slice(slice_record)
        for slice_record in raw_slices
        if isinstance(slice_record, MappingABC)
    )
    positive_slices = tuple(
        slice_result for slice_result in slice_results if slice_result['slice_positive']
    )
    case_support_flags = _iter_support_flags(case_record)
    has_positive_slice = len(positive_slices) >= REQUIRED_POSITIVE_SLICE_COUNT
    has_case_support = len(case_support_flags) >= REQUIRED_CASE_SUPPORT_FLAG_COUNT
    review_required = bool(has_positive_slice and has_case_support)

    result.update(
        {
            'review_required': review_required,
            'reason': 'eligible' if review_required else 'insufficient_topology_evidence',
            'positive_slice_count': len(positive_slices),
            'positive_slice_ids': tuple(
                slice_result['slice_id'] for slice_result in positive_slices
            ),
            'case_support_flags': case_support_flags,
            'case_support_flag_count': len(case_support_flags),
            'slice_results': slice_results,
        }
    )
    return result


def static_validate() -> Dict[str, Any]:
    '''Return a static validation report for the draft rule.'''
    violations = []
    if RULE_ID != 'R-LGE-MYO-TOPOLOGY-REVIEW-01':
        violations.append('rule_id_mismatch')
    if DEFAULT_ENABLED is not False:
        violations.append('default_enabled_must_be_false')
    if REVIEW_ONLY is not True:
        violations.append('review_only_must_be_true')
    if PRODUCTION_THRESHOLDS_CHANGED != 0:
        violations.append('production_thresholds_changed_must_be_zero')
    if LABELS_CHANGED != 0:
        violations.append('labels_changed_must_be_zero')
    if REQUIRED_SLICE_SUPPORT_FLAG_COUNT != 1:
        violations.append('slice_support_cardinality_mismatch')
    if REQUIRED_POSITIVE_SLICE_COUNT != 1:
        violations.append('positive_slice_cardinality_mismatch')
    if REQUIRED_CASE_SUPPORT_FLAG_COUNT != 2:
        violations.append('case_support_cardinality_mismatch')
    return {
        'rule_id': RULE_ID,
        'valid': not violations,
        'violations': tuple(violations),
        'default_enabled': DEFAULT_ENABLED,
        'review_only': REVIEW_ONLY,
        'production_thresholds_changed': PRODUCTION_THRESHOLDS_CHANGED,
        'labels_changed': LABELS_CHANGED,
        'required_slice_support_flag_count': REQUIRED_SLICE_SUPPORT_FLAG_COUNT,
        'required_positive_slice_count': REQUIRED_POSITIVE_SLICE_COUNT,
        'required_case_support_flag_count': REQUIRED_CASE_SUPPORT_FLAG_COUNT,
    }


def audit_positive_consistency(
    cases: IterableABC, expected_labels: IterableABC, *, enabled: bool = True
) -> Dict[str, Any]:
    '''Compare review markers against expected audit-positive labels.'''
    cases = tuple(cases)
    expected_labels = tuple(expected_labels)
    if len(cases) != len(expected_labels):
        raise ValueError('cases and expected_labels must have the same length')
    matches = 0
    mismatches = []
    for index, (case, expected) in enumerate(zip(cases, expected_labels)):
        actual = bool(evaluate_case(case, enabled=enabled)['review_required'])
        if actual == bool(expected):
            matches += 1
        else:
            mismatches.append(
                {'index': index, 'expected': bool(expected), 'actual': actual}
            )
    total = len(cases)
    consistency = 1.0 if total == 0 else matches / total
    return {
        'rule_id': RULE_ID,
        'cases': total,
        'matches': matches,
        'consistency': consistency,
        'mismatches': tuple(mismatches),
    }
