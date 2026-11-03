'''Draft, review-only rule R-CMR-MYO-TEMPORAL-HOLE-REVIEW-01.

Scope
=====
Draft definition of the CMR myocardial temporal-hole review rule for the
CMR/LGE cardiac-MRI QC orchestrator. The rule is default OFF and review-only;
it never writes to the production rule library and adds no numeric threshold.

Approved semantics (zero new numeric thresholds)
-----------------------------------------------
* Slice is POSITIVE  <=>  modality == cmr
                          AND cmr_myo_missing_temporal_hole is truthy
                          AND at least one support flag is truthy on that slice.
* Case ESCALATES for human review  <=>  at least one positive slice
                          AND at least two distinct support flags at case level.

Only four pre-existing support flags are recognised: cmr_myo_small_area,
myo_topology, cmr_myo_empty and cmr_myo_touches_image_border. Any other flag is
ignored by construction (the whitelist is iterated, never the payload keys).

MIN_SUPPORT_FLAGS_PER_SLICE / MIN_POSITIVE_SLICES_PER_CASE /
MIN_SUPPORT_FLAGS_PER_CASE are the approved structural requirements of the rule:
cardinality conditions on boolean flags, not numeric thresholds on
image-derived magnitudes. The rule never compares a measurement against a
cut-off (see NUMERIC_THRESHOLDS_ADDED).

Interpretation note
-------------------
Case level support flags are the distinct union of the support flags observed on
any slice of the case plus any support flag declared directly on the case
payload. The stricter variant (positive slices only) is available through
case_support_flags(..., only_positive_slices=True); both sets are reported in
the decision record so a reviewer can see the difference.

Ordering contract
-----------------
* slice_support_flags() is always reported in static whitelist order.
* case_support_flags() is reported in first-seen (observation) order across the
  slices, falling back to whitelist order within a single slice, with support
  flags declared directly on the case payload appended last. This makes the
  decision record reproducible and readable: it shows which slice/variant
  introduced each support flag.

Safety posture
--------------
* DEFAULT_ENABLED is False and the rule is review-only: the emitted action is
  either none or review, nothing else.
* Nothing in this module writes to the production rule library.
* guard_activation() and guard_production_write() always raise: promotion needs
  a separate approved change after the A1 audit.
* Flag parsing is fail-closed: unknown or unparsable values count as False.
'''

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

RULE_ID = 'R-CMR-MYO-TEMPORAL-HOLE-REVIEW-01'
RULE_NAME = 'cmr_myo_temporal_hole_review'
RULE_VERSION = 'draft-0.1.0'
RULE_STATUS = 'draft'
ACTION_CLASS = 'rule_draft'
OWNER_ROLE = 'rule_curator'
DECISION_SCOPE = 'review_only'

DEFAULT_ENABLED = False
REVIEW_ONLY = True
WRITES_PRODUCTION_RULE_LIBRARY = False
A1_AUDIT_REQUIRED = True
NUMERIC_THRESHOLDS_ADDED = 0

MODALITY_KEY = 'cmr'
TRIGGER_FLAG = 'cmr_myo_missing_temporal_hole'
SUPPORT_FLAGS: Tuple[str, ...] = (
    'cmr_myo_small_area',
    'myo_topology',
    'cmr_myo_empty',
    'cmr_myo_touches_image_border',
)

# Structural cardinality requirements approved with the rule draft.
MIN_SUPPORT_FLAGS_PER_SLICE = 1
MIN_POSITIVE_SLICES_PER_CASE = 1
MIN_SUPPORT_FLAGS_PER_CASE = 2

ALLOWED_ACTIONS: Tuple[str, ...] = ('none', 'review')

_SLICE_CONTAINER_KEYS: Tuple[str, ...] = (
    'slices',
    'slice_results',
    'slices_by_index',
    'series',
)
_FLAG_CONTAINER_KEYS: Tuple[str, ...] = ('flags', 'qc_flags', 'support_flags')
_MODALITY_KEYS: Tuple[str, ...] = (
    'modality',
    'modality_key',
    'modality_code',
    'series_modality',
)
_INDEX_KEYS: Tuple[str, ...] = (
    'slice_index',
    'slice',
    'index',
    'slice_number',
    'instance_number',
)
_CASE_ID_KEYS: Tuple[str, ...] = ('case_id', 'case')

_TRUTHY_STRINGS = frozenset((
    'true', 't', '1', 'yes', 'y', 'positive', 'present', 'flagged',
))


class RuleNotActivatableError(RuntimeError):
    '''Raised when something tries to activate or publish this draft rule.'''


class ReviewOnlyViolation(RuntimeError):
    '''Raised when a decision record would not be review-only.'''


def _truthy(value: Any) -> bool:
    '''Fail-closed truthiness used for QC flags.'''
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    return False


def _as_mapping(value: Any) -> Mapping:
    if isinstance(value, Mapping):
        return value
    return {}


def flag_is_set(payload: Any, flag_name: str) -> bool:
    '''True only when flag_name is explicitly set to a truthy value.'''
    mapping = _as_mapping(payload)
    if not mapping:
        return False
    if _truthy(mapping.get(flag_name)):
        return True
    for key in _FLAG_CONTAINER_KEYS:
        nested = _as_mapping(mapping.get(key))
        if nested and _truthy(nested.get(flag_name)):
            return True
    return False


def slice_support_flags(slice_payload: Any) -> Tuple[str, ...]:
    '''Whitelisted support flags that are truthy on this slice, and no others.

    Reported in static whitelist order, which keeps a single slice's flag list
    canonical regardless of payload key order.
    '''
    return tuple(flag for flag in SUPPORT_FLAGS if flag_is_set(slice_payload, flag))


def explicit_case_support_flags(case_payload: Any) -> Tuple[str, ...]:
    '''Support flags declared directly on the case payload (whitelist order).'''
    return tuple(flag for flag in SUPPORT_FLAGS if flag_is_set(case_payload, flag))


def _normalised_modality(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    return value.strip().lower()


def case_modality(case_payload: Any) -> str:
    mapping = _as_mapping(case_payload)
    for key in _MODALITY_KEYS:
        value = _normalised_modality(mapping.get(key))
        if value:
            return value
    return ''


def slice_modality(slice_payload: Any, case_payload: Any = None) -> str:
    mapping = _as_mapping(slice_payload)
    for key in _MODALITY_KEYS:
        value = _normalised_modality(mapping.get(key))
        if value:
            return value
    return case_modality(case_payload)


def slice_is_cmr(slice_payload: Any, case_payload: Any = None) -> bool:
    return slice_modality(slice_payload, case_payload) == MODALITY_KEY


def slice_has_temporal_hole_flag(slice_payload: Any) -> bool:
    return flag_is_set(slice_payload, TRIGGER_FLAG)


def iter_slices(case_payload: Any) -> Tuple[Tuple[Any, Mapping], ...]:
    '''Deterministic (index_hint, slice_payload) pairs for a case payload.'''
    if case_payload is None:
        return ()
    if isinstance(case_payload, Mapping):
        for key in _SLICE_CONTAINER_KEYS:
            container = case_payload.get(key)
            if container is None:
                continue
            if isinstance(container, Mapping):
                return tuple(
                    (key_name, value)
                    for key_name, value in sorted(
                        container.items(), key=lambda item: str(item[0])
                    )
                    if isinstance(value, Mapping)
                )
            if isinstance(container, (list, tuple)):
                return tuple(
                    (None, value)
                    for value in container
                    if isinstance(value, Mapping)
                )
        return ()
    if isinstance(case_payload, (list, tuple)):
        return tuple(
            (None, value) for value in case_payload if isinstance(value, Mapping)
        )
    return ()


def _coerce_index(value: Any) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip('-').isdigit():
            return int(text)
    return None


def slice_index(slice_payload: Any, index_hint: Any = None, position: int = 0) -> Any:
    mapping = _as_mapping(slice_payload)
    for key in _INDEX_KEYS:
        if key in mapping:
            coerced = _coerce_index(mapping.get(key))
            if coerced is not None:
                return coerced
    coerced_hint = _coerce_index(index_hint)
    if coerced_hint is not None:
        return coerced_hint
    return position


def slice_is_positive(slice_payload: Any, case_payload: Any = None) -> bool:
    '''cmr slice AND temporal-hole flag AND at least one support flag.'''
    if not slice_is_cmr(slice_payload, case_payload):
        return False
    if not slice_has_temporal_hole_flag(slice_payload):
        return False
    return len(slice_support_flags(slice_payload)) >= MIN_SUPPORT_FLAGS_PER_SLICE


def case_support_flags(
    case_payload: Any, only_positive_slices: bool = False
) -> Tuple[str, ...]:
    '''Distinct case-level support flags in first-seen (observation) order.

    Flags are collected while iterating the slices (deterministic slice order,
    whitelist order inside a slice); a flag is reported once, at the position
    where it was first observed. Support flags declared directly on the case
    payload are appended afterwards, in whitelist order. Only whitelisted flags
    can ever appear in the result.
    '''
    observed: List[str] = []
    for _index_hint, slice_payload in iter_slices(case_payload):
        if only_positive_slices and not slice_is_positive(slice_payload, case_payload):
            continue
        for flag in slice_support_flags(slice_payload):
            if flag not in observed:
                observed.append(flag)
    for flag in explicit_case_support_flags(case_payload):
        if flag not in observed:
            observed.append(flag)
    return tuple(observed)


def _case_id(case_payload: Any, case_id: Any = None) -> Any:
    if case_id is not None:
        return case_id
    mapping = _as_mapping(case_payload)
    for key in _CASE_ID_KEYS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _assert_review_only(record: Dict[str, Any]) -> Dict[str, Any]:
    '''Hard invariant: this draft may only describe a non-writing review hint.'''
    if record.get('action') not in ALLOWED_ACTIONS:
        raise ReviewOnlyViolation(
            'rule %s may only emit %r, got %r'
            % (RULE_ID, ALLOWED_ACTIONS, record.get('action'))
        )
    if record.get('enabled') is not False:
        raise ReviewOnlyViolation('rule %s must stay disabled in this draft' % RULE_ID)
    if record.get('review_only') is not True:
        raise ReviewOnlyViolation('rule %s must stay review-only' % RULE_ID)
    if record.get('writes_production_rule_library') is not False:
        raise ReviewOnlyViolation(
            'rule %s must not write to the production rule library' % RULE_ID
        )
    if record.get('numeric_thresholds_added') != NUMERIC_THRESHOLDS_ADDED:
        raise ReviewOnlyViolation(
            'rule %s must not add numeric thresholds' % RULE_ID
        )
    return record


def evaluate_case(case_payload: Any, case_id: Any = None) -> Dict[str, Any]:
    '''Review one case and return an advisory decision record (never a write).'''
    slices = iter_slices(case_payload)
    positive_slices: List[Dict[str, Any]] = []
    for position, (index_hint, slice_payload) in enumerate(slices):
        if slice_is_positive(slice_payload, case_payload):
            positive_slices.append({
                'slice_index': slice_index(slice_payload, index_hint, position),
                'support_flags': list(slice_support_flags(slice_payload)),
            })
    case_flags = case_support_flags(case_payload)
    positive_slice_flags = case_support_flags(
        case_payload, only_positive_slices=True
    )
    positive_count = len(positive_slices)
    escalate = (
        positive_count >= MIN_POSITIVE_SLICES_PER_CASE
        and len(case_flags) >= MIN_SUPPORT_FLAGS_PER_CASE
    )
    if escalate:
        reason = (
            'escalate: positive_slice_count >= %d and case_support_flag_count >= %d'
            % (MIN_POSITIVE_SLICES_PER_CASE, MIN_SUPPORT_FLAGS_PER_CASE)
        )
    elif positive_count == 0:
        reason = 'no_escalation: no positive slice'
    else:
        reason = 'no_escalation: case_support_flag_count %d < %d' % (
            len(case_flags), MIN_SUPPORT_FLAGS_PER_CASE,
        )
    record = {
        'rule_id': RULE_ID,
        'rule_version': RULE_VERSION,
        'rule_status': RULE_STATUS,
        'decision_scope': DECISION_SCOPE,
        'enabled': DEFAULT_ENABLED,
        'review_only': REVIEW_ONLY,
        'writes_production_rule_library': WRITES_PRODUCTION_RULE_LIBRARY,
        'numeric_thresholds_added': NUMERIC_THRESHOLDS_ADDED,
        'case_id': _case_id(case_payload, case_id),
        'action': 'review' if escalate else 'none',
        'escalate_for_review': escalate,
        'slice_count': len(slices),
        'positive_slice_count': positive_count,
        'positive_slices': positive_slices,
        'case_support_flags': list(case_flags),
        'positive_slice_support_flags': list(positive_slice_flags),
        'trigger_flag': TRIGGER_FLAG,
        'support_flag_whitelist': list(SUPPORT_FLAGS),
        'reason': reason,
    }
    return _assert_review_only(record)


def evaluate_case_batch(cases: Any) -> List[Dict[str, Any]]:
    '''Evaluate many cases; mapping keys are used as explicit case ids.'''
    if cases is None:
        return []
    if isinstance(cases, Mapping):
        return [
            evaluate_case(payload, case_id=case_key)
            for case_key, payload in sorted(
                cases.items(), key=lambda item: str(item[0])
            )
        ]
    if isinstance(cases, (list, tuple)):
        return [evaluate_case(payload) for payload in cases]
    return []


def activation_status() -> Dict[str, Any]:
    '''This draft is never enabled here; promotion is a separate change.'''
    return {
        'rule_id': RULE_ID,
        'rule_version': RULE_VERSION,
        'status': RULE_STATUS,
        'enabled': DEFAULT_ENABLED,
        'review_only': REVIEW_ONLY,
        'a1_audit_required': A1_AUDIT_REQUIRED,
        'a1_audit_verified_in_this_draft': False,
        'writes_production_rule_library': WRITES_PRODUCTION_RULE_LIBRARY,
        'activation_allowed_in_this_draft': False,
        'numeric_thresholds_added': NUMERIC_THRESHOLDS_ADDED,
    }


def guard_activation(*_args: Any, **_kwargs: Any) -> None:
    '''This draft can never be activated in place.'''
    raise RuleNotActivatableError(
        'rule %s is a default-off review-only draft; enabling it requires a '
        'separate approved promotion change after the A1 audit' % RULE_ID
    )


def guard_production_write(*_args: Any, **_kwargs: Any) -> None:
    '''This draft never writes to the production rule library.'''
    raise RuleNotActivatableError(
        'rule %s must not write to the production rule library before the A1 audit'
        % RULE_ID
    )


def rule_draft_manifest() -> Dict[str, Any]:
    '''Metadata-only description of the disabled draft registry entry.'''
    return {
        'rule_id': RULE_ID,
        'rule_name': RULE_NAME,
        'rule_version': RULE_VERSION,
        'status': RULE_STATUS,
        'action_class': ACTION_CLASS,
        'owner_role': OWNER_ROLE,
        'decision_scope': DECISION_SCOPE,
        'enabled': DEFAULT_ENABLED,
        'review_only': REVIEW_ONLY,
        'writes_production_rule_library': WRITES_PRODUCTION_RULE_LIBRARY,
        'a1_audit_required': A1_AUDIT_REQUIRED,
        'numeric_thresholds_added': NUMERIC_THRESHOLDS_ADDED,
        'trigger_flag': TRIGGER_FLAG,
        'support_flags': list(SUPPORT_FLAGS),
        'min_support_flags_per_slice': MIN_SUPPORT_FLAGS_PER_SLICE,
        'min_positive_slices_per_case': MIN_POSITIVE_SLICES_PER_CASE,
        'min_support_flags_per_case': MIN_SUPPORT_FLAGS_PER_CASE,
        'allowed_actions': list(ALLOWED_ACTIONS),
    }
