'''Draft read-only QC review rule R-CMR-MYO-EXCESS-REVIEW-01.

Disabled by default. This module does not activate, promote, or write
production labels. It is composed only of existing boolean flags and
introduces zero new numeric thresholds.
'''
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

RULE_ID = 'R-CMR-MYO-EXCESS-REVIEW-01'
RULE_VERSION = 'draft-0'
DEFAULT_ENABLED = False

SLICE_REQUIRED_FLAGS: Tuple[str, ...] = ('cmr_myo_excess_components',)
SLICE_SUPPORT_FLAGS: Tuple[str, ...] = (
    'cmr_myo_small_area',
    'cmr_myo_touches_image_border',
    'cmr_myo_missing_temporal_hole',
    'cmr_centroid_drift',
    'myo_topology',
)
CASE_ESCALATION_FLAGS: Tuple[str, ...] = (
    'cmr_layer_area_jump',
    'cmr_myo_empty',
    'cmr_centroid_drift',
)

METADATA: Dict[str, Any] = {
    'rule_id': RULE_ID,
    'version': RULE_VERSION,
    'status': 'draft',
    'enabled_by_default': False,
    'activation': 'not_activated',
    'promotion': 'not_promoted',
    'mode': 'read_only_shadow_qc_review',
    'writes_production_labels': False,
    'purpose': 'QC review trigger only',
    'not_for': ('segmentation correction', 'clinical diagnosis'),
    'input_flags': {
        'slice_required': SLICE_REQUIRED_FLAGS,
        'slice_support': SLICE_SUPPORT_FLAGS,
        'case_escalation': CASE_ESCALATION_FLAGS,
    },
    'new_numeric_thresholds': 0,
    'percentile_on_run_values': False,
    'non_overlap': {
        'rule': 'R-CMR-MYO-SMALL-01',
        'statement': (
            'This draft does not duplicate or broaden R-CMR-MYO-SMALL-01. '
            'It requires cmr_myo_excess_components == true and at least one '
            'same-slice support flag; cmr_myo_small_area alone is not sufficient '
            'and no small-area-only case becomes positive. The two rules may '
            'co-occur on a slice, but neither subsumes the other.'
        ),
    },
}


def _flag_is_true(record: Mapping[str, Any], key: str) -> bool:
    return record.get(key) is True


def _modality_is_cmr(record: Mapping[str, Any]) -> bool:
    return record.get('modality') == 'cmr'


def slice_positive(slice_record: Mapping[str, Any]) -> bool:
    '''Return True iff the slice/frame satisfies the draft predicate.'''
    if not _modality_is_cmr(slice_record):
        return False
    if not _flag_is_true(slice_record, 'cmr_myo_excess_components'):
        return False
    return any(_flag_is_true(slice_record, flag) for flag in SLICE_SUPPORT_FLAGS)


def evaluate_slice(slice_record: Mapping[str, Any]) -> Dict[str, Any]:
    '''Evaluate one slice/frame for R-CMR-MYO-EXCESS-REVIEW-01.'''
    support = tuple(
        flag for flag in SLICE_SUPPORT_FLAGS if _flag_is_true(slice_record, flag)
    )
    positive = (
        _modality_is_cmr(slice_record)
        and _flag_is_true(slice_record, 'cmr_myo_excess_components')
        and bool(support)
    )
    reason = (
        'QC review trigger: cmr excess components with same-slice support flag'
        if positive
        else 'No QC review trigger'
    )
    return {
        'rule_id': RULE_ID,
        'positive': positive,
        'support_flags': support,
        'reason': reason,
    }


def _escalation_flag_present(
    case_record: Mapping[str, Any],
    slices: Sequence[Mapping[str, Any]],
    flag: str,
) -> bool:
    if _flag_is_true(case_record, flag):
        return True
    return any(_flag_is_true(slice_record, flag) for slice_record in slices)


def evaluate_case(case_record: Mapping[str, Any]) -> Dict[str, Any]:
    '''Evaluate one casebook entry, including escalation -E.'''
    slices = tuple(case_record.get('slices', ()))
    slice_results = tuple(evaluate_slice(slice_record) for slice_record in slices)
    positive_slice_indices = tuple(
        index for index, result in enumerate(slice_results) if result['positive']
    )
    escalation_flags = tuple(
        flag
        for flag in CASE_ESCALATION_FLAGS
        if _escalation_flag_present(case_record, slices, flag)
    )
    case_escalation = bool(positive_slice_indices) and bool(escalation_flags)
    return {
        'rule_id': RULE_ID,
        'case_id': case_record.get('case_id'),
        'slice_results': slice_results,
        'positive_slice_indices': positive_slice_indices,
        'escalation_flags': escalation_flags,
        'case_escalation': case_escalation,
        'reason': (
            'Casebook escalation -E QC review trigger'
            if case_escalation
            else 'No casebook escalation'
        ),
    }


def replay_cached_digest(digest: Mapping[str, Any]) -> Dict[str, Any]:
    '''Read-only shadow replay over a cached digest of boolean flags.'''
    cases = digest.get('cases')
    if cases is None:
        cases = (
            {
                'case_id': digest.get('case_id', 'single-case'),
                'slices': digest.get('slices', ()),
            },
        )
    cases = tuple(cases)

    per_slice_total = 0
    per_slice_positives = 0
    per_case_total = 0
    per_case_escalations = 0
    support_flag_cooccurrence = {flag: 0 for flag in SLICE_SUPPORT_FLAGS}
    control_total = 0
    control_escalations = 0
    case_reports = []

    for case_record in cases:
        case_report = evaluate_case(case_record)
        case_reports.append(case_report)
        per_case_total += 1
        slices = tuple(case_record.get('slices', ()))
        per_slice_total += len(slices)
        per_slice_positives += len(case_report['positive_slice_indices'])
        if case_report['case_escalation']:
            per_case_escalations += 1
        for slice_result in case_report['slice_results']:
            if slice_result['positive']:
                for flag in slice_result['support_flags']:
                    support_flag_cooccurrence[flag] += 1
        if case_record.get('is_control') is True:
            control_total += 1
            if case_report['case_escalation']:
                control_escalations += 1

    control_false_positive_rate = (
        control_escalations / control_total if control_total else 0.0
    )

    return {
        'rule_id': RULE_ID,
        'mode': 'shadow_replay_read_only',
        'per_slice_total': per_slice_total,
        'per_slice_positives': per_slice_positives,
        'per_case_total': per_case_total,
        'per_case_escalations': per_case_escalations,
        'support_flag_cooccurrence': support_flag_cooccurrence,
        'control_total': control_total,
        'control_escalations': control_escalations,
        'control_false_positive_rate': control_false_positive_rate,
        'case_reports': tuple(case_reports),
    }
