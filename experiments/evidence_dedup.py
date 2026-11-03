'''Read-only evidence payload deduplication and iteration-history duplication guard.

This module is intentionally side-effect free: it never writes to production,
never edits labels, and never activates rules. It only transforms in-memory
payloads and returns new objects.

Public entry points:
- deduplicate_representative_cases: one row per (case_id, modality)
- rollup_cmr_flags: per-case CMR/LGE rollup with addressable slice indices
- round_trip_check / preservation_report: offline acceptance checks
- guard_iteration_history: flags re-proposals of run-3/4/5/6 classes
'''
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    'deduplicate_representative_cases',
    'rollup_cmr_flags',
    'round_trip_check',
    'preservation_report',
    'guard_iteration_history',
    'is_run_3_6_class',
    'severity_tier',
    'canonical_slice_index',
]

_CMR_MODALITIES = frozenset({'CMR', 'LGE', 'CINE', 'MRI', 'MR'})


def canonical_slice_index(value: Any) -> Any:
    '''Return a hashable, sortable slice-index representation.'''
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    body = text[1:] if text[:1] in {'+', '-'} else text
    if body.isdigit():
        return int(text)
    return text


def _slice_sort_key(value: Any) -> tuple:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value, '')
    return (1, 0, str(value))


def _sorted_slices(values: Iterable[Any]) -> list:
    return sorted(set(values), key=_slice_sort_key)


def _iter_values(value: Any):
    if value is None:
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _extract_slice_indices(record: Mapping) -> list:
    values = []
    for key in (
        'slice_index',
        'slice_indices',
        'slice_index_list',
        'flagged_slice_indices',
        'slices',
    ):
        if key in record:
            values.extend(_iter_values(record[key]))
    nested = record.get('slice_indices_by_issue')
    if isinstance(nested, Mapping):
        for item in nested.values():
            values.extend(_iter_values(item))
    return [canonical_slice_index(value) for value in values if value is not None]


def _extract_issue_type(record: Mapping) -> str:
    for key in ('issue_type', 'issue', 'flag_type'):
        if key in record and record[key] is not None:
            return str(record[key])
    if 'issue_types' in record:
        value = record['issue_types']
        if isinstance(value, (list, tuple, set, frozenset)):
            return '|'.join(sorted(str(item) for item in value))
        if value is not None:
            return str(value)
    return 'unknown'


def _norm_case_id(value: Any) -> str:
    return str(value).strip() if value is not None else ''


def _norm_modality(value: Any) -> str:
    if value is None:
        return ''
    return ' '.join(str(value).strip().upper().split())


def _norm_issue_type(value: Any) -> str:
    if value is None:
        return 'unknown'
    return ' '.join(str(value).strip().lower().split())


def is_cmr_modality(modality: Any) -> bool:
    return _norm_modality(modality) in _CMR_MODALITIES


def severity_tier(flag_count: int) -> str:
    if flag_count <= 0:
        return 'none'
    if flag_count <= 2:
        return 'low'
    if flag_count <= 5:
        return 'moderate'
    return 'high'


def _worst_slice_index(slices: Iterable[Any]) -> Any:
    values = list(slices)
    if not values:
        return None
    numeric = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    if numeric:
        return max(numeric)
    return max((str(value) for value in values), default=None)


def deduplicate_representative_cases(
    records: Iterable[Mapping],
    *,
    high_volume_threshold: int = 3,
    total_slices_by_case: Mapping | None = None,
) -> dict:
    '''Collapse repeated evidence rows to one row per (case_id, modality).'''
    grouped: dict = {}
    original_row_count = 0
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        original_row_count += 1
        case_id = _norm_case_id(record.get('case_id'))
        modality = _norm_modality(record.get('modality'))
        issue_type = _norm_issue_type(_extract_issue_type(record))
        slices = _extract_slice_indices(record)
        key = (case_id, modality)
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                'case_id': case_id,
                'modality': modality,
                'issue_slices': {},
                'source_row_count': 0,
            }
            grouped[key] = bucket
        bucket['source_row_count'] += 1
        issue_bucket = bucket['issue_slices'].setdefault(issue_type, set())
        issue_bucket.update(slices)

    representative_cases = []
    for key in sorted(grouped.keys(), key=lambda item: (str(item[0]), str(item[1]))):
        bucket = grouped[key]
        issue_slices = {}
        for issue in sorted(bucket['issue_slices']):
            issue_slices[issue] = _sorted_slices(bucket['issue_slices'][issue])
        all_slices = _sorted_slices(
            slice_index
            for slices in issue_slices.values()
            for slice_index in slices
        )
        representative_cases.append(
            {
                'case_id': bucket['case_id'],
                'modality': bucket['modality'],
                'issue_types': sorted(issue_slices.keys()),
                'slice_indices_by_issue': issue_slices,
                'all_slice_indices': all_slices,
                'source_row_count': bucket['source_row_count'],
            }
        )

    deduplicated_row_count = len(representative_cases)
    row_reduction = 0.0
    if original_row_count:
        row_reduction = 1.0 - (deduplicated_row_count / original_row_count)

    stats = {
        'original_row_count': original_row_count,
        'deduplicated_row_count': deduplicated_row_count,
        'row_reduction': row_reduction,
        'row_reduction_pct': round(row_reduction * 100.0, 2),
        'unique_issue_types': sorted(
            {
                issue
                for representative in representative_cases
                for issue in representative['issue_types']
            }
        ),
        'all_slice_indices': _sorted_slices(
            slice_index
            for representative in representative_cases
            for slice_index in representative['all_slice_indices']
        ),
    }

    rollups = rollup_cmr_flags(
        representative_cases,
        total_slices_by_case=total_slices_by_case,
        high_volume_threshold=high_volume_threshold,
    )
    return {
        'representative_cases': representative_cases,
        'case_rollups': rollups,
        'stats': stats,
    }


def rollup_cmr_flags(
    representative_cases: Iterable[Mapping],
    *,
    total_slices_by_case: Mapping | None = None,
    high_volume_threshold: int = 3,
) -> list:
    '''Roll up CMR/LGE flags per case while keeping every slice addressable.'''
    totals = total_slices_by_case or {}
    by_case: dict = {}
    for representative in representative_cases or []:
        modality = _norm_modality(representative.get('modality'))
        if not is_cmr_modality(modality):
            continue
        case_id = _norm_case_id(representative.get('case_id'))
        bucket = by_case.get(case_id)
        if bucket is None:
            bucket = {
                'case_id': case_id,
                'issue_slices': {},
                'modalities': set(),
                'source_row_count': 0,
            }
            by_case[case_id] = bucket
        bucket['modalities'].add(modality)
        bucket['source_row_count'] += int(representative.get('source_row_count', 1) or 0)
        for issue, slices in (representative.get('slice_indices_by_issue') or {}).items():
            issue_bucket = bucket['issue_slices'].setdefault(_norm_issue_type(issue), set())
            issue_bucket.update(slices)

    rollups = []
    for case_id in sorted(by_case.keys(), key=str):
        bucket = by_case[case_id]
        issue_slices = {}
        for issue in sorted(bucket['issue_slices']):
            issue_slices[issue] = _sorted_slices(bucket['issue_slices'][issue])
        flagged = _sorted_slices(
            slice_index
            for slices in issue_slices.values()
            for slice_index in slices
        )
        count = len(flagged)
        rate = None
        if case_id in totals:
            try:
                total = int(totals[case_id])
            except (TypeError, ValueError):
                total = 0
            if total > 0:
                rate = count / total
        rollups.append(
            {
                'case_id': case_id,
                'cmr_flagged_count': count,
                'cmr_flag_rate': rate,
                'worst_slice_index': _worst_slice_index(flagged),
                'severity_tier': severity_tier(count),
                'high_volume': count >= high_volume_threshold,
                'flagged_slice_indices': flagged,
                'slice_indices_by_issue': issue_slices,
                'issue_types': sorted(issue_slices.keys()),
                'modalities': sorted(bucket['modalities']),
                'source_row_count': bucket['source_row_count'],
            }
        )
    return rollups


def round_trip_check(original_records: Iterable[Mapping], deduplicated_payload: Mapping) -> dict:
    '''Verify every original (case, modality, issue, slice) key remains addressable.'''
    original_keys = set()
    for record in original_records or []:
        if not isinstance(record, Mapping):
            continue
        case_id = _norm_case_id(record.get('case_id'))
        modality = _norm_modality(record.get('modality'))
        issue_type = _norm_issue_type(_extract_issue_type(record))
        for slice_index in _extract_slice_indices(record):
            original_keys.add((case_id, modality, issue_type, slice_index))

    deduplicated_keys = set()
    for representative in (deduplicated_payload or {}).get('representative_cases', []) or []:
        case_id = _norm_case_id(representative.get('case_id'))
        modality = _norm_modality(representative.get('modality'))
        for issue, slices in (representative.get('slice_indices_by_issue') or {}).items():
            issue_type = _norm_issue_type(issue)
            for slice_index in slices or []:
                deduplicated_keys.add(
                    (case_id, modality, issue_type, canonical_slice_index(slice_index))
                )

    missing = original_keys - deduplicated_keys
    return {
        'ok': not missing,
        'original_key_count': len(original_keys),
        'deduplicated_key_count': len(deduplicated_keys),
        'missing': [
            {
                'case_id': case_id,
                'modality': modality,
                'issue_type': issue_type,
                'slice_index': slice_index,
            }
            for case_id, modality, issue_type, slice_index in sorted(
                missing,
                key=lambda item: (
                    str(item[0]),
                    str(item[1]),
                    str(item[2]),
                    _slice_sort_key(item[3]),
                ),
            )
        ],
    }


def preservation_report(original_records: Iterable[Mapping], deduplicated_payload: Mapping) -> dict:
    '''Combine issue-type preservation and slice round-trip checks.'''
    original_issues = set()
    for record in original_records or []:
        if isinstance(record, Mapping):
            original_issues.add(_norm_issue_type(_extract_issue_type(record)))
    deduplicated_issues = set()
    for representative in (deduplicated_payload or {}).get('representative_cases', []) or []:
        for issue in representative.get('issue_types', []) or []:
            deduplicated_issues.add(_norm_issue_type(issue))
    round_trip = round_trip_check(original_records, deduplicated_payload)
    missing_issues = original_issues - deduplicated_issues
    return {
        'ok': round_trip['ok'] and not missing_issues,
        'missing_issue_types': sorted(missing_issues),
        'round_trip': round_trip,
    }


def normalize_experiment_class(value: Any) -> str:
    return ' '.join(str(value or '').strip().lower().split())


def _class_tokens(value: Any) -> list:
    text = normalize_experiment_class(value)
    for separator in ('-', '_', '/', '.', ','):
        text = text.replace(separator, ' ')
    return text.split()


def is_run_3_6_class(value: Any) -> bool:
    '''True for run-3, run-4, run-5, run-6 class names, False for run-30/B1.'''
    tokens = _class_tokens(value)
    for index, token in enumerate(tokens):
        if token == 'run' and index + 1 < len(tokens):
            candidate = tokens[index + 1]
            if candidate[:1] in {'3', '4', '5', '6'} and not candidate[1:2].isdigit():
                return True
        if token.startswith('run'):
            rest = token[3:]
            if rest[:1] in {'3', '4', '5', '6'} and not rest[1:2].isdigit():
                return True
    return False


def _matched_run(value: Any) -> str | None:
    tokens = _class_tokens(value)
    for index, token in enumerate(tokens):
        if token == 'run' and index + 1 < len(tokens):
            candidate = tokens[index + 1]
            if candidate[:1] in {'3', '4', '5', '6'} and not candidate[1:2].isdigit():
                return candidate[:1]
        if token.startswith('run'):
            rest = token[3:]
            if rest[:1] in {'3', '4', '5', '6'} and not rest[1:2].isdigit():
                return rest[:1]
    return None


def _entry_experiment_class(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        for key in (
            'experiment_class',
            'class',
            'name',
            'proposal',
            'action_class',
            'module',
            'path',
        ):
            if key in entry and entry[key] is not None:
                value = entry[key]
                if isinstance(value, (list, tuple, set, frozenset)):
                    return ','.join(str(item) for item in value)
                return str(value)
    return str(entry or '')


def guard_iteration_history(history: Iterable[Any], *, new_class: Any = None) -> dict:
    '''Flag re-proposals of run-3/4/5/6 experiment classes.'''
    flagged = []
    allowed = []
    for index, entry in enumerate(history or []):
        experiment_class = _entry_experiment_class(entry)
        if is_run_3_6_class(experiment_class):
            flagged.append(
                {
                    'index': index,
                    'experiment_class': experiment_class,
                    'matched_run': _matched_run(experiment_class),
                    'reason': 're-proposal of run-3/4/5/6 experiment class',
                }
            )
        else:
            allowed.append(
                {
                    'index': index,
                    'experiment_class': experiment_class,
                }
            )
    new_class_text = _entry_experiment_class(new_class) if new_class is not None else None
    new_class_flagged = bool(
        new_class_text is not None and is_run_3_6_class(new_class_text)
    )
    return {
        'flagged': flagged,
        'allowed': allowed,
        'flagged_count': len(flagged),
        'new_class': new_class_text,
        'new_class_flagged': new_class_flagged,
    }
