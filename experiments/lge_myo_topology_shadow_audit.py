'''Bounded read-only shadow audit for LGE myo_topology (n=578).

The module consumes already-materialized cache summaries. It never mutates
labels, production rules, datasets, patient images or runtime databases.
'''

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

SHADOW_ONLY = True
ALLOWED_CACHE_RUNS = frozenset({4, 5, 9, 13, 14, 15, 16, 17, 18})
FALLBACK_RUN9_B1 = 'run-9:B1'
DEFAULT_CACHE_REUSE_TARGET = 0.90


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 't'}


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get('case_id', '')), str(row.get('modality', '')))


def _iter_slices(row: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    slices = row.get('slices')
    if isinstance(slices, Sequence) and not isinstance(slices, (str, bytes)):
        for item in slices:
            if isinstance(item, Mapping):
                yield item
        return
    yield row


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = str(row.get('case_id', '')).strip()
        modality = str(row.get('modality', '')).strip()
        if not case_id or not modality:
            continue
        slices = []
        for item in _iter_slices(row):
            slices.append(
                {
                    'slice_index': _as_int(item.get('slice_index')),
                    'mi_multiple_components': _as_bool(item.get('mi_multiple_components')),
                    'mi_fraction_high': _as_bool(item.get('mi_fraction_high')),
                }
            )
        normalized.append(
            {
                'case_id': case_id,
                'modality': modality,
                'stratum': str(row.get('stratum', '')).strip().lower(),
                'run_id': row.get('run_id'),
                'cache_hit': _as_bool(row.get('cache_hit')),
                'expected_hash': row.get('expected_hash'),
                'observed_hash': row.get('observed_hash'),
                'slices': slices,
            }
        )
    return normalized


def load_cache_summaries(path: str) -> list[dict[str, Any]]:
    '''Load JSON, JSON array or JSONL cache summaries from an explicit path.'''
    with open(path, 'r', encoding='utf-8') as handle:
        text = handle.read().strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError('cache summary must be a JSON object, JSON array or JSONL')
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def _slice_rate(slices: Sequence[Mapping[str, Any]]) -> float:
    if not slices:
        return 0.0
    return sum(1 for item in slices if item['mi_multiple_components']) / len(slices)


def _case_support_rate(support: set, total: int) -> float:
    if total <= 0:
        return 0.0
    return len(support) / total


def audit_topology_shadow(
    rows: Iterable[Mapping[str, Any]],
    *,
    positive_n: int = 50,
    control_n: int = 10,
    cache_reuse_target: float = DEFAULT_CACHE_REUSE_TARGET,
) -> dict[str, Any]:
    normalized = normalize_rows(rows)

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    duplicate_rows = 0
    for row in normalized:
        key = _row_key(row)
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        unique.append(row)

    stratum_slices: dict[str, list[Mapping[str, Any]]] = {'positive': [], 'control': []}
    stratum_cases: dict[str, set[tuple[str, str]]] = {'positive': set(), 'control': set()}
    high_support: dict[str, set[tuple[str, str]]] = {'positive': set(), 'control': set()}
    slice_index_unaddressable = 0

    for row in unique:
        stratum = row['stratum']
        if stratum not in stratum_slices:
            continue
        key = _row_key(row)
        stratum_cases[stratum].add(key)
        for item in row['slices']:
            if item['slice_index'] is None:
                slice_index_unaddressable += 1
            stratum_slices[stratum].append(item)
            if item['mi_fraction_high']:
                high_support[stratum].add(key)

    cache_hits = 0
    fallback_count = 0
    fallback_reasons: Counter[str] = Counter()
    fallback_events: list[dict[str, Any]] = []

    for row in unique:
        run_id = _as_int(row.get('run_id'))
        expected = row.get('expected_hash')
        observed = row.get('observed_hash')
        hash_mismatch = bool(expected) and bool(observed) and str(expected) != str(observed)

        if hash_mismatch:
            reason: Optional[str] = 'hash_mismatch'
        elif not row['cache_hit']:
            reason = 'cache_miss'
        elif run_id not in ALLOWED_CACHE_RUNS:
            reason = 'run_not_in_allowed_cache_set'
        else:
            reason = None

        if reason is None:
            cache_hits += 1
        else:
            fallback_reasons[reason] += 1
            if reason == 'hash_mismatch':
                fallback_count += 1
                fallback_events.append(
                    {
                        'case_id': row['case_id'],
                        'modality': row['modality'],
                        'run_id': run_id,
                        'reason': reason,
                        'fallback_extractor': FALLBACK_RUN9_B1,
                        'expected_hash': expected,
                        'observed_hash': observed,
                    }
                )

    total_rows = len(unique)
    reuse_rate = cache_hits / total_rows if total_rows else 0.0
    positive_rate = _slice_rate(stratum_slices['positive'])
    control_rate = _slice_rate(stratum_slices['control'])
    positive_high_rate = _case_support_rate(
        high_support['positive'], len(stratum_cases['positive'])
    )
    control_high_rate = _case_support_rate(
        high_support['control'], len(stratum_cases['control'])
    )

    return {
        'shadow_only': SHADOW_ONLY,
        'source': 'lge_myo_topology_shadow_audit',
        'counts': {
            'rows': len(normalized),
            'unique_case_modality_rows': total_rows,
            'duplicate_rows': duplicate_rows,
            'positive_cases': len(stratum_cases['positive']),
            'control_cases': len(stratum_cases['control']),
            'positive_expected': positive_n,
            'control_expected': control_n,
        },
        'cooccurrence': {
            'positive_slice_count': len(stratum_slices['positive']),
            'control_slice_count': len(stratum_slices['control']),
            'positive_mi_multiple_components_rate': positive_rate,
            'control_mi_multiple_components_rate': control_rate,
            'difference': positive_rate - control_rate,
        },
        'case_level_mi_fraction_high_support': {
            'positive_support_rate': positive_high_rate,
            'control_support_rate': control_high_rate,
            'difference': positive_high_rate - control_high_rate,
        },
        'cache_reuse': {
            'target': cache_reuse_target,
            'total_rows': total_rows,
            'hits': cache_hits,
            'reuse_rate': reuse_rate,
            'meets_target': reuse_rate >= cache_reuse_target,
            'fallback_count': fallback_count,
            'fallback_reasons': dict(fallback_reasons),
            'hash_mismatch_fallbacks': fallback_count,
            'fallback_extractor': FALLBACK_RUN9_B1,
            'fallback_events': fallback_events,
        },
        'addressability': {
            'slice_index_unaddressable': slice_index_unaddressable,
            'all_slices_addressable': slice_index_unaddressable == 0,
        },
        'uniqueness_ok': duplicate_rows == 0,
        'ready_for_shadow_review': duplicate_rows == 0 and slice_index_unaddressable == 0,
    }
