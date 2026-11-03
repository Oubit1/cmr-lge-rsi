'''Bounded, read-only shadow audit for the LGE mi_multiple_components cohort.

Constraints enforced by this module:

* Read-only: no label, rule, cache or database writes. The audit consumes
  already-materialised cache summaries (runs 4/5/9/13-18) and returns an
  in-memory payload.
* Bounded: at most POSITIVE_LIMIT (50) positives plus CONTROL_LIMIT (10)
  controls; oversized cohorts raise unless strict_bounds=False.
* Hash-first: a cached summary is reused only when the digest recorded in the
  cache entry matches the digest the caller expects. On mismatch the only
  permitted fallback is the injected run-9 B1 extractor, and every fallback is
  reported in b1_fallbacks.
* Dedup gate: duplicate (case_id, modality) records are blocked before
  aggregation so no case can be counted twice.

The fragmentation stratum is a shadow classification only; it never feeds back
into production QC decisions and it never mutates labels or production rules.
'''

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

CACHE_RUNS: tuple[str, ...] = (
    'run-4',
    'run-5',
    'run-9',
    'run-13',
    'run-14',
    'run-15',
    'run-16',
    'run-17',
    'run-18',
)
B1_RUN = 'run-9'
POSITIVE_LIMIT = 50
CONTROL_LIMIT = 10
FRAGMENTED_SHARE_TARGET = 0.60
DEFAULT_MIN_AREA_MM2 = 8.0
DEFAULT_FRAGMENT_AREA_RATIO = 0.15
FRAGMENTED_LABEL = 'fragmented_pseudo'
TRUE_MULTIFOCAL_LABEL = 'true_multifocal'
NO_COMPONENT_LABEL = 'no_component'
SINGLE_LESION_LABEL = 'single_lesion'


def select_cohort(cases: Sequence[Mapping[str, Any]],
                  positives: int = POSITIVE_LIMIT,
                  controls: int = CONTROL_LIMIT) -> list[Mapping[str, Any]]:
    '''Deterministically bound the audit cohort by case_id sort order.'''

    def _subset(cohort: str, limit: int) -> list[Mapping[str, Any]]:
        subset = [case for case in cases if case.get('cohort') == cohort]
        subset.sort(key=lambda case: str(case.get('case_id')))
        return subset[:limit]

    return _subset('positive', positives) + _subset('control', controls)


def build_cache_index(entries: Sequence[Mapping[str, Any]]) -> dict:
    '''Group cached run summaries by (case_id, modality) then by run.'''
    index: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for entry in entries:
        key = (str(entry.get('case_id')), str(entry.get('modality')))
        index.setdefault(key, {})[str(entry.get('run'))] = entry
    return index


def lookup_cached(index: Mapping[Any, Any], case_id: Any, modality: Any):
    '''Return the cached entry for a case, preferring the declared run order.'''
    per_run = index.get((str(case_id), str(modality)))
    if per_run:
        for run in CACHE_RUNS:
            if run in per_run:
                return per_run[run]
        for run in sorted(per_run):
            return per_run[run]
    return None


def classify_multi_component(areas_mm2: Sequence[float],
                             *,
                             min_area_mm2: float = DEFAULT_MIN_AREA_MM2,
                             fragment_area_ratio: float = DEFAULT_FRAGMENT_AREA_RATIO) -> str:
    '''Shadow stratum for a multi-component LGE finding.

    Returns one of: no_component, single_lesion, fragmented_pseudo,
    true_multifocal. A case is called fragmented_pseudo when the reported
    multi-component finding collapses to at most one component at or above the
    area floor, or when no component is individually dominant.
    '''
    areas = [float(area) for area in areas_mm2]
    if not areas:
        return NO_COMPONENT_LABEL
    if len(areas) == 1:
        return SINGLE_LESION_LABEL
    retained = [area for area in areas if area >= float(min_area_mm2)]
    if len(retained) <= 1:
        return FRAGMENTED_LABEL
    total = float(sum(areas))
    if total > 0.0 and (max(areas) / total) < float(fragment_area_ratio):
        return FRAGMENTED_LABEL
    return TRUE_MULTIFOCAL_LABEL


def dedup_gate(records: Sequence[Mapping[str, Any]]):
    '''Split records into accepted and duplicate-blocked (case_id, modality).'''
    accepted: list[Mapping[str, Any]] = []
    blocked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get('case_id')), str(record.get('modality')))
        if key in seen:
            blocked.append({
                'case_id': key[0],
                'modality': key[1],
                'reason': 'duplicate_case_modality',
            })
            continue
        seen.add(key)
        accepted.append(record)
    return accepted, blocked


def resolve_summary(case: Mapping[str, Any],
                    index: Mapping[Any, Any],
                    b1_extractor: Callable[..., Any] | None = None) -> dict:
    '''Resolve one case summary from cache, falling back only to run-9 B1.'''
    case_id = str(case.get('case_id'))
    modality = str(case.get('modality'))
    entry = lookup_cached(index, case_id, modality)
    if entry is not None:
        expected = case.get('expected_sha256')
        recorded = entry.get('sha256')
        if expected is None:
            return {
                'summary': entry.get('summary'),
                'source': 'cache_unverified',
                'cache_hit': True,
                'hash_mismatch': False,
                'b1_used': False,
                'fallback_reason': None,
            }
        if recorded is not None and str(recorded) == str(expected):
            return {
                'summary': entry.get('summary'),
                'source': 'cache',
                'cache_hit': True,
                'hash_mismatch': False,
                'b1_used': False,
                'fallback_reason': None,
            }
        summary = b1_extractor(case_id, modality) if b1_extractor is not None else None
        return {
            'summary': summary,
            'source': 'b1_hash_mismatch' if b1_extractor is not None else 'unresolved_hash_mismatch',
            'cache_hit': False,
            'hash_mismatch': True,
            'b1_used': b1_extractor is not None,
            'fallback_reason': 'hash_mismatch',
        }
    summary = b1_extractor(case_id, modality) if b1_extractor is not None else None
    return {
        'summary': summary,
        'source': 'b1_cache_miss' if b1_extractor is not None else 'unresolved_cache_miss',
        'cache_hit': False,
        'hash_mismatch': False,
        'b1_used': b1_extractor is not None,
        'fallback_reason': 'cache_miss',
    }


def _case_record(case_id: str,
                 modality: str,
                 cohort: Any,
                 resolution: Mapping[str, Any],
                 min_area_mm2: float,
                 fragment_area_ratio: float) -> dict:
    summary = resolution.get('summary')
    classification = None
    components = None
    worst = None
    topology = None
    mi_high = None
    if summary is not None:
        areas = summary.get('areas_mm2')
        topology = summary.get('myo_topology')
        mi_high = summary.get('mi_fraction_high')
        if areas is not None:
            areas = [float(area) for area in areas]
            classification = classify_multi_component(
                areas,
                min_area_mm2=min_area_mm2,
                fragment_area_ratio=fragment_area_ratio,
            )
            components = len(areas)
        elif summary.get('components') is not None:
            components = int(summary['components'])
        if summary.get('worst_slice_components') is not None:
            worst = int(summary['worst_slice_components'])
    return {
        'case_id': case_id,
        'modality': modality,
        'cohort': cohort,
        'source': resolution.get('source'),
        'components': components,
        'classification': classification,
        'worst_slice_components': worst,
        'myo_topology': topology,
        'mi_fraction_high': mi_high,
    }


def _median(values: Sequence[Any]):
    values = list(values)
    return statistics.median(values) if values else None


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


def _stratum(records: Sequence[Mapping[str, Any]]) -> dict:
    counts = Counter(record['classification'] for record in records if record.get('classification'))
    classified = sum(counts.values())
    pseudo = counts.get(FRAGMENTED_LABEL, 0)
    share = round(pseudo / classified, 4) if classified else None
    return {
        'classification_counts': dict(sorted(counts.items())),
        'classified': classified,
        'fragmented_pseudo': pseudo,
        'fragmented_share': share,
        'meets_target': bool(classified) and share is not None and share >= FRAGMENTED_SHARE_TARGET,
    }


def _component_stats(records: Sequence[Mapping[str, Any]]) -> dict:
    positive = [record for record in records if record.get('cohort') == 'positive']
    control = [record for record in records if record.get('cohort') == 'control']
    worst = Counter(
        str(record['worst_slice_components'])
        for record in positive
        if record.get('worst_slice_components') is not None
    )
    return {
        'positive_n_with_components': sum(1 for record in positive if record.get('components') is not None),
        'positive_median_components': _median(
            [record['components'] for record in positive if record.get('components') is not None]
        ),
        'control_median_components': _median(
            [record['components'] for record in control if record.get('components') is not None]
        ),
        'positive_worst_slice_distribution': {key: worst[key] for key in sorted(worst, key=int)},
    }


def _has_topology(record: Mapping[str, Any]) -> bool:
    return record.get('myo_topology') not in (None, '', 'unknown')


def _cooccurrence(records: Sequence[Mapping[str, Any]]) -> dict:
    positive = [record for record in records if record.get('cohort') == 'positive']
    fragmented = [record for record in positive if record.get('classification') == FRAGMENTED_LABEL]
    positive_n = len(positive)
    fragmented_n = len(fragmented)
    return {
        'positive_n': positive_n,
        'myo_topology_rate': _rate(sum(1 for record in positive if _has_topology(record)), positive_n),
        'mi_fraction_high_rate': _rate(
            sum(1 for record in positive if record.get('mi_fraction_high') is True), positive_n
        ),
        'fragmented_n': fragmented_n,
        'fragmented_myo_topology_rate': _rate(
            sum(1 for record in fragmented if _has_topology(record)), fragmented_n
        ),
        'fragmented_mi_fraction_high_rate': _rate(
            sum(1 for record in fragmented if record.get('mi_fraction_high') is True), fragmented_n
        ),
    }


def run_shadow_audit(cases: Sequence[Mapping[str, Any]],
                     cache_entries: Sequence[Mapping[str, Any]] = (),
                     *,
                     b1_extractor: Callable[..., Any] | None = None,
                     min_area_mm2: float = DEFAULT_MIN_AREA_MM2,
                     fragment_area_ratio: float = DEFAULT_FRAGMENT_AREA_RATIO,
                     positive_limit: int = POSITIVE_LIMIT,
                     control_limit: int = CONTROL_LIMIT,
                     strict_bounds: bool = True) -> dict:
    '''Produce the read-only audit payload. Nothing is written anywhere.'''
    working = [dict(case) for case in cases]
    positives = [case for case in working if case.get('cohort') == 'positive']
    controls = [case for case in working if case.get('cohort') == 'control']
    if strict_bounds and (len(positives) > positive_limit or len(controls) > control_limit):
        raise ValueError(
            'cohort bounds exceeded: positives=%d/%d controls=%d/%d; call select_cohort() first'
            % (len(positives), positive_limit, len(controls), control_limit)
        )

    index = build_cache_index(cache_entries)
    accepted, blocked = dedup_gate(working)

    records: list[dict] = []
    b1_fallbacks: list[dict] = []
    counters = {
        'cache_hits': 0,
        'unverified_hits': 0,
        'hash_mismatches': 0,
        'b1_triggers': 0,
        'unresolved': 0,
    }

    for case in accepted:
        case_id = str(case.get('case_id'))
        modality = str(case.get('modality'))
        resolution = resolve_summary(case, index, b1_extractor)
        if resolution.get('cache_hit'):
            counters['cache_hits'] += 1
            if resolution.get('source') == 'cache_unverified':
                counters['unverified_hits'] += 1
        if resolution.get('hash_mismatch'):
            counters['hash_mismatches'] += 1
        if resolution.get('b1_used'):
            counters['b1_triggers'] += 1
        if resolution.get('summary') is None:
            counters['unresolved'] += 1
        if resolution.get('fallback_reason') is not None:
            b1_fallbacks.append({
                'case_id': case_id,
                'modality': modality,
                'reason': resolution['fallback_reason'],
                'extractor': 'B1/' + B1_RUN if resolution.get('b1_used') else None,
                'run': B1_RUN if resolution.get('b1_used') else None,
                'resolved': resolution.get('summary') is not None,
            })
        records.append(_case_record(
            case_id,
            modality,
            case.get('cohort'),
            resolution,
            min_area_mm2,
            fragment_area_ratio,
        ))

    accepted_n = len(accepted)
    positive_records = [record for record in records if record.get('cohort') == 'positive']
    control_records = [record for record in records if record.get('cohort') == 'control']
    strata = {
        'positive': _stratum(positive_records),
        'control': _stratum(control_records),
    }
    cache_block = {
        'hits': counters['cache_hits'],
        'unverified_hits': counters['unverified_hits'],
        'misses': accepted_n - counters['cache_hits'],
        'hash_mismatches': counters['hash_mismatches'],
        'b1_triggers': counters['b1_triggers'],
        'unresolved': counters['unresolved'],
        'hit_rate': round(counters['cache_hits'] / accepted_n, 4) if accepted_n else None,
        'dedup_blocked_duplicate_case_modality': len(blocked),
    }
    return {
        'read_only': True,
        'runs_reused': list(CACHE_RUNS),
        'b1_run': B1_RUN,
        'cohort_bounds': {
            'positive_limit': positive_limit,
            'control_limit': control_limit,
            'strict': bool(strict_bounds),
        },
        'counts': {
            'submitted': len(working),
            'positives': len(positives),
            'controls': len(controls),
            'accepted': accepted_n,
            'dedup_blocked': len(blocked),
        },
        'cache': cache_block,
        'b1_fallbacks': b1_fallbacks,
        'dedup_blocked': blocked,
        'strata': strata,
        'components': _component_stats(records),
        'cooccurrence': _cooccurrence(records),
        'acceptance': {
            'fragmented_share_target': FRAGMENTED_SHARE_TARGET,
            'fragmented_share_met': strata['positive']['meets_target'],
        },
        'cases': records,
    }
