'''Bounded read-only shadow audit for the cmr_myo_excess_components stratum.

The excess-components stratum has no dedicated primary route, so this module only
re-reads cached shadow summaries produced by runs 4/5/9/13-18 and reports:

* dedup compliance - exactly one rolled-up record per (case_id, modality), with
  merged issue types and the union of the per-slice component indices;
* shadow-confirmed excess-component true-positive rate over a bounded sample of
  at most 50 positive cases plus 10 controls, floor-gated at 50 positives;
* component-count distribution, worst slice and severity tier;
* hash-mismatch and run-9 B1 import-extractor fallback counts;
* same-slice co-occurrence with small_area, border and topology.

Nothing is written back: labels, production rules and QC safety checks are never
touched, and no dataset, credential or runtime database is read. A stale summary
is never re-read with its original extractor - the run-9 B1 import-type extractor
is the only permitted fallback, and every fallback is reported.

Expected cached summary schema::

    {
        'case_id': 'case-0001',
        'modality': 'LGE',
        'run_id': 'run-4',
        'extractor': 'run4_cache',
        'label': 'positive',
        'issue_types': ['excess_components'],
        'source_hash': 'sha256:...',
        'expected_hash': 'sha256:...',
        'hash_mismatch': False,
        'slices': [
            {
                'slice_index': 3,
                'components': 3,
                'component_indices': [0, 1, 2],
                'issue_types': ['excess_components'],
                'small_area': False,
                'border': True,
                'topology': False,
            },
        ],
    }
'''

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Tuple

STRATUM = 'cmr_myo_excess_components'

# Cached shadow-summary runs that may be re-used by this bounded audit.
CACHE_RUN_IDS = (
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

# The only extractor allowed to re-read a summary whose hash does not match.
B1_FALLBACK_RUN = 'run-9'
B1_FALLBACK_EXTRACTOR = 'B1_import'

# Issue-type tokens that mark excess components in cached summaries.
EXCESS_ISSUE_TYPES = frozenset(
    ('excess_components', 'myo_excess_components', 'component_excess')
)

# Same-slice co-occurrence probes reported alongside excess components.
COOCCURRENCE_ISSUES = ('small_area', 'border', 'topology')

# Per-slice boolean flags that survive rollup.
SLICE_FLAG_KEYS = ('small_area', 'border', 'topology')

DEFAULT_MAX_POSITIVE = 50
DEFAULT_MAX_CONTROL = 10
DEFAULT_EXCESS_COMPONENTS_THRESHOLD = 2
DEFAULT_MIN_POSITIVE_SAMPLE = 50

POSITIVE_LABELS = frozenset(('positive', 'pos', 'tp', 'case'))
CONTROL_LABELS = frozenset(('control', 'neg', 'negative', 'fp'))
TIER_ORDER = ('none', 'mild', 'moderate', 'severe')

__all__ = [
    'STRATUM',
    'CACHE_RUN_IDS',
    'B1_FALLBACK_RUN',
    'B1_FALLBACK_EXTRACTOR',
    'EXCESS_ISSUE_TYPES',
    'COOCCURRENCE_ISSUES',
    'SLICE_FLAG_KEYS',
    'DEFAULT_MAX_POSITIVE',
    'DEFAULT_MAX_CONTROL',
    'DEFAULT_EXCESS_COMPONENTS_THRESHOLD',
    'DEFAULT_MIN_POSITIVE_SAMPLE',
    'TIER_ORDER',
    'AuditInputError',
    'severity_tier',
    'hash_mismatch',
    'resolve_extractor',
    'dedup_records',
    'dedup_compliance_rate',
    'select_sample',
    'evaluate_group',
    'audit',
]


class AuditInputError(ValueError):
    '''Raised when a cached shadow summary is malformed or absent.'''


def severity_tier(components: int) -> str:
    '''Severity bucket for a per-slice connected-component count.

    Two or more components means the myocardial structure is split on that
    slice, so tier 'mild' starts at the excess-components threshold.
    '''
    try:
        value = int(components)
    except (TypeError, ValueError):
        raise AuditInputError('components must be an integer')
    if value >= 5:
        return 'severe'
    if value >= 3:
        return 'moderate'
    if value >= 2:
        return 'mild'
    return 'none'


def _require_mapping(record: Any) -> Mapping:
    if not isinstance(record, Mapping):
        raise AuditInputError(
            'cached summary must be a mapping, got {0}'.format(type(record).__name__)
        )
    for field in ('case_id', 'modality'):
        value = record.get(field)
        if value is None or str(value).strip() == '':
            raise AuditInputError(
                'cached summary requires non-empty {0!r}'.format(field)
            )
    return record


def _slice_index(slice_summary: Any) -> int:
    if not isinstance(slice_summary, Mapping):
        raise AuditInputError('slice summary must be a mapping')
    try:
        return int(slice_summary.get('slice_index'))
    except (TypeError, ValueError):
        raise AuditInputError("slice summary requires an integer 'slice_index'")


def _components(slice_summary: Mapping) -> int:
    try:
        components = int(slice_summary.get('components', 0))
    except (TypeError, ValueError):
        raise AuditInputError("slice summary 'components' must be an integer")
    if components < 0:
        raise AuditInputError("slice summary 'components' must be >= 0")
    return components


def _normalise_label(record: Mapping) -> str:
    raw = record.get('label')
    if raw is None:
        return 'unknown'
    text = str(raw).strip().lower()
    if text in POSITIVE_LABELS:
        return 'positive'
    if text in CONTROL_LABELS:
        return 'control'
    return 'unknown'


def _iter_issue_types(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, Iterable):
        for item in value:
            text = str(item).strip()
            if text:
                yield text


def hash_mismatch(record: Mapping) -> bool:
    '''True when a cached summary is stale relative to its expected hash.

    An explicit 'hash_mismatch' flag wins over hash comparison, so callers can
    pin the decision for summaries that carry no hashes at all.
    '''
    flag = record.get('hash_mismatch')
    if flag is not None:
        return bool(flag)
    source = record.get('source_hash')
    expected = record.get('expected_hash')
    if not source or not expected:
        return False
    return str(source) != str(expected)


def resolve_extractor(record: Mapping) -> Tuple[str, bool]:
    '''Return (extractor, used_b1_fallback) for one cached summary.

    On a hash mismatch the original extractor is never re-used: the run-9 B1
    import-type extractor is the only permitted reader, and the caller is told
    that a fallback happened so it can be reported.
    '''
    if hash_mismatch(record):
        return B1_FALLBACK_EXTRACTOR, True
    return str(record.get('extractor', '')), False


def dedup_records(records: Iterable[Mapping]) -> List[Dict[str, Any]]:
    '''Roll cached summaries up to one record per (case_id, modality).

    Issue types are merged, per-slice component indices are unioned, per-slice
    component counts take the maximum observed value, and every slice flag is
    OR-combined. The input mappings are never mutated.
    '''
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str]] = []
    for raw in records:
        record = _require_mapping(raw)
        key = (str(record['case_id']).strip(), str(record['modality']).strip())
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                'case_id': key[0],
                'modality': key[1],
                'labels': set(),
                'issue_types': set(),
                'slices': {},
                'run_ids': set(),
                'extractors': set(),
                'hash_mismatch': False,
                'b1_fallback': False,
                'input_rows': 0,
            }
            buckets[key] = bucket
            order.append(key)
        bucket['input_rows'] += 1
        bucket['labels'].add(_normalise_label(record))
        run_id = record.get('run_id')
        if run_id:
            bucket['run_ids'].add(str(run_id))
        extractor, used_b1 = resolve_extractor(record)
        if extractor:
            bucket['extractors'].add(extractor)
        if used_b1:
            bucket['b1_fallback'] = True
            bucket['hash_mismatch'] = True
        for token in _iter_issue_types(record.get('issue_types')):
            bucket['issue_types'].add(token)
        for slice_summary in record.get('slices') or ():
            index = _slice_index(slice_summary)
            slot = bucket['slices'].get(index)
            if slot is None:
                slot = {
                    'slice_index': index,
                    'components': 0,
                    'component_indices': set(),
                    'issue_types': set(),
                    'small_area': False,
                    'border': False,
                    'topology': False,
                }
                bucket['slices'][index] = slot
            components = _components(slice_summary)
            if components > slot['components']:
                slot['components'] = components
            for idx in slice_summary.get('component_indices') or ():
                try:
                    slot['component_indices'].add(int(idx))
                except (TypeError, ValueError):
                    raise AuditInputError(
                        "'component_indices' entries must be integers"
                    )
            for token in _iter_issue_types(slice_summary.get('issue_types')):
                slot['issue_types'].add(token)
                bucket['issue_types'].add(token)
            for flag in SLICE_FLAG_KEYS:
                if bool(slice_summary.get(flag)):
                    slot[flag] = True
    result: List[Dict[str, Any]] = []
    for key in order:
        bucket = buckets[key]
        labels = bucket['labels']
        if 'positive' in labels:
            label = 'positive'
        elif 'control' in labels:
            label = 'control'
        else:
            label = 'unknown'
        slices: List[Dict[str, Any]] = []
        for index in sorted(bucket['slices']):
            slot = bucket['slices'][index]
            slices.append(
                {
                    'slice_index': slot['slice_index'],
                    'components': slot['components'],
                    'component_indices': sorted(slot['component_indices']),
                    'issue_types': sorted(slot['issue_types']),
                    'small_area': bool(slot['small_area']),
                    'border': bool(slot['border']),
                    'topology': bool(slot['topology']),
                }
            )
        issue_types = sorted(bucket['issue_types'])
        result.append(
            {
                'case_id': bucket['case_id'],
                'modality': bucket['modality'],
                'label': label,
                'issue_types': issue_types,
                'has_excess_issue': bool(
                    set(issue_types).intersection(EXCESS_ISSUE_TYPES)
                ),
                'slices': slices,
                'run_ids': sorted(bucket['run_ids']),
                'extractors': sorted(bucket['extractors']),
                'hash_mismatch': bool(bucket['hash_mismatch']),
                'b1_fallback': bool(bucket['b1_fallback']),
                'input_rows': bucket['input_rows'],
            }
        )
    result.sort(key=lambda item: (item['case_id'], item['modality']))
    return result


def dedup_compliance_rate(records: Iterable[Mapping]) -> float:
    '''1.0 when the rollup yields exactly one row per (case_id, modality).'''
    deduped = dedup_records(records)
    keys = [(item['case_id'], item['modality']) for item in deduped]
    if not keys:
        return 1.0
    return 1.0 if len(set(keys)) == len(keys) else 0.0


def select_sample(
    records: Iterable[Mapping],
    max_positive: int = DEFAULT_MAX_POSITIVE,
    max_control: int = DEFAULT_MAX_CONTROL,
) -> Dict[str, Any]:
    '''Bounded, deterministic sample: at most max_positive plus max_control.'''
    rows = list(records)
    positives = [row for row in rows if row.get('label') == 'positive']
    controls = [row for row in rows if row.get('label') == 'control']
    positives.sort(key=lambda item: (str(item.get('case_id')), str(item.get('modality'))))
    controls.sort(key=lambda item: (str(item.get('case_id')), str(item.get('modality'))))
    positive_limit = max(0, int(max_positive))
    control_limit = max(0, int(max_control))
    selected_positives = positives[:positive_limit]
    selected_controls = controls[:control_limit]
    return {
        'positives': selected_positives,
        'controls': selected_controls,
        'positive_pool': len(positives),
        'control_pool': len(controls),
        'positive_truncated': len(positives) > len(selected_positives),
        'control_truncated': len(controls) > len(selected_controls),
    }


def evaluate_group(
    records: Iterable[Mapping],
    threshold: int = DEFAULT_EXCESS_COMPONENTS_THRESHOLD,
    expected_min_sample: int = DEFAULT_MIN_POSITIVE_SAMPLE,
) -> Dict[str, Any]:
    '''Shadow metrics for one label group over already-rolled-up records.'''
    group = list(records)
    threshold = int(threshold)
    if threshold < 1:
        raise AuditInputError('excess-components threshold must be >= 1')
    component_counts: Counter = Counter()
    max_component_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    worst_slice_histogram: Counter = Counter()
    worst_slices: List[Dict[str, Any]] = []
    confirmed_case_ids: List[str] = []
    unconfirmed_case_ids: List[str] = []
    excess_slice_count = 0
    cooccurrence_counts = {name: 0 for name in COOCCURRENCE_ISSUES}
    for record in group:
        slices = list(record.get('slices') or ())
        counts = [int(slice_summary.get('components', 0)) for slice_summary in slices]
        for value in counts:
            component_counts[value] += 1
        max_components = max(counts) if counts else 0
        max_component_counts[max_components] += 1
        severity_counts[severity_tier(max_components)] += 1
        worst = None
        for slice_summary in slices:
            if worst is None or int(slice_summary.get('components', 0)) > int(
                worst.get('components', 0)
            ):
                worst = slice_summary
        if worst is not None:
            worst_index = int(worst.get('slice_index'))
            worst_slices.append(
                {
                    'case_id': str(record.get('case_id')),
                    'slice_index': worst_index,
                    'components': int(worst.get('components', 0)),
                }
            )
            worst_slice_histogram[worst_index] += 1
        if max_components >= threshold:
            confirmed_case_ids.append(str(record.get('case_id')))
        else:
            unconfirmed_case_ids.append(str(record.get('case_id')))
        for slice_summary in slices:
            if int(slice_summary.get('components', 0)) >= threshold:
                excess_slice_count += 1
                for name in COOCCURRENCE_ISSUES:
                    if bool(slice_summary.get(name)):
                        cooccurrence_counts[name] += 1
    sample_size = len(group)
    confirmed_count = len(confirmed_case_ids)
    true_positive_rate = (confirmed_count / sample_size) if sample_size else None
    max_tier = 'none'
    for name in TIER_ORDER:
        if severity_counts.get(name):
            max_tier = name
    cooccurrence = {
        name: (
            cooccurrence_counts[name] / excess_slice_count
            if excess_slice_count
            else 0.0
        )
        for name in COOCCURRENCE_ISSUES
    }
    return {
        'sample_size': sample_size,
        'expected_min_sample': int(expected_min_sample),
        'insufficient_sample': sample_size < int(expected_min_sample),
        'excess_components_threshold': threshold,
        'confirmed_count': confirmed_count,
        'unconfirmed_count': len(unconfirmed_case_ids),
        'true_positive_rate': true_positive_rate,
        'confirmed_case_ids': sorted(confirmed_case_ids),
        'unconfirmed_case_ids': sorted(unconfirmed_case_ids),
        'component_count_distribution': {
            key: component_counts[key] for key in sorted(component_counts)
        },
        'max_component_count_distribution': {
            key: max_component_counts[key] for key in sorted(max_component_counts)
        },
        'severity_tier_distribution': {
            name: severity_counts.get(name, 0) for name in TIER_ORDER
        },
        'max_severity_tier': max_tier,
        'worst_slice_histogram': {
            key: worst_slice_histogram[key] for key in sorted(worst_slice_histogram)
        },
        'worst_slices': worst_slices,
        'excess_slice_count': excess_slice_count,
        'cooccurrence': cooccurrence,
        'cooccurrence_counts': {
            name: cooccurrence_counts[name] for name in COOCCURRENCE_ISSUES
        },
    }


def audit(
    records: Iterable[Mapping],
    max_positive: int = DEFAULT_MAX_POSITIVE,
    max_control: int = DEFAULT_MAX_CONTROL,
    threshold: int = DEFAULT_EXCESS_COMPONENTS_THRESHOLD,
    expected_min_sample: int = DEFAULT_MIN_POSITIVE_SAMPLE,
) -> Dict[str, Any]:
    '''Read-only shadow audit report for the excess-components stratum.'''
    raw = list(records)
    deduped = dedup_records(raw)
    sample = select_sample(deduped, max_positive=max_positive, max_control=max_control)
    positive_metrics = evaluate_group(
        sample['positives'], threshold=threshold, expected_min_sample=expected_min_sample
    )
    control_metrics = evaluate_group(
        sample['controls'], threshold=threshold, expected_min_sample=1
    )
    hash_mismatch_cases = [item['case_id'] for item in deduped if item['hash_mismatch']]
    fallback_cases = [item['case_id'] for item in deduped if item['b1_fallback']]
    return {
        'stratum': STRATUM,
        'read_only': True,
        'cache_runs_reused': list(CACHE_RUN_IDS),
        'dedup': {
            'input_rows': len(raw),
            'unique_keys': len(deduped),
            'collapsed_rows': len(raw) - len(deduped),
            'dedup_compliance_rate': dedup_compliance_rate(raw),
            'records': deduped,
        },
        'sample': {
            'max_positive': int(max_positive),
            'max_control': int(max_control),
            'positive_pool': sample['positive_pool'],
            'control_pool': sample['control_pool'],
            'positive_sample_size': len(sample['positives']),
            'control_sample_size': len(sample['controls']),
            'positive_truncated': sample['positive_truncated'],
            'control_truncated': sample['control_truncated'],
            'positive_case_ids': [row['case_id'] for row in sample['positives']],
            'control_case_ids': [row['case_id'] for row in sample['controls']],
        },
        'positives': positive_metrics,
        'controls': control_metrics,
        'hash_provenance': {
            'hash_mismatch_count': len(hash_mismatch_cases),
            'hash_mismatch_cases': sorted(set(hash_mismatch_cases)),
            'b1_fallback_count': len(fallback_cases),
            'b1_fallback_cases': sorted(set(fallback_cases)),
            'b1_fallback_run': B1_FALLBACK_RUN,
            'b1_fallback_extractor': B1_FALLBACK_EXTRACTOR,
        },
        'cooccurrence': positive_metrics['cooccurrence'],
    }
