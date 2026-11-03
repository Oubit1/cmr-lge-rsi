'''Bounded read-only main-provenance shadow audit for cmr_myo_missing_temporal_hole.

Scope and guardrails
--------------------
* READ-ONLY: this module never writes to disk, never mutates QC labels or the
  production routing rules, and never touches datasets, patient images or
  runtime databases. It consumes in-memory cache summaries supplied by the
  caller and asserts at the end of the audit that its input was not mutated.
* SINGLE TOOLSET: when a cache hash does not match, the only permitted recovery
  path is the unique run-9 'B1 contract' import-style extractor defined here
  (``extract_via_run9_b1_contract``). No second extractor or toolset may be
  introduced; mismatches that cannot be recovered through that contract are
  reported as unrecoverable instead of being repaired.
* BOUNDED: the audited cohort is capped at 50 positive + 10 control cases and
  only cache summaries produced by runs 4/5/9/13-18 are reused.

Canonical record shape consumed by this module (already loaded in memory)::

    {
      'case_id': 'C-001',
      'modality': 'LGE',
      'run_id': 'run-4',
      'cache_payload': {...},        # summary that was hashed at cache time
      'cache_hash': '<sha256>',
      'slices': [
        {
          'slice_index': 3,
          'markers': ['cmr_myo_missing_temporal_hole', ...],
          'holes': [
            {'hole_id': 'h1', 'frames': [3, 4], 'slices': [3, 4], 'area_mm2': 7.5}
          ]
        }
      ]
    }

Each hole record is anchored to a single slice record and carries its temporal
span in ``frames`` / ``slices``. A hole is 'persistent' when its span covers at
least two adjacent frames or slices; otherwise it is a single-frame/slice
isolated hole. An explicit boolean ``persistent`` flag, when present, wins.
'''

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

AUDIT_MODE = 'read_only_shadow'
HOLE_MARKER = 'cmr_myo_missing_temporal_hole'
COOCCURRENCE_MARKERS: Tuple[str, ...] = (
    'cmr_myo_empty',
    'cmr_myo_small_area',
    'cmr_centroid_drift',
    'cmr_layer_area_jump',
)
REUSED_CACHE_RUNS: Tuple[str, ...] = (
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
MAX_POSITIVE_CASES = 50
MAX_CONTROL_CASES = 10
RUN9_B1_RUN_ID = 'run-9'
RUN9_B1_CONTRACT_VERSION = 'b1'
PERSISTENCE_MIN_SPAN = 2


def _ratio(numerator: int, denominator: int) -> float:
    '''Return numerator/denominator rounded to 6 dp; 0.0 for empty denominators.'''
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def canonical_json(payload: Any) -> str:
    '''Deterministic JSON rendering used for cache hashing (key order independent).'''
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def canonical_hash(payload: Any) -> str:
    '''SHA-256 over the canonical JSON rendering of a cache summary payload.'''
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()


def case_modality_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    '''Addressing key: exactly one record per (case_id, modality).'''
    return (str(record.get('case_id')), str(record.get('modality')))


def filter_reused_runs(
    records: Iterable[Mapping[str, Any]],
    allowed_runs: Sequence[str] = REUSED_CACHE_RUNS,
) -> Dict[str, Any]:
    '''Keep only cache summaries produced by the permitted runs (read-only filter).'''
    allowed = set(allowed_runs)
    kept: List[Mapping[str, Any]] = []
    skipped: List[Mapping[str, Any]] = []
    for record in records:
        if record.get('run_id') in allowed:
            kept.append(record)
        else:
            skipped.append(record)
    return {
        'kept': kept,
        'skipped_case_ids': sorted({str(r.get('case_id')) for r in skipped}),
        'skipped_runs': sorted({str(r.get('run_id')) for r in skipped}),
        'skipped_count': len(skipped),
    }


def enforce_single_record_per_case_modality(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    '''Keep the first record per (case_id, modality) and report duplicates.

    The audit is read-only, so duplicates are reported as violations rather
    than deleted; downstream consumers must treat ``one_record_per_case_modality``
    == False as a hard data-provenance finding.
    '''
    kept: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    duplicates: List[Dict[str, str]] = []
    for record in records:
        key = case_modality_key(record)
        if key in kept:
            duplicates.append({'case_id': key[0], 'modality': key[1]})
            continue
        kept[key] = record
    return {
        'records': list(kept.values()),
        'duplicates': duplicates,
        'unique_keys': len(kept),
        'violations': len(duplicates),
        'one_record_per_case_modality': len(duplicates) == 0,
    }


def verify_cache_hashes(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    '''Recompute the cache hash of every record and report pass rate/mismatches.'''
    checked = 0
    mismatched: List[Dict[str, Any]] = []
    for record in records:
        checked += 1
        expected = record.get('cache_hash')
        actual = canonical_hash(record.get('cache_payload', {}))
        if expected != actual:
            mismatched.append(
                {
                    'case_id': str(record.get('case_id')),
                    'modality': str(record.get('modality')),
                    'expected_hash': expected,
                    'recomputed_hash': actual,
                }
            )
    passed = checked - len(mismatched)
    return {
        'checked': checked,
        'passed': passed,
        'mismatch_count': len(mismatched),
        'pass_rate': _ratio(passed, checked),
        'mismatched': mismatched,
    }


def index_slices_for_record(case_record: Mapping[str, Any]) -> Dict[str, Any]:
    '''Build a slice_index -> slice_record index and flag duplicate indices.'''
    index: Dict[int, Mapping[str, Any]] = {}
    duplicate_indices: List[int] = []
    total_slices = 0
    for slice_record in case_record.get('slices', []) or []:
        total_slices += 1
        idx = int(slice_record.get('slice_index', -1))
        if idx in index:
            duplicate_indices.append(idx)
            continue
        index[idx] = slice_record
    return {
        'index': index,
        'duplicate_slice_indices': sorted(set(duplicate_indices)),
        'n_slices': len(index),
        'addressable': len(duplicate_indices) == 0 and len(index) == total_slices,
    }


def _max_contiguous_run(values: Iterable[Any]) -> int:
    '''Longest run of consecutive integers in ``values`` (0 for empty input).'''
    ordered = sorted({int(value) for value in values})
    if not ordered:
        return 0
    best = 1
    run = 1
    for previous, current in zip(ordered, ordered[1:]):
        run = run + 1 if current == previous + 1 else 1
        best = max(best, run)
    return best


def hole_is_persistent(hole: Mapping[str, Any]) -> bool:
    '''True when the hole survives across adjacent frames or slices.'''
    explicit = hole.get('persistent')
    if isinstance(explicit, bool):
        return explicit
    frames = hole.get('frames') or []
    slices = hole.get('slices') or hole.get('slice_span') or []
    span = max(_max_contiguous_run(frames), _max_contiguous_run(slices))
    return span >= PERSISTENCE_MIN_SPAN


def classify_temporal_holes(holes: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    '''Count persistent (multi-frame/slice) vs isolated single-frame holes.'''
    persistent = 0
    isolated = 0
    for hole in holes or []:
        if hole_is_persistent(hole):
            persistent += 1
        else:
            isolated += 1
    total = persistent + isolated
    return {
        'total': total,
        'persistent': persistent,
        'isolated': isolated,
        'persistent_ratio': _ratio(persistent, total),
        'isolated_ratio': _ratio(isolated, total),
    }


def audit_case_temporal_holes(case_record: Mapping[str, Any]) -> Dict[str, Any]:
    '''Per-slice and aggregate temporal-hole statistics for one case record.'''
    per_slice: Dict[str, Dict[str, Any]] = {}
    aggregate = {'total': 0, 'persistent': 0, 'isolated': 0}
    for slice_record in case_record.get('slices', []) or []:
        stats = classify_temporal_holes(slice_record.get('holes') or [])
        key = str(int(slice_record.get('slice_index', -1)))
        per_slice[key] = stats
        for field in ('total', 'persistent', 'isolated'):
            aggregate[field] += stats[field]
    aggregate['persistent_ratio'] = _ratio(aggregate['persistent'], aggregate['total'])
    aggregate['isolated_ratio'] = _ratio(aggregate['isolated'], aggregate['total'])
    return {'per_slice': per_slice, 'aggregate': aggregate}


def cooccurrence_rates(
    slices: Iterable[Mapping[str, Any]],
    hole_marker: str = HOLE_MARKER,
    markers: Sequence[str] = COOCCURRENCE_MARKERS,
) -> Dict[str, Any]:
    '''Same-slice co-occurrence of the hole marker with the sibling QC markers.'''
    hole_slices: List[Mapping[str, Any]] = []
    for slice_record in slices or []:
        if hole_marker in (slice_record.get('markers') or []):
            hole_slices.append(slice_record)
    denominator = len(hole_slices)
    per_marker: Dict[str, Dict[str, Any]] = {}
    for marker in markers:
        both = sum(1 for slice_record in hole_slices if marker in (slice_record.get('markers') or []))
        per_marker[marker] = {
            'cooccurring_slices': both,
            'hole_slices': denominator,
            'rate': _ratio(both, denominator),
        }
    return {'hole_marker': hole_marker, 'hole_marker_slices': denominator, 'per_marker': per_marker}


def extract_via_run9_b1_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    '''Unique run-9 'B1 contract' import-style extractor (mismatch fallback only).

    Pure declarative key remap of a run-9 B1 payload onto the canonical audit
    shape. This is the single permitted recovery path; it never recomputes,
    rewrites or re-labels anything.
    '''
    if not isinstance(payload, Mapping):
        raise ValueError('run-9 B1 payload must be a mapping')
    if payload.get('contract_version') != RUN9_B1_CONTRACT_VERSION:
        raise ValueError('payload is not a run-9 B1 contract payload')
    run_id = payload.get('run_id', RUN9_B1_RUN_ID)
    if run_id != RUN9_B1_RUN_ID:
        raise ValueError('run-9 B1 extractor only accepts run-9 payloads')
    slices: List[Dict[str, Any]] = []
    for raw_slice in payload.get('b1_slices', []) or []:
        holes: List[Dict[str, Any]] = []
        for raw_hole in raw_slice.get('b1_holes', []) or []:
            holes.append(
                {
                    'hole_id': raw_hole.get('b1_hole_id'),
                    'frames': list(raw_hole.get('b1_frames', []) or []),
                    'slices': list(raw_hole.get('b1_slice_span', []) or []),
                    'area_mm2': raw_hole.get('b1_area_mm2'),
                }
            )
        slices.append(
            {
                'slice_index': int(raw_slice['b1_slice_index']),
                'markers': list(raw_slice.get('b1_markers', []) or []),
                'holes': holes,
            }
        )
    return {
        'case_id': str(payload.get('case_id')),
        'modality': str(payload.get('modality')),
        'run_id': RUN9_B1_RUN_ID,
        'cache_hash': None,
        'cache_payload': None,
        'slices': slices,
        'extracted_by': 'run9_b1_contract',
    }


def resolve_hash_mismatches(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    '''Route hash-mismatched records exclusively through the run-9 B1 extractor.'''
    resolved: List[Mapping[str, Any]] = []
    unrecoverable: List[Dict[str, str]] = []
    for record in records:
        payload = record.get('run9_b1_payload') or record.get('cache_payload')
        try:
            resolved.append(extract_via_run9_b1_contract(payload))
        except (ValueError, TypeError, KeyError) as exc:
            unrecoverable.append(
                {
                    'case_id': str(record.get('case_id')),
                    'modality': str(record.get('modality')),
                    'reason': str(exc),
                }
            )
    return {
        'extractor': 'run9_b1_contract',
        'resolved': resolved,
        'unrecoverable': unrecoverable,
        'resolved_count': len(resolved),
        'unrecoverable_count': len(unrecoverable),
        'second_toolset_used': False,
    }


def run_shadow_audit(
    records: Iterable[Mapping[str, Any]],
    *,
    cohort: Optional[Mapping[str, int]] = None,
    allowed_runs: Sequence[str] = REUSED_CACHE_RUNS,
    max_positive: int = MAX_POSITIVE_CASES,
    max_control: int = MAX_CONTROL_CASES,
) -> Dict[str, Any]:
    '''Run the bounded read-only provenance shadow audit and return a report.

    Raises ValueError when the declared cohort exceeds the approved bound and
    RuntimeError if the read-only contract is violated on the input records.
    '''
    cohort = dict(cohort or {})
    positive = int(cohort.get('positive', 0))
    control = int(cohort.get('control', 0))
    if positive > max_positive or control > max_control:
        raise ValueError(
            'bounded audit exceeded: expected <= %d positive and <= %d control cases, got %d/%d'
            % (max_positive, max_control, positive, control)
        )

    records = list(records)
    snapshot = copy.deepcopy(records)

    filtered = filter_reused_runs(records, allowed_runs)
    dedup = enforce_single_record_per_case_modality(filtered['kept'])
    integrity = verify_cache_hashes(dedup['records'])

    mismatched_keys = {(m['case_id'], m['modality']) for m in integrity['mismatched']}
    mismatched_records = [r for r in dedup['records'] if case_modality_key(r) in mismatched_keys]
    if mismatched_records:
        fallback = resolve_hash_mismatches(mismatched_records)
    else:
        fallback = {
            'extractor': None,
            'resolved': [],
            'unrecoverable': [],
            'resolved_count': 0,
            'unrecoverable_count': 0,
            'second_toolset_used': False,
        }

    effective: List[Mapping[str, Any]] = [
        r for r in dedup['records'] if case_modality_key(r) not in mismatched_keys
    ]
    effective.extend(fallback['resolved'])

    slice_index_summary: Dict[str, Dict[str, Any]] = {}
    per_case_holes: Dict[str, Dict[str, Any]] = {}
    per_case_cooccurrence: Dict[str, Dict[str, Any]] = {}
    all_slices: List[Mapping[str, Any]] = []
    aggregate_holes = {'total': 0, 'persistent': 0, 'isolated': 0}

    for record in effective:
        case_id, modality = case_modality_key(record)
        label = '%s|%s' % (case_id, modality)
        indexed = index_slices_for_record(record)
        slice_index_summary[label] = {
            'case_id': case_id,
            'modality': modality,
            'n_slices': indexed['n_slices'],
            'duplicate_slice_indices': indexed['duplicate_slice_indices'],
            'addressable': indexed['addressable'],
        }
        case_slices = list(indexed['index'].values())
        all_slices.extend(case_slices)
        holes = audit_case_temporal_holes(record)
        per_case_holes[label] = holes
        for field in ('total', 'persistent', 'isolated'):
            aggregate_holes[field] += holes['aggregate'][field]
        per_case_cooccurrence[label] = cooccurrence_rates(case_slices)

    aggregate_holes['persistent_ratio'] = _ratio(aggregate_holes['persistent'], aggregate_holes['total'])
    aggregate_holes['isolated_ratio'] = _ratio(aggregate_holes['isolated'], aggregate_holes['total'])

    if records != snapshot:
        raise RuntimeError('read-only contract violated: shadow audit mutated its input')

    return {
        'audit_mode': AUDIT_MODE,
        'marker': HOLE_MARKER,
        'cohort': {
            'positive': positive,
            'control': control,
            'bounded_positive_max': max_positive,
            'bounded_control_max': max_control,
        },
        'runs_reused': sorted(allowed_runs),
        'cache_filter': {
            'kept': len(filtered['kept']),
            'skipped_count': filtered['skipped_count'],
            'skipped_runs': filtered['skipped_runs'],
            'skipped_case_ids': filtered['skipped_case_ids'],
        },
        'key_uniqueness': {
            'unique_keys': dedup['unique_keys'],
            'violations': dedup['violations'],
            'one_record_per_case_modality': dedup['one_record_per_case_modality'],
        },
        'cache_integrity': {
            'checked': integrity['checked'],
            'passed': integrity['passed'],
            'mismatch_count': integrity['mismatch_count'],
            'pass_rate': integrity['pass_rate'],
        },
        'cache_mismatches': integrity['mismatched'],
        'run9_b1_fallback': {
            'extractor': fallback['extractor'],
            'resolved_count': fallback['resolved_count'],
            'unrecoverable_count': fallback['unrecoverable_count'],
            'unrecoverable': fallback['unrecoverable'],
            'second_toolset_used': fallback['second_toolset_used'],
        },
        'slice_index': slice_index_summary,
        'temporal_holes': {'aggregate': aggregate_holes, 'per_case': per_case_holes},
        'cooccurrence': {
            'overall': cooccurrence_rates(all_slices),
            'per_case': per_case_cooccurrence,
        },
        'guardrails': {
            'read_only': True,
            'labels_modified': False,
            'production_rules_modified': False,
            'datasets_touched': False,
            'second_toolset_used': False,
        },
    }
