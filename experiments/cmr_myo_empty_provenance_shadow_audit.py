"""Bounded read-only main-provenance shadow audit for ``cmr_myo_empty``.

The stratum ``cmr_myo_empty`` (n=106) has no dedicated main route, so this
module only *observes* provenance that was already verified (cache runs
4/5/9/13-18) and reports metrics.  It never rewrites labels, never edits
production rules and never touches datasets, patient images or runtime
databases.

Contract
--------
* READ-ONLY: ``run_shadow_audit`` snapshots its input and asserts the snapshot
  is unchanged before it returns.
* At most ONE extractor may ever be created: the verified run-9 ``B1`` import
  extractor (``ALLOWED_EXTRACTOR``).  Any second toolset raises
  ``PermissionError``, and a repair is requested at most once per audit no
  matter how many cache digests mismatch.
* A cache digest that matches the verified summary must create zero extractors.

Acceptance metrics
------------------
1. Dedup pass rate == 1.0: one merged record per (case_id, modality); issue
   sets are unioned and every contributing slice index is kept per issue.
2. Cache-hash reuse rate (``reuse_rate``: 1.0 == cache accepted wholesale,
   0.0 == reuse invalidated by at least one mismatch) plus mismatch counters
   at run / record / case granularity; ``new_extractor_count == 0`` whenever
   every digest is valid.
3. Empty-slice distribution over basal / mid / apical, empty<->non-empty
   temporal continuity and same-slice co-occurrence with ``small_area``,
   ``missing_temporal_hole``, ``touches_image_border`` and ``centroid_drift``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    'STRATUM_NAME',
    'STRATUM_TOTAL',
    'VERIFIED_RUN_IDS',
    'ALLOWED_EXTRACTOR',
    'EMPTY_ISSUE',
    'SEGMENTS',
    'CO_OCCURRENCE_ISSUES',
    'POSITIVE_COUNT',
    'CONTROL_COUNT',
    'SingleImportExtractorFactory',
    'classify_slice',
    'cooccurrence_summary',
    'dedupe_records',
    'dedup_pass_rate',
    'empty_slice_distribution',
    'run_cache_digest',
    'same_slice_cooccurrence',
    'select_cohort',
    'slice_counts_from_records',
    'temporal_continuity',
    'temporal_continuity_summary',
    'verify_cache_digests',
    'run_shadow_audit',
]

STRATUM_NAME = 'cmr_myo_empty'
STRATUM_TOTAL = 106

# Provenance caches already verified; may be reused as digests.
VERIFIED_RUN_IDS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)

# The single run-9 B1 import extractor allowed for hash-mismatch repair.
ALLOWED_EXTRACTOR = 'run9_b1_import'

EMPTY_ISSUE = 'empty'
SEGMENTS: Tuple[str, ...] = ('basal', 'mid', 'apical')
CO_OCCURRENCE_ISSUES: Tuple[str, ...] = (
    'small_area',
    'missing_temporal_hole',
    'touches_image_border',
    'centroid_drift',
)
POSITIVE_COUNT = 30
CONTROL_COUNT = 10


class SingleImportExtractorFactory:
    """Lazily creates at most one extractor: the verified run-9 B1 importer."""

    def __init__(self) -> None:
        self._instance: Optional[Dict[str, Any]] = None
        self.new_extractor_count = 0
        self.requested_kinds: List[str] = []

    def get(self, kind: str = ALLOWED_EXTRACTOR) -> Dict[str, Any]:
        self.requested_kinds.append(kind)
        if kind != ALLOWED_EXTRACTOR:
            raise PermissionError(
                'second toolset is forbidden by the audit contract: '
                '{0!r} != {1!r}'.format(kind, ALLOWED_EXTRACTOR)
            )
        if self._instance is None:
            self._instance = {
                'kind': kind,
                'source_run': 9,
                'family': 'B1',
                'mode': 'import',
            }
            self.new_extractor_count += 1
        return self._instance


def _snapshot(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """Read-only guard: immutable, comparable view of incoming records."""
    return tuple(
        tuple(sorted((key, repr(value)) for key, value in record.items()))
        for record in records
    )


def select_cohort(
    records: Sequence[Mapping[str, Any]],
    n_positive: int = POSITIVE_COUNT,
    n_control: int = CONTROL_COUNT,
) -> Dict[str, Any]:
    """Deterministic 30 positive + 10 control slice of the empty stratum."""
    if n_positive + n_control > STRATUM_TOTAL:
        raise ValueError(
            'cohort {0}+{1} exceeds stratum total {2}'.format(
                n_positive, n_control, STRATUM_TOTAL
            )
        )
    empty_cases = set()
    all_cases = set()
    for record in records:
        case_id = record.get('case_id')
        all_cases.add(case_id)
        if record.get('issue') == EMPTY_ISSUE:
            empty_cases.add(case_id)
    positives = sorted(empty_cases)
    controls = sorted(all_cases - empty_cases)
    if len(positives) < n_positive:
        raise ValueError(
            'need >= {0} cases with {1!r}, found {2}'.format(
                n_positive, EMPTY_ISSUE, len(positives)
            )
        )
    if len(controls) < n_control:
        raise ValueError(
            'need >= {0} control cases, found {1}'.format(n_control, len(controls))
        )
    return {
        'positives': positives[:n_positive],
        'controls': controls[:n_control],
        'cohort_size': n_positive + n_control,
    }


def dedupe_records(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse to one record per (case_id, modality), unioning issue slices."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (record['case_id'], record['modality'])
        entry = merged.get(key)
        if entry is None:
            entry = {
                'case_id': record['case_id'],
                'modality': record['modality'],
                'issues': {},
                'runs': set(),
            }
            merged[key] = entry
        issue = record['issue']
        bucket = entry['issues'].setdefault(issue, [])
        slice_index = record.get('slice_index')
        if slice_index is not None and slice_index not in bucket:
            bucket.append(slice_index)
        run_id = record.get('run_id')
        if run_id is not None:
            entry['runs'].add(run_id)
    out: List[Dict[str, Any]] = []
    for key in sorted(merged):
        entry = merged[key]
        out.append(
            {
                'case_id': entry['case_id'],
                'modality': entry['modality'],
                'issues': {
                    issue: sorted(indices)
                    for issue, indices in sorted(entry['issues'].items())
                },
                'runs': sorted(entry['runs']),
            }
        )
    return out


def dedup_pass_rate(
    records: Sequence[Mapping[str, Any]],
    merged_records: Sequence[Mapping[str, Any]],
) -> float:
    """1.0 iff exactly one output record per distinct input (case_id, modality)."""
    keys = {(record['case_id'], record['modality']) for record in records}
    if not keys:
        return 1.0
    out_keys = [(entry['case_id'], entry['modality']) for entry in merged_records]
    if len(out_keys) != len(set(out_keys)):
        return 0.0
    if set(out_keys) != keys:
        return 0.0
    return 1.0


def run_cache_digest(
    run_records: Sequence[Mapping[str, Any]],
    digester: Callable[[Mapping[str, Any]], str],
) -> str:
    """Deterministic digest of one verified run's cached records.

    Per-record digests are folded in sorted order, so the run digest does not
    depend on the replay order of the cache summary.
    """
    return '||'.join(sorted(digester(record) for record in run_records))


def _group_records_by_run(
    records: Sequence[Mapping[str, Any]],
) -> Dict[int, List[Mapping[str, Any]]]:
    """Group cache records by verified run id (unverified runs are ignored)."""
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for record in records:
        run_id = record.get('run_id')
        if run_id in VERIFIED_RUN_IDS:
            grouped.setdefault(run_id, []).append(record)
    return grouped


def verify_cache_digests(
    records: Sequence[Mapping[str, Any]],
    digester: Callable[[Mapping[str, Any]], str],
    verified_digests: Mapping[int, str],
    extractor_factory: SingleImportExtractorFactory,
) -> Dict[str, Any]:
    """Reuse the verified run-4/5/9/13-18 cache digests.

    Each verified run present in ``records`` is digested exactly once with
    :func:`run_cache_digest`.  A run whose digest equals the stored summary is
    reused; any mismatch invalidates the wholesale reuse of the cache
    (``reuse_rate == 0.0``) and triggers exactly ONE repair attempt with the
    single allowed run-9 ``B1`` import extractor, no matter how many runs
    mismatch.  When every digest is valid no extractor is built and no
    extraction is requested.
    """
    grouped = _group_records_by_run(records)
    reused_runs = 0
    reused_records = 0
    checked_records = 0
    mismatches: List[Dict[str, Any]] = []
    for run_id in sorted(grouped):
        run_records = grouped[run_id]
        checked_records += len(run_records)
        observed = run_cache_digest(run_records, digester)
        expected = verified_digests.get(run_id)
        if expected is not None and observed == expected:
            reused_runs += 1
            reused_records += len(run_records)
            continue
        mismatches.append(
            {
                'run_id': run_id,
                'expected': expected,
                'observed': observed,
                'record_count': len(run_records),
                'case_ids': sorted(
                    {str(record.get('case_id')) for record in run_records}
                ),
            }
        )
    if mismatches:
        # Exactly one bounded repair pass with the single verified extractor;
        # the factory itself refuses any second toolset.
        extractor_factory.get(ALLOWED_EXTRACTOR)

    checked_runs = len(grouped)
    mismatch_record_count = sum(item['record_count'] for item in mismatches)
    mismatch_case_count = len(
        {
            (record.get('case_id'), record.get('modality'))
            for item in mismatches
            for record in grouped[item['run_id']]
        }
    )
    return {
        'verified_runs': list(VERIFIED_RUN_IDS),
        'checked_runs': checked_runs,
        'reused_runs': reused_runs,
        'run_reuse_rate': (
            reused_runs / checked_runs if checked_runs else 1.0
        ),
        'checked_records': checked_records,
        'reused_records': reused_records,
        'record_reuse_rate': (
            reused_records / checked_records if checked_records else 1.0
        ),
        # Wholesale cache-reuse acceptance: 1.0 == every digest valid;
        # 0.0 == reuse invalidated (a repair extractor was required).
        'reuse_rate': 0.0 if mismatches else 1.0,
        'reuse_accepted': not mismatches,
        'mismatch_count': len(mismatches),
        'mismatch_record_count': mismatch_record_count,
        'mismatch_case_count': mismatch_case_count,
        'mismatches': mismatches,
        'new_extractor_count': extractor_factory.new_extractor_count,
    }


def slice_counts_from_records(
    records: Sequence[Mapping[str, Any]], default_n_slices: int
) -> Dict[Tuple[str, str], int]:
    """Slice-count map keyed by (case_id, modality); n_slices overrides default."""
    counts: Dict[Tuple[str, str], int] = {}
    for record in records:
        key = (record['case_id'], record['modality'])
        n_slices = int(record.get('n_slices', default_n_slices))
        previous = counts.get(key)
        counts[key] = n_slices if previous is None else max(previous, n_slices)
    return counts


def classify_slice(slice_index: int, n_slices: int) -> str:
    """Basal / mid / apical segment of a slice index in a short-axis stack."""
    if n_slices <= 0:
        raise ValueError('n_slices must be positive, got {0}'.format(n_slices))
    if not 0 <= slice_index < n_slices:
        raise ValueError(
            'slice_index {0} out of range for {1} slices'.format(
                slice_index, n_slices
            )
        )
    third = n_slices / 3.0
    if slice_index < third:
        return 'basal'
    if slice_index < 2.0 * third:
        return 'mid'
    return 'apical'


def empty_slice_distribution(
    merged_records: Sequence[Mapping[str, Any]],
    slice_counts: Mapping[Tuple[str, str], int],
    empty_issue: str = EMPTY_ISSUE,
) -> Dict[str, Any]:
    """Distribution of empty slices over the basal / mid / apical segments."""
    counts = {segment: 0 for segment in SEGMENTS}
    per_case: List[Dict[str, Any]] = []
    for entry in merged_records:
        key = (entry['case_id'], entry['modality'])
        n_slices = slice_counts[key]
        indices = sorted(entry['issues'].get(empty_issue, ()))
        if not indices:
            continue
        segments = {segment: 0 for segment in SEGMENTS}
        for index in indices:
            segment = classify_slice(index, n_slices)
            counts[segment] += 1
            segments[segment] += 1
        per_case.append(
            {
                'case_id': entry['case_id'],
                'modality': entry['modality'],
                'n_slices': n_slices,
                'empty_slices': indices,
                'segments': segments,
            }
        )
    total = sum(counts.values())
    shares = {
        segment: (counts[segment] / total) if total else 0.0
        for segment in SEGMENTS
    }
    return {
        'counts': counts,
        'shares': shares,
        'total_empty_slices': total,
        'per_case': per_case,
    }


def temporal_continuity(
    merged_record: Mapping[str, Any],
    n_slices: int,
    empty_issue: str = EMPTY_ISSUE,
) -> Dict[str, Any]:
    """Empty <-> non-empty transitions along the slice axis of one record."""
    empty = set(merged_record['issues'].get(empty_issue, ()))
    flags = [1 if index in empty else 0 for index in range(n_slices)]
    if n_slices <= 1:
        transitions = 0
        continuity = 1.0
    else:
        transitions = sum(1 for a, b in zip(flags, flags[1:]) if a != b)
        continuity = 1.0 - transitions / (n_slices - 1)
    return {
        'case_id': merged_record['case_id'],
        'modality': merged_record['modality'],
        'n_slices': n_slices,
        'empty_flags': flags,
        'transitions': transitions,
        'continuity': continuity,
    }


def temporal_continuity_summary(
    merged_records: Sequence[Mapping[str, Any]],
    slice_counts: Mapping[Tuple[str, str], int],
    empty_issue: str = EMPTY_ISSUE,
) -> Dict[str, Any]:
    """Per-record and aggregate empty/non-empty temporal continuity."""
    per_record = [
        temporal_continuity(
            entry,
            slice_counts[(entry['case_id'], entry['modality'])],
            empty_issue,
        )
        for entry in merged_records
    ]
    with_empty = [item for item in per_record if any(item['empty_flags'])]
    total_transitions = sum(item['transitions'] for item in with_empty)
    mean_continuity = (
        sum(item['continuity'] for item in with_empty) / len(with_empty)
        if with_empty
        else 1.0
    )
    max_transitions = sum(
        max(item['n_slices'] - 1, 0) for item in with_empty
    )
    return {
        'records': len(per_record),
        'records_with_empty': len(with_empty),
        'total_transitions': total_transitions,
        'mean_transitions': (
            total_transitions / len(with_empty) if with_empty else 0.0
        ),
        'mean_continuity': mean_continuity,
        'is_continuous': total_transitions <= 2 * len(with_empty),
        'max_transitions': max_transitions,
        'per_record': per_record,
    }


def same_slice_cooccurrence(
    merged_records: Sequence[Mapping[str, Any]],
    issues: Sequence[str] = CO_OCCURRENCE_ISSUES,
    empty_issue: str = EMPTY_ISSUE,
) -> Dict[str, Any]:
    """Same-slice co-occurrence of the empty issue with each listed issue."""
    stats: Dict[str, Dict[str, Any]] = {}
    for issue in issues:
        stats[issue] = {
            'empty_slices': 0,
            'cooccurring_slices': 0,
            'rate': 0.0,
            'cases': [],
        }
    for entry in merged_records:
        empty_slices = set(entry['issues'].get(empty_issue, ()))
        if not empty_slices:
            continue
        for issue in issues:
            other = set(entry['issues'].get(issue, ()))
            if not other:
                continue
            shared = sorted(empty_slices & other)
            bucket = stats[issue]
            bucket['empty_slices'] += len(empty_slices)
            bucket['cooccurring_slices'] += len(shared)
            bucket['cases'].append(
                {
                    'case_id': entry['case_id'],
                    'modality': entry['modality'],
                    'shared_slices': shared,
                }
            )
    for issue in issues:
        bucket = stats[issue]
        denominator = bucket['empty_slices']
        bucket['rate'] = (
            bucket['cooccurring_slices'] / denominator if denominator else 0.0
        )
    return stats


def cooccurrence_summary(
    merged_records: Sequence[Mapping[str, Any]],
    issues: Sequence[str] = CO_OCCURRENCE_ISSUES,
    empty_issue: str = EMPTY_ISSUE,
) -> Dict[str, float]:
    """Flat {issue: same-slice rate} view for compact reporting."""
    stats = same_slice_cooccurrence(merged_records, issues, empty_issue)
    return {issue: stats[issue]['rate'] for issue in issues}


def _assert_read_only(
    before: Tuple[Tuple[Tuple[str, str], ...], ...],
    records: Sequence[Mapping[str, Any]],
) -> None:
    after = _snapshot(records)
    if before != after:
        raise RuntimeError('shadow audit mutated its read-only input records')


def run_shadow_audit(
    records: Sequence[Mapping[str, Any]],
    digester: Callable[[Mapping[str, Any]], str],
    verified_digests: Mapping[int, str],
    default_n_slices: int = 12,
    n_positive: int = POSITIVE_COUNT,
    n_control: int = CONTROL_COUNT,
    extractor_factory: Optional[SingleImportExtractorFactory] = None,
) -> Dict[str, Any]:
    """Read-only bounded audit over the cmr_myo_empty provenance records.

    ``records`` are the cached run-4/5/9/13-18 summaries: each is a mapping
    with ``case_id``, ``modality``, ``issue``, ``slice_index`` (optional),
    ``run_id`` (optional) and ``n_slices`` (optional).  Nothing is written.
    """
    before = _snapshot(records)
    if extractor_factory is None:
        extractor_factory = SingleImportExtractorFactory()

    cohort = select_cohort(records, n_positive, n_control)
    merged = dedupe_records(records)
    pass_rate = dedup_pass_rate(records, merged)
    if pass_rate != 1.0:
        raise RuntimeError('dedup pass rate below 1.0: {0}'.format(pass_rate))

    hash_report = verify_cache_digests(
        records, digester, verified_digests, extractor_factory
    )
    if hash_report['mismatch_count'] == 0 and hash_report['new_extractor_count']:
        raise RuntimeError('valid digests must not create a new extractor')

    slice_counts = slice_counts_from_records(records, default_n_slices)
    empty_distribution = empty_slice_distribution(merged, slice_counts)
    continuity = temporal_continuity_summary(merged, slice_counts)
    cooccurrence = same_slice_cooccurrence(merged)

    _assert_read_only(before, records)

    return {
        'stratum': STRATUM_NAME,
        'stratum_total': STRATUM_TOTAL,
        'cohort': cohort,
        'dedup': {
            'input_records': len(records),
            'merged_records': len(merged),
            'dedup_pass_rate': pass_rate,
            'records': merged,
        },
        'hash_reuse': hash_report,
        'empty_slice_distribution': empty_distribution,
        'temporal_continuity': continuity,
        'cooccurrence': cooccurrence,
        'extractor': {
            'allowed': ALLOWED_EXTRACTOR,
            'new_extractor_count': extractor_factory.new_extractor_count,
            'requested_kinds': list(extractor_factory.requested_kinds),
        },
        'read_only': True,
    }
