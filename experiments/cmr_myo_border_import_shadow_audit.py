'''Bounded read-only import shadow audit for ``cmr_myo_touches_image_border``.

Approved action: audit the myocardium-touches-image-border flag (population
n=931) with a bounded cohort of 50 positives plus 10 controls, reusing the
cached summaries from runs 4/5/9/13-18.  Cache-hash mismatches may only be
re-resolved through the run-9 B1 extractor, and every such call is reported.
Labels, thresholds and production routing rules are never modified.

Pure and offline: callers inject the cached summaries (``CachedCaseSummary``
instances or plain mappings), so this module never touches files, datasets,
patient images, databases or the network.  Records whose ``declared_hash``
differs from their ``payload_hash`` are treated as unverified: they are excluded
from the co-occurrence and slice-index distributions and surfaced through the
run-9 B1 resolution channel.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    'CACHE_RUN_IDS',
    'B1_EXTRACTOR_NAME',
    'B1_EXTRACTOR_RUN_ID',
    'CACHE_EXTRACTOR_NAME',
    'PERMITTED_EXTRACTORS',
    'CO_FLAGS',
    'DEFAULT_POSITIVE_BUDGET',
    'DEFAULT_CONTROL_BUDGET',
    'MAX_COHORT_SIZE',
    'BORDER_FLAG_POPULATION_N',
    'CachedCaseSummary',
    'coerce_record',
    'select_cohort',
    'cooccurrence_summary',
    'default_worst_slice_index',
    'slice_index_profile',
    'single_record_compliance',
    'integrity_summary',
    'build_shadow_audit_report',
]

CACHE_RUN_IDS: Tuple[str, ...] = (
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
B1_EXTRACTOR_RUN_ID: str = 'run-9'
B1_EXTRACTOR_NAME: str = 'run-9-B1'
CACHE_EXTRACTOR_NAME: str = 'cache'
PERMITTED_EXTRACTORS: Tuple[str, ...] = (CACHE_EXTRACTOR_NAME, B1_EXTRACTOR_NAME)
DEFAULT_POSITIVE_BUDGET: int = 50
DEFAULT_CONTROL_BUDGET: int = 10
MAX_COHORT_SIZE: int = DEFAULT_POSITIVE_BUDGET + DEFAULT_CONTROL_BUDGET
BORDER_FLAG_POPULATION_N: int = 931
CO_FLAGS: Tuple[str, ...] = ('small_area', 'excess_components', 'missing_temporal_hole')
_REQUIRED_FIELDS: Tuple[str, ...] = (
    'case_id',
    'modality',
    'run_id',
    'declared_hash',
    'payload_hash',
    'touches_image_border',
)


@dataclass(frozen=True)
class CachedCaseSummary:
    '''One cached (case, modality) summary reused read-only by the audit.'''

    case_id: str
    modality: str
    run_id: str
    declared_hash: str
    payload_hash: str
    touches_image_border: bool
    small_area: bool = False
    excess_components: bool = False
    missing_temporal_hole: bool = False
    marked_slice_indices: Tuple[int, ...] = ()
    extractor: str = CACHE_EXTRACTOR_NAME

    @property
    def key(self) -> Tuple[str, str]:
        return (self.case_id, self.modality)

    @property
    def hash_matches(self) -> bool:
        return self.declared_hash == self.payload_hash


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _run_order(run_id: str) -> Tuple[int, str]:
    digits = ''.join(character for character in str(run_id) if character.isdigit())
    return (int(digits) if digits else -1, str(run_id))


def coerce_record(raw: Any) -> CachedCaseSummary:
    '''Normalise one cached summary without mutating the caller's object.'''
    if isinstance(raw, CachedCaseSummary):
        record = raw
    elif isinstance(raw, Mapping):
        missing = [name for name in _REQUIRED_FIELDS if name not in raw]
        if missing:
            raise ValueError(f'cache summary is missing required field(s): {missing}')
        record = CachedCaseSummary(
            case_id=str(raw['case_id']),
            modality=str(raw['modality']),
            run_id=str(raw['run_id']),
            declared_hash=str(raw['declared_hash']),
            payload_hash=str(raw['payload_hash']),
            touches_image_border=bool(raw['touches_image_border']),
            small_area=bool(raw.get('small_area', False)),
            excess_components=bool(raw.get('excess_components', False)),
            missing_temporal_hole=bool(raw.get('missing_temporal_hole', False)),
            marked_slice_indices=tuple(
                int(index) for index in (raw.get('marked_slice_indices') or ())
            ),
            extractor=str(raw.get('extractor', CACHE_EXTRACTOR_NAME)),
        )
    else:
        raise TypeError(f'unsupported cache summary type: {type(raw)!r}')

    indices = tuple(int(index) for index in record.marked_slice_indices)
    if any(index < 0 for index in indices):
        raise ValueError(
            f'cache summary for {record.case_id}/{record.modality} carries a negative marked slice index'
        )
    if indices != tuple(record.marked_slice_indices):
        record = replace(record, marked_slice_indices=indices)
    return record


def select_cohort(
    records: Sequence[CachedCaseSummary],
    *,
    positive_budget: int = DEFAULT_POSITIVE_BUDGET,
    control_budget: int = DEFAULT_CONTROL_BUDGET,
) -> Tuple[List[CachedCaseSummary], List[CachedCaseSummary]]:
    '''Pick at most ``positive_budget`` border cases plus ``control_budget`` controls.

    Ordering is ``(case_id, modality, run_number)`` so selection is deterministic;
    the first record for each ``(case_id, modality)`` wins and later copies never
    inflate the cohort (they are reported as compliance violations elsewhere).
    Read-only: nothing is written back.
    '''
    ordered = sorted(
        records,
        key=lambda item: (item.case_id, item.modality, _run_order(item.run_id)),
    )
    seen: set = set()
    deduped: List[CachedCaseSummary] = []
    for record in ordered:
        if record.key in seen:
            continue
        seen.add(record.key)
        deduped.append(record)

    positives = [
        record for record in deduped if record.touches_image_border
    ][: max(int(positive_budget), 0)]
    controls = [
        record for record in deduped if not record.touches_image_border
    ][: max(int(control_budget), 0)]
    return positives, controls


def cooccurrence_summary(
    group: Sequence[CachedCaseSummary],
    analyzed_cohort_size: int,
) -> Dict[str, Any]:
    '''Co-flag counts and shares of one cohort group against the other QC flags.

    Returns ``group_size`` and ``analyzed_cohort_size`` at the top level, the
    per-flag counts/shares under ``flags``, and the per-case co-flag tallies
    under ``summary``.
    '''
    size = len(group)
    flags: Dict[str, Dict[str, Any]] = {}
    for flag in CO_FLAGS:
        count = sum(1 for record in group if bool(getattr(record, flag)))
        flags[flag] = {
            'count': count,
            'share_of_group': _ratio(count, size),
            'share_of_analyzed_cohort': _ratio(count, int(analyzed_cohort_size)),
        }
    per_case_flag_counts = [
        sum(1 for flag in CO_FLAGS if bool(getattr(record, flag))) for record in group
    ]
    summary = {
        'cases_with_no_co_flag': sum(1 for value in per_case_flag_counts if value == 0),
        'cases_with_multiple_co_flags': sum(1 for value in per_case_flag_counts if value >= 2),
        'cases_with_all_three_co_flags': sum(
            1 for value in per_case_flag_counts if value == len(CO_FLAGS)
        ),
    }
    return {
        'group_size': size,
        'analyzed_cohort_size': int(analyzed_cohort_size),
        'flags': flags,
        'summary': summary,
    }


def default_worst_slice_index(indices: Sequence[int]) -> Optional[int]:
    '''``None`` when no slice is marked, otherwise the highest marked index.'''
    return max(indices) if indices else None


def slice_index_profile(
    records: Sequence[CachedCaseSummary],
    *,
    worst_slice_selector: Optional[Callable[[Sequence[int]], Optional[int]]] = None,
) -> Dict[str, Any]:
    '''Per-case marked slice indices, index histograms and the worst slice per case.

    ``worst_slice_selector`` defaults to the highest marked index; callers may
    inject another severity ordering without changing the read-only contract.
    '''
    selector = worst_slice_selector or default_worst_slice_index
    per_case: List[Dict[str, Any]] = []
    histogram: Dict[int, int] = {}
    worst_histogram: Dict[int, int] = {}
    without_marked: List[Dict[str, str]] = []

    for record in sorted(records, key=lambda item: (item.case_id, item.modality)):
        indices = tuple(sorted({int(index) for index in record.marked_slice_indices}))
        worst_raw = selector(indices)
        worst = None if worst_raw is None else int(worst_raw)
        per_case.append(
            {
                'case_id': record.case_id,
                'modality': record.modality,
                'cached_run_id': record.run_id,
                'group': 'border' if record.touches_image_border else 'control',
                'marked_slice_count': len(indices),
                'marked_slice_indices': list(indices),
                'worst_slice_index': worst,
            }
        )
        for index in indices:
            histogram[index] = histogram.get(index, 0) + 1
        if worst is None:
            without_marked.append(
                {'case_id': record.case_id, 'modality': record.modality}
            )
        else:
            worst_histogram[worst] = worst_histogram.get(worst, 0) + 1

    return {
        'per_case': per_case,
        'marked_slice_index_histogram': dict(sorted(histogram.items())),
        'worst_slice_index_histogram': dict(sorted(worst_histogram.items())),
        'cases_without_marked_slices': without_marked,
        'cases_with_marked_slices': len(per_case) - len(without_marked),
    }


def single_record_compliance(records: Sequence[CachedCaseSummary]) -> Dict[str, Any]:
    '''Exactly one cache record per (case_id, modality) is the required shape.'''
    grouped: Dict[Tuple[str, str], List[CachedCaseSummary]] = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record)

    duplicates = sorted(key for key, items in grouped.items() if len(items) > 1)
    single = sum(1 for items in grouped.values() if len(items) == 1)
    distinct = len(grouped)
    return {
        'distinct_case_modality_keys': distinct,
        'keys_with_single_record': single,
        'keys_with_multiple_records': len(duplicates),
        'duplicate_keys': [
            {'case_id': key[0], 'modality': key[1]} for key in duplicates
        ],
        'single_record_compliance_rate': _ratio(single, distinct),
        'compliant': not duplicates,
    }


def integrity_summary(
    cohort: Sequence[CachedCaseSummary],
    compliance: Mapping[str, Any],
) -> Dict[str, Any]:
    '''Hash-mismatch accounting plus the run-9 B1 extraction channel.'''
    mismatched = [record for record in cohort if not record.hash_matches]
    b1_calls = [
        {
            'case_id': record.case_id,
            'modality': record.modality,
            'cached_run_id': record.run_id,
            'reason': 'cache_hash_mismatch',
            'extractor': B1_EXTRACTOR_NAME,
            'extractor_run': B1_EXTRACTOR_RUN_ID,
        }
        for record in mismatched
    ]

    violations: List[Dict[str, Any]] = []
    for record in cohort:
        if record.extractor in PERMITTED_EXTRACTORS:
            continue
        if record.hash_matches:
            violations.append(
                {
                    'code': 'unrecognized_extractor',
                    'case_id': record.case_id,
                    'modality': record.modality,
                    'observed_extractor': record.extractor,
                    'required_extractor': CACHE_EXTRACTOR_NAME,
                }
            )
        else:
            violations.append(
                {
                    'code': 'hash_mismatch_resolved_by_non_b1_extractor',
                    'case_id': record.case_id,
                    'modality': record.modality,
                    'observed_extractor': record.extractor,
                    'required_extractor': B1_EXTRACTOR_NAME,
                }
            )

    return {
        'cohort_records': len(cohort),
        'verified_records': len(cohort) - len(mismatched),
        'cache_hash_mismatch_count': len(mismatched),
        'cache_hash_mismatch_rate': _ratio(len(mismatched), len(cohort)),
        'run9_b1_call_count': len(b1_calls),
        'run9_b1_calls': b1_calls,
        'unverified_records': [
            {
                'case_id': record.case_id,
                'modality': record.modality,
                'cached_run_id': record.run_id,
            }
            for record in mismatched
        ],
        'single_record_compliance': dict(compliance),
        'violations': violations,
    }


def build_shadow_audit_report(
    raw_records: Sequence[Any],
    *,
    positive_budget: int = DEFAULT_POSITIVE_BUDGET,
    control_budget: int = DEFAULT_CONTROL_BUDGET,
    total_population: int = BORDER_FLAG_POPULATION_N,
    worst_slice_selector: Optional[Callable[[Sequence[int]], Optional[int]]] = None,
) -> Dict[str, Any]:
    '''Build the read-only shadow-audit report for one injected cache snapshot.'''
    records = [coerce_record(raw) for raw in raw_records]
    if not records:
        raise ValueError('no cached summaries supplied for the border-flag shadow audit')

    positives, controls = select_cohort(
        records,
        positive_budget=positive_budget,
        control_budget=control_budget,
    )
    cohort = list(positives) + list(controls)
    analyzed = [record for record in cohort if record.hash_matches]
    unverified = [record for record in cohort if not record.hash_matches]
    border_analyzed = [record for record in analyzed if record.touches_image_border]
    control_analyzed = [record for record in analyzed if not record.touches_image_border]

    compliance = single_record_compliance(records)
    integrity = integrity_summary(cohort, compliance)

    warnings: List[Dict[str, Any]] = []
    if len(positives) < int(positive_budget):
        warnings.append(
            {
                'code': 'fewer_positive_records_than_requested',
                'requested': int(positive_budget),
                'available': len(positives),
            }
        )
    if len(controls) < int(control_budget):
        warnings.append(
            {
                'code': 'fewer_control_records_than_requested',
                'requested': int(control_budget),
                'available': len(controls),
            }
        )

    blocking_findings: List[Dict[str, Any]] = []
    for entry in compliance['duplicate_keys']:
        blocking_findings.append(
            {
                'code': 'multiple_records_for_case_modality',
                'case_id': entry['case_id'],
                'modality': entry['modality'],
            }
        )
    blocking_findings.extend(integrity['violations'])

    return {
        'action': 'cmr_myo_border_import_shadow_audit',
        'mode': 'read_only_import_shadow',
        'population': {
            'flag': 'cmr_myo_touches_image_border',
            'declared_n': int(total_population),
            'cache_runs_reused': list(CACHE_RUN_IDS),
            'extractor_run_for_hash_mismatch': B1_EXTRACTOR_RUN_ID,
        },
        'cohort': {
            'positives_requested': int(positive_budget),
            'positives_selected': len(positives),
            'controls_requested': int(control_budget),
            'controls_selected': len(controls),
            'cohort_size': len(cohort),
            'max_cohort_size': MAX_COHORT_SIZE,
            'input_records': len(records),
            'analyzed_records': len(analyzed),
            'excluded_unverified_hash': len(unverified),
        },
        'cooccurrence': {
            'border_cases': cooccurrence_summary(border_analyzed, len(analyzed)),
            'control_cases': cooccurrence_summary(control_analyzed, len(analyzed)),
        },
        'slice_index': slice_index_profile(
            analyzed, worst_slice_selector=worst_slice_selector
        ),
        'integrity': integrity,
        'warnings': warnings,
        'violations': integrity['violations'],
        'blocking_findings': blocking_findings,
        'safety': {
            'read_only': True,
            'label_mutation': False,
            'production_rule_change': False,
            'threshold_change': False,
            'qc_safety_check_weakened': False,
            'external_io': False,
            'hash_mismatch_policy': 'run-9 B1 extractor only, always reported',
        },
    }
