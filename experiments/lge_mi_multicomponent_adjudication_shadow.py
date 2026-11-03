'''Bounded read-only shadow adjudication for LGE MI multiple components.

This module is deliberately self-contained and side-effect free. It consumes
already-cached case evidence and produces shadow metrics only. It never writes
to disk, never mutates production labels, and never changes review rules or
thresholds.
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Sequence

__all__ = [
    'ADJUDICATION_LABELS',
    'BOUNDED_SAMPLE_SIZE',
    'CACHE_RUN_IDS',
    'CONTROL_SAMPLE_SIZE',
    'HASH_MISMATCH_EXTRACTOR',
    'POSITIVE_SAMPLE_SIZE',
    'READER_AGREEMENT_FLOOR',
    'SUPPORT_MIN_AREA_MM2',
    'SUPPORT_MIN_CENTROID_DISTANCE_MM',
    'SUPPORT_MIN_CROSS_SLICE_COUNT',
    'CaseEvidence',
    'ComponentEvidence',
    'classify_case',
    'is_downgrade_candidate',
    'resolve_cache_source',
    'secondary_component_deficits',
    'select_bounded_sample',
    'summarize_shadow_audit',
]

CACHE_RUN_IDS: tuple[str, ...] = (
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

HASH_MISMATCH_EXTRACTOR = 'run-9:B1'
READER_AGREEMENT_FLOOR = 0.8
POSITIVE_SAMPLE_SIZE = 50
CONTROL_SAMPLE_SIZE = 10
BOUNDED_SAMPLE_SIZE = POSITIVE_SAMPLE_SIZE + CONTROL_SAMPLE_SIZE

ADJUDICATION_LABELS: tuple[str, ...] = (
    'true_multifocal',
    'fragmentation_or_noise',
    'indeterminate',
)

# Shadow-only support criteria. These are local audit constants used only to
# estimate downgrade candidates. They are not production gates.
SUPPORT_MIN_AREA_MM2 = 15.0
SUPPORT_MIN_CROSS_SLICE_COUNT = 2
SUPPORT_MIN_CENTROID_DISTANCE_MM = 10.0


@dataclass(frozen=True)
class ComponentEvidence:
    '''One myocardial-infarction component from a cached case payload.'''

    area_mm2: float
    centroid_distance_mm: float
    touches_myocardium: bool
    cross_slice_support: int

    def is_supported(self) -> bool:
        '''Return True when the component meets shadow support criteria.'''

        return (
            self.area_mm2 >= SUPPORT_MIN_AREA_MM2
            and self.touches_myocardium
            and self.cross_slice_support >= SUPPORT_MIN_CROSS_SLICE_COUNT
        )


@dataclass(frozen=True)
class CaseEvidence:
    '''Cached evidence for a single case in the bounded shadow audit.'''

    case_id: str
    stratum: str
    cache_run_id: str
    components: Sequence[ComponentEvidence]
    cache_hash: str = ''
    expected_cache_hash: str = ''
    reader_label: str | None = None
    slice_count: int = 0


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _primary_component(components: Sequence[ComponentEvidence]) -> ComponentEvidence | None:
    if not components:
        return None
    return max(
        components,
        key=lambda component: (
            component.area_mm2,
            component.cross_slice_support,
            -component.centroid_distance_mm,
        ),
    )


def _secondary_components(
    components: Sequence[ComponentEvidence],
) -> tuple[ComponentEvidence, ...]:
    primary = _primary_component(components)
    if primary is None:
        return ()
    remaining = list(components)
    remaining.remove(primary)
    return tuple(remaining)


def classify_case(case: CaseEvidence) -> str:
    '''Return shadow three-class adjudication for one case.'''

    components = tuple(case.components)
    if not components:
        return 'indeterminate'
    if len(components) == 1:
        return 'fragmentation_or_noise'

    secondary = _secondary_components(components)
    if not secondary:
        return 'fragmentation_or_noise'

    supported_secondary = [component for component in secondary if component.is_supported()]
    unsupported_secondary = [
        component for component in secondary if not component.is_supported()
    ]

    if supported_secondary and not unsupported_secondary:
        if any(
            component.centroid_distance_mm >= SUPPORT_MIN_CENTROID_DISTANCE_MM
            for component in supported_secondary
        ):
            return 'true_multifocal'
        return 'indeterminate'

    if not supported_secondary:
        return 'fragmentation_or_noise'

    if any(
        component.centroid_distance_mm >= SUPPORT_MIN_CENTROID_DISTANCE_MM
        for component in supported_secondary
    ):
        return 'indeterminate'
    return 'fragmentation_or_noise'


def resolve_cache_source(case: CaseEvidence) -> dict[str, Any]:
    '''Resolve the only permitted evidence source for a case.

    Allowed cached runs are reused when their hash matches. Hash mismatches or
    unsupported run ids are reported and routed to the run-9 B1 extractor.
    '''

    if case.cache_run_id not in CACHE_RUN_IDS:
        return {
            'case_id': case.case_id,
            'run_id': 'run-9',
            'extractor': 'B1',
            'reason': 'unsupported_cache_run',
            'report_required': True,
            'reused_cache': False,
        }

    if case.expected_cache_hash and case.cache_hash != case.expected_cache_hash:
        return {
            'case_id': case.case_id,
            'run_id': 'run-9',
            'extractor': 'B1',
            'reason': 'hash_mismatch',
            'report_required': True,
            'reused_cache': False,
        }

    return {
        'case_id': case.case_id,
        'run_id': case.cache_run_id,
        'extractor': None,
        'reason': 'cache_reuse',
        'report_required': False,
        'reused_cache': True,
    }


def select_bounded_sample(
    cases: Iterable[CaseEvidence],
    positive_stratum: str = 'mi_multiple_components',
    control_stratum: str = 'lge',
    positive_limit: int = POSITIVE_SAMPLE_SIZE,
    control_limit: int = CONTROL_SAMPLE_SIZE,
) -> tuple[CaseEvidence, ...]:
    '''Select a deterministic bounded sample of positives plus controls.'''

    ordered = sorted(cases, key=lambda case: case.case_id)
    positives = [
        case for case in ordered if case.stratum == positive_stratum
    ][:positive_limit]
    controls = [
        case for case in ordered if case.stratum == control_stratum
    ][:control_limit]
    return tuple(positives + controls)


def secondary_component_deficits(case: CaseEvidence) -> dict[str, Any]:
    '''Summarize support deficits among non-primary components.'''

    secondary = _secondary_components(tuple(case.components))
    total = len(secondary)
    lacking_myocardium = sum(
        1 for component in secondary if not component.touches_myocardium
    )
    lacking_cross_slice = sum(
        1
        for component in secondary
        if component.cross_slice_support < SUPPORT_MIN_CROSS_SLICE_COUNT
    )
    lacking_either = sum(
        1
        for component in secondary
        if not (
            component.touches_myocardium
            and component.cross_slice_support >= SUPPORT_MIN_CROSS_SLICE_COUNT
        )
    )
    supported = sum(1 for component in secondary if component.is_supported())

    return {
        'case_id': case.case_id,
        'secondary_component_count': total,
        'supported_secondary_component_count': supported,
        'lacking_myocardial_adjacency': lacking_myocardium,
        'lacking_cross_slice_support': lacking_cross_slice,
        'lacking_myocardial_adjacency_or_cross_slice': lacking_either,
        'lacking_myocardial_adjacency_fraction': _safe_div(lacking_myocardium, total),
        'lacking_cross_slice_support_fraction': _safe_div(lacking_cross_slice, total),
        'lacking_myocardial_adjacency_or_cross_slice_fraction': _safe_div(
            lacking_either, total
        ),
    }


def is_downgrade_candidate(case: CaseEvidence) -> bool:
    '''Return True if a mi_multiple_components case may be downgraded in shadow.

    This is an audit-only estimate. It does not change any production label.
    '''

    if case.stratum != 'mi_multiple_components':
        return False
    if classify_case(case) == 'fragmentation_or_noise':
        return True
    secondary = _secondary_components(tuple(case.components))
    if not secondary:
        return True
    return not any(component.is_supported() for component in secondary)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {'n': 0, 'min': None, 'median': None, 'p90': None, 'max': None}
    p90_index = min(len(clean) - 1, max(0, int(round(0.9 * (len(clean) - 1)))))
    return {
        'n': len(clean),
        'min': clean[0],
        'median': median(clean),
        'p90': clean[p90_index],
        'max': clean[-1],
    }


def summarize_shadow_audit(cases: Iterable[CaseEvidence]) -> dict[str, Any]:
    '''Build the read-only shadow-audit summary for a bounded case set.'''

    bounded = select_bounded_sample(cases)
    total = len(bounded)

    labels = Counter(classify_case(case) for case in bounded)
    label_proportions = {
        label: (_safe_div(labels.get(label, 0), total) if total else 0.0)
        for label in ADJUDICATION_LABELS
    }

    reader_pairs = [
        (classify_case(case), case.reader_label)
        for case in bounded
        if case.reader_label in ADJUDICATION_LABELS
    ]
    reader_matches = sum(
        1 for shadow_label, reader_label in reader_pairs if shadow_label == reader_label
    )
    reader_agreement = _safe_div(reader_matches, len(reader_pairs)) if reader_pairs else None

    component_counts = [len(case.components) for case in bounded]
    component_areas = [
        component.area_mm2 for case in bounded for component in case.components
    ]
    secondary_spacings = [
        component.centroid_distance_mm
        for case in bounded
        for component in _secondary_components(tuple(case.components))
    ]

    deficits = [secondary_component_deficits(case) for case in bounded]
    secondary_total = sum(item['secondary_component_count'] for item in deficits)
    lacking_myocardium = sum(item['lacking_myocardial_adjacency'] for item in deficits)
    lacking_cross_slice = sum(item['lacking_cross_slice_support'] for item in deficits)
    lacking_either = sum(
        item['lacking_myocardial_adjacency_or_cross_slice'] for item in deficits
    )

    mi_cases = [case for case in bounded if case.stratum == 'mi_multiple_components']
    downgrade_cases = [case for case in mi_cases if is_downgrade_candidate(case)]

    cache_sources = [resolve_cache_source(case) for case in bounded]
    cache_reports = [source for source in cache_sources if source['report_required']]

    return {
        'shadow_only': True,
        'write_operations': 0,
        'label_mutations': 0,
        'rule_or_threshold_mutations': 0,
        'bounded_limit': BOUNDED_SAMPLE_SIZE,
        'bounded_case_count': total,
        'label_counts': dict(labels),
        'label_proportions': label_proportions,
        'reader_agreement': {
            'n': len(reader_pairs),
            'matches': reader_matches,
            'agreement': reader_agreement,
            'floor': READER_AGREEMENT_FLOOR,
            'meets_floor': (
                reader_agreement is not None and reader_agreement >= READER_AGREEMENT_FLOOR
            ),
        },
        'component_count_distribution': _distribution(component_counts),
        'component_area_mm2_distribution': _distribution(component_areas),
        'secondary_component_spacing_mm_distribution': _distribution(secondary_spacings),
        'secondary_component_deficits': {
            'secondary_component_count': secondary_total,
            'lacking_myocardial_adjacency': lacking_myocardium,
            'lacking_cross_slice_support': lacking_cross_slice,
            'lacking_myocardial_adjacency_or_cross_slice': lacking_either,
            'lacking_myocardial_adjacency_fraction': _safe_div(
                lacking_myocardium, secondary_total
            ),
            'lacking_cross_slice_support_fraction': _safe_div(
                lacking_cross_slice, secondary_total
            ),
            'lacking_myocardial_adjacency_or_cross_slice_fraction': _safe_div(
                lacking_either, secondary_total
            ),
        },
        'downgrade_estimate': {
            'stratum': 'mi_multiple_components',
            'stratum_case_count': len(mi_cases),
            'downgrade_candidate_count': len(downgrade_cases),
            'downgrade_candidate_fraction': _safe_div(len(downgrade_cases), len(mi_cases)),
            'downgrade_candidate_case_ids': [case.case_id for case in downgrade_cases],
            'shadow_only': True,
        },
        'cache_source_report': {
            'allowed_cache_run_ids': list(CACHE_RUN_IDS),
            'hash_mismatch_extractor': HASH_MISMATCH_EXTRACTOR,
            'sources': cache_sources,
            'reports_required': cache_reports,
        },
    }
