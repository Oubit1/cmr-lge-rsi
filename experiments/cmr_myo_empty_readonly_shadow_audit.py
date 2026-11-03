'''Read-only shadow audit for cmr_myo_empty positives against cmr controls.

This module never writes labels, rules, thresholds, patient images, or runtime
databases. It consumes case payload bytes and caller-supplied cache records, then
produces audit findings and aggregate metrics.

Contract:
* Cache runs considered: 4, 5, 9, 13, 14, 15, 16, 17, 18.
* A cache record is a valid hit only when payload_hash matches SHA-256 of the
  case payload.
* On hash mismatch the only permitted extractor is the registered run-9 B1
  extractor; the mismatch is registered in the finding events.
* Cache misses use the same registered read-only extractor as a fallback and
  are reported separately from hash mismatches.
* No QC safety check, label, rule, or threshold is modified.
'''

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

POSITIVE_LABEL = 'cmr_myo_empty'
CONTROL_LABEL = 'cmr'
CACHE_RUNS = (4, 5, 9, 13, 14, 15, 16, 17, 18)
REGISTERED_EXTRACTOR_RUN = 9
REGISTERED_EXTRACTOR_VARIANT = 'B1'
MIN_COMPLETION_RATE = 0.95
POSITIVE_CLASSIFICATIONS = frozenset({'failure', 'fail', 'positive', 'cmr_myo_empty'})
COOCCURRENCE_FEATURES = {
    'small_area': ('small_area',),
    'excess_components': ('excess_components', 'excess_component'),
    'border': ('border', 'touches_border', 'border_contact'),
    'temporal_hole': ('temporal_hole', 'missing_temporal_hole'),
    'centroid_drift': ('centroid_drift', 'centroid_drift_flag'),
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    label: str
    slice_index: int
    temporal_index: int
    payload: bytes = b''

    @property
    def is_positive(self) -> bool:
        return self.label == POSITIVE_LABEL

    @property
    def is_control(self) -> bool:
        return self.label == CONTROL_LABEL

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.case_id, self.slice_index, self.temporal_index)


@dataclass(frozen=True)
class CacheRecord:
    run_id: int
    case_id: str
    slice_index: int
    temporal_index: int
    payload_hash: str
    classification: str = 'pass'
    features: Mapping[str, Any] = field(default_factory=dict)
    source: str = 'cache'

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.case_id, self.slice_index, self.temporal_index)


@dataclass(frozen=True)
class AuditFinding:
    case_id: str
    label: str
    slice_index: int
    temporal_index: int
    status: str
    route: str
    classification: Optional[str]
    features: Mapping[str, Any]
    cache_hit: bool
    hash_mismatch: bool
    events: tuple[Mapping[str, Any], ...] = ()

    def as_report(self) -> dict[str, Any]:
        return {
            'case_id': self.case_id,
            'label': self.label,
            'slice_index': self.slice_index,
            'temporal_index': self.temporal_index,
            'status': self.status,
            'route': self.route,
            'classification': self.classification,
            'features': dict(self.features),
            'cache_hit': self.cache_hit,
            'hash_mismatch': self.hash_mismatch,
            'events': [dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class AuditSummary:
    total_cases: int
    completed_cases: int
    completion_rate: float
    cache_hits: int
    cache_misses: int
    hash_mismatches: int
    cache_hit_rate: float
    cache_miss_rate: float
    hash_mismatch_rate: float
    positive_total: int
    positive_evaluated: int
    positive_failures: int
    positive_true_failure_rate: float
    positive_true_failure_interval: tuple[float, float]
    control_total: int
    control_evaluated: int
    control_false_positives: int
    control_false_positive_rate: float
    control_false_positive_interval: tuple[float, float]
    stratification: Mapping[str, Any]
    cooccurrence_rates: Mapping[str, Any]
    findings: tuple[AuditFinding, ...] = ()

    def as_report(self) -> dict[str, Any]:
        return {
            'total_cases': self.total_cases,
            'completed_cases': self.completed_cases,
            'completion_rate': self.completion_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hash_mismatches': self.hash_mismatches,
            'cache_hit_rate': self.cache_hit_rate,
            'cache_miss_rate': self.cache_miss_rate,
            'hash_mismatch_rate': self.hash_mismatch_rate,
            'positive_total': self.positive_total,
            'positive_evaluated': self.positive_evaluated,
            'positive_failures': self.positive_failures,
            'positive_true_failure_rate': self.positive_true_failure_rate,
            'positive_true_failure_interval': list(self.positive_true_failure_interval),
            'control_total': self.control_total,
            'control_evaluated': self.control_evaluated,
            'control_false_positives': self.control_false_positives,
            'control_false_positive_rate': self.control_false_positive_rate,
            'control_false_positive_interval': list(self.control_false_positive_interval),
            'stratification': dict(self.stratification),
            'cooccurrence_rates': {key: dict(value) for key, value in self.cooccurrence_rates.items()},
            'findings': [finding.as_report() for finding in self.findings],
        }


def payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cache_lookup(cache: Any, case: CaseRecord) -> Optional[CacheRecord]:
    if cache is None:
        return None
    if callable(cache):
        return cache(case)
    if isinstance(cache, Mapping):
        record = cache.get(case.key)
        if record is None:
            record = cache.get(case.case_id)
        return record
    raise TypeError('cache must be a mapping, callable, or None')


def _feature_flag(features: Mapping[str, Any], names: Sequence[str]) -> bool:
    for name in names:
        if name not in features:
            continue
        value = features[name]
        if isinstance(value, bool):
            return value
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            return bool(value)
    return False


def cooccurrence_flags(features: Mapping[str, Any]) -> dict[str, bool]:
    return {
        name: _feature_flag(features, aliases)
        for name, aliases in COOCCURRENCE_FEATURES.items()
    }


def is_positive_classification(classification: Optional[str]) -> bool:
    if classification is None:
        return False
    return str(classification).strip().lower() in POSITIVE_CLASSIFICATIONS


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = z * ((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n)) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stratify_cases(cases: Iterable[CaseRecord]) -> dict[str, Any]:
    by_slice: dict[str, list[str]] = defaultdict(list)
    by_temporal: dict[str, list[str]] = defaultdict(list)
    by_temporal_neighbor: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_slice[f'slice_{case.slice_index}'].append(case.case_id)
        by_temporal[f'temporal_{case.temporal_index}'].append(case.case_id)
        left = min(case.temporal_index, case.temporal_index + 1)
        right = max(case.temporal_index, case.temporal_index + 1)
        by_temporal_neighbor[f'temporal_neighbor_{left}_{right}'].append(case.case_id)
    return {
        'by_slice': dict(by_slice),
        'by_temporal': dict(by_temporal),
        'by_temporal_neighbor': dict(by_temporal_neighbor),
    }


def registered_extractor_event(
    case: CaseRecord,
    record: Optional[CacheRecord],
    reason: str,
) -> dict[str, Any]:
    return {
        'event': f'cmr_myo_empty_shadow_audit.{reason}',
        'case_id': case.case_id,
        'slice_index': case.slice_index,
        'temporal_index': case.temporal_index,
        'registered_extractor_run': REGISTERED_EXTRACTOR_RUN,
        'registered_extractor_variant': REGISTERED_EXTRACTOR_VARIANT,
        'cache_run_id': getattr(record, 'run_id', None),
        'read_only': True,
    }


def _extract(
    case: CaseRecord,
    extractor: Callable[[CaseRecord], Mapping[str, Any]],
) -> tuple[Optional[str], Mapping[str, Any]]:
    extracted = extractor(case)
    return extracted.get('classification'), extracted.get('features', {})


def audit_case(
    case: CaseRecord,
    cache: Any,
    extractor: Optional[Callable[[CaseRecord], Mapping[str, Any]]] = None,
) -> AuditFinding:
    record = cache_lookup(cache, case)
    if record is not None:
        expected_hash = payload_hash(case.payload)
        if record.payload_hash == expected_hash:
            return AuditFinding(
                case_id=case.case_id,
                label=case.label,
                slice_index=case.slice_index,
                temporal_index=case.temporal_index,
                status='completed',
                route='cache_hit',
                classification=record.classification,
                features=record.features,
                cache_hit=True,
                hash_mismatch=False,
                events=(),
            )
        event = registered_extractor_event(case, record, 'hash_mismatch')
        if extractor is None:
            return AuditFinding(
                case_id=case.case_id,
                label=case.label,
                slice_index=case.slice_index,
                temporal_index=case.temporal_index,
                status='incomplete',
                route='cache_hash_mismatch_run9_b1_extractor_unavailable',
                classification=None,
                features={},
                cache_hit=False,
                hash_mismatch=True,
                events=(event,),
            )
        classification, features = _extract(case, extractor)
        return AuditFinding(
            case_id=case.case_id,
            label=case.label,
            slice_index=case.slice_index,
            temporal_index=case.temporal_index,
            status='completed',
            route='cache_hash_mismatch_run9_b1',
            classification=classification,
            features=features,
            cache_hit=False,
            hash_mismatch=True,
            events=(event,),
        )

    event = registered_extractor_event(case, None, 'cache_miss')
    if extractor is None:
        return AuditFinding(
            case_id=case.case_id,
            label=case.label,
            slice_index=case.slice_index,
            temporal_index=case.temporal_index,
            status='incomplete',
            route='cache_miss_run9_b1_extractor_unavailable',
            classification=None,
            features={},
            cache_hit=False,
            hash_mismatch=False,
            events=(event,),
        )
    classification, features = _extract(case, extractor)
    return AuditFinding(
        case_id=case.case_id,
        label=case.label,
        slice_index=case.slice_index,
        temporal_index=case.temporal_index,
        status='completed',
        route='cache_miss_run9_b1',
        classification=classification,
        features=features,
        cache_hit=False,
        hash_mismatch=False,
        events=(event,),
    )


def audit_cases(
    cases: Iterable[CaseRecord],
    cache: Any,
    extractor: Optional[Callable[[CaseRecord], Mapping[str, Any]]] = None,
    enforce_completion: bool = False,
    min_completion: float = MIN_COMPLETION_RATE,
) -> AuditSummary:
    case_tuple = tuple(cases)
    findings = tuple(audit_case(case, cache, extractor) for case in case_tuple)
    total = len(case_tuple)
    completed = tuple(finding for finding in findings if finding.status == 'completed')
    completed_count = len(completed)
    completion_rate = completed_count / total if total else 0.0
    if enforce_completion and completion_rate < min_completion:
        raise ValueError(
            f'audit completion rate {completion_rate:.4f} below required {min_completion:.4f}'
        )

    cache_hits = sum(1 for finding in findings if finding.cache_hit)
    hash_mismatches = sum(1 for finding in findings if finding.hash_mismatch)
    cache_misses = total - cache_hits - hash_mismatches

    positive_total = sum(1 for case in case_tuple if case.is_positive)
    control_total = sum(1 for case in case_tuple if case.is_control)
    positive_completed = [finding for finding in completed if finding.label == POSITIVE_LABEL]
    control_completed = [finding for finding in completed if finding.label == CONTROL_LABEL]
    positive_failures = sum(
        1 for finding in positive_completed if is_positive_classification(finding.classification)
    )
    control_false_positives = sum(
        1 for finding in control_completed if is_positive_classification(finding.classification)
    )
    positive_evaluated = len(positive_completed)
    control_evaluated = len(control_completed)
    positive_rate = positive_failures / positive_evaluated if positive_evaluated else 0.0
    control_rate = control_false_positives / control_evaluated if control_evaluated else 0.0
    positive_interval = wilson_interval(positive_failures, positive_evaluated)
    control_interval = wilson_interval(control_false_positives, control_evaluated)

    cooccurrence_rates: dict[str, Any] = {}
    for flag in COOCCURRENCE_FEATURES:
        positive_flag_count = sum(
            1 for finding in positive_completed if cooccurrence_flags(finding.features)[flag]
        )
        control_flag_count = sum(
            1 for finding in control_completed if cooccurrence_flags(finding.features)[flag]
        )
        all_flag_count = sum(
            1 for finding in completed if cooccurrence_flags(finding.features)[flag]
        )
        cooccurrence_rates[flag] = {
            'positive_count': positive_flag_count,
            'positive_rate': positive_flag_count / positive_evaluated if positive_evaluated else 0.0,
            'control_count': control_flag_count,
            'control_rate': control_flag_count / control_evaluated if control_evaluated else 0.0,
            'overall_count': all_flag_count,
            'overall_rate': all_flag_count / completed_count if completed_count else 0.0,
        }

    return AuditSummary(
        total_cases=total,
        completed_cases=completed_count,
        completion_rate=completion_rate,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hash_mismatches=hash_mismatches,
        cache_hit_rate=cache_hits / total if total else 0.0,
        cache_miss_rate=cache_misses / total if total else 0.0,
        hash_mismatch_rate=hash_mismatches / total if total else 0.0,
        positive_total=positive_total,
        positive_evaluated=positive_evaluated,
        positive_failures=positive_failures,
        positive_true_failure_rate=positive_rate,
        positive_true_failure_interval=positive_interval,
        control_total=control_total,
        control_evaluated=control_evaluated,
        control_false_positives=control_false_positives,
        control_false_positive_rate=control_rate,
        control_false_positive_interval=control_interval,
        stratification=stratify_cases(case_tuple),
        cooccurrence_rates=cooccurrence_rates,
        findings=findings,
    )


def check_completion_acceptance(summary: AuditSummary, min_completion: float = MIN_COMPLETION_RATE) -> None:
    if summary.completion_rate < min_completion:
        raise ValueError(
            f'audit completion rate {summary.completion_rate:.4f} below required {min_completion:.4f}'
        )
