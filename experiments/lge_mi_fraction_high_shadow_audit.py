"""Read-only shadow audit for lge mi_fraction_high."""
from __future__ import annotations
import hashlib, json, statistics
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

POSITIVE_COHORT = 'positive'
CONTROL_COHORT = 'control'
ALLOWED_CACHE_RUNS = frozenset(('run-4', 'run-5', 'run-9', 'run-13', 'run-14', 'run-15', 'run-16', 'run-17', 'run-18'))
RUN9_B1_EXTRACTOR = 'run-9:B1'
HIGH_MI_SCORE_THRESHOLD = 0.30
MAX_CONCENTRATION_FOR_TRUE_HIGH = 0.50
ABNORMAL_TOPOLOGIES = frozenset(('fragmented', 'multi_component', 'eccentric', 'discontinuous', 'detached'))

class ShadowAuditError(ValueError): pass
class DuplicateCaseModalityError(ShadowAuditError): pass
class ScopeViolationError(ShadowAuditError): pass
class MissingRun9B1EvidenceError(ShadowAuditError): pass
class CohortSizeError(ShadowAuditError): pass

def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def compute_record_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()

def _norm_cohort(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ('positive', 'pos', 'case', 'lge_mi_fraction_high'):
        return POSITIVE_COHORT
    if text in ('control', 'ctrl', 'negative', 'reference'):
        return CONTROL_COHORT
    raise ShadowAuditError(f'unsupported cohort: {value!r}')

def _norm_modality(value: Any) -> str:
    return str(value).strip().upper() or 'LGE'

def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ShadowAuditError(f'{name} must be numeric, got {value!r}') from exc

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y')

def _coerce(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ShadowAuditError('record must be a mapping')
    try:
        case_id = str(record['case_id']).strip()
        run_id = str(record['run_id']).strip()
        content_hash = str(record['content_hash']).strip()
    except KeyError as exc:
        raise ShadowAuditError(f'missing required record key: {exc.args[0]}') from exc
    if not case_id:
        raise ShadowAuditError('case_id must be non-empty')
    return {
        'case_id': case_id,
        'modality': _norm_modality(record.get('modality', 'LGE')),
        'cohort': _norm_cohort(record.get('cohort')),
        'run_id': run_id,
        'content_hash': content_hash,
        'summary': dict(record.get('summary', {})),
        'b1_payload': dict(record.get('b1_payload', {})),
    }

def _norm_summary(summary: Mapping[str, Any], source: str) -> dict[str, Any]:
    missing = [key for key in ('mi_score', 'worst_slice_fraction', 'mi_multiple_components', 'myo_topology') if key not in summary]
    if missing:
        raise ShadowAuditError(f'{source} summary missing fields: {missing}')
    return {
        'mi_score': _as_float(summary['mi_score'], 'mi_score'),
        'worst_slice_fraction': _as_float(summary['worst_slice_fraction'], 'worst_slice_fraction'),
        'mi_multiple_components': _as_bool(summary['mi_multiple_components']),
        'myo_topology': str(summary['myo_topology']).strip().lower() or 'unknown',
        'artifact_flags': tuple(str(x) for x in summary.get('artifact_flags', ())),
    }

def run9_b1_extractor(record: Mapping[str, Any]) -> dict[str, Any]:
    coerced = _coerce(record)
    if not coerced['b1_payload']:
        case_id = coerced['case_id']
        modality = coerced['modality']
        raise MissingRun9B1EvidenceError(f'run-9 B1 evidence required for hash mismatch: {case_id}/{modality}')
    return _norm_summary(coerced['b1_payload'], RUN9_B1_EXTRACTOR)

def dedupe_validate(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    coerced = [_coerce(record) for record in records]
    counts = Counter((record['case_id'], record['modality']) for record in coerced)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    if duplicates:
        key = sorted(duplicates)[0]
        raise DuplicateCaseModalityError(f'duplicate case/modality records: {key[0]}/{key[1]} count={duplicates[key]}')
    for record in coerced:
        if record['run_id'] not in ALLOWED_CACHE_RUNS:
            run_id = record['run_id']
            raise ScopeViolationError(f'cache run not allowed for this shadow audit: {run_id}')
    return sorted(coerced, key=lambda record: (record['case_id'], record['modality']))

def _resolve(record: Mapping[str, Any], hash_lookup: Mapping[tuple[str, str], str] | None) -> tuple[dict[str, Any], str, bool, dict[str, Any] | None]:
    if hash_lookup is None:
        expected_hash = compute_record_hash(record['summary'])
    else:
        expected_hash = hash_lookup.get((record['case_id'], record['modality']))
        if expected_hash is None:
            expected_hash = compute_record_hash(record['summary'])
    if record['content_hash'] == expected_hash:
        return _norm_summary(record['summary'], record['run_id']), record['run_id'], False, None
    fallback = run9_b1_extractor(record)
    provenance = {
        'case_id': record['case_id'],
        'modality': record['modality'],
        'cached_run': record['run_id'],
        'extractor': RUN9_B1_EXTRACTOR,
        'reason': 'content_hash_mismatch',
    }
    return fallback, RUN9_B1_EXTRACTOR, True, provenance

def adjudicate_positive(summary: Mapping[str, Any], source: str, fallback_used: bool, case_id: str = 'unknown') -> dict[str, Any]:
    normalised = _norm_summary(summary, source)
    mi_score = normalised['mi_score']
    worst = normalised['worst_slice_fraction']
    multi = normalised['mi_multiple_components']
    topology = normalised['myo_topology']
    artifacts = normalised['artifact_flags']
    reasons = []
    if artifacts:
        reasons.append('artifact_flags:' + ','.join(artifacts))
        verdict = 'artifact'
    elif mi_score < HIGH_MI_SCORE_THRESHOLD:
        reasons.append('mi_score_below_high_threshold')
        verdict = 'indeterminate'
    else:
        reasons.append('mi_score_ge_high_threshold')
        overestimated = False
        if worst > MAX_CONCENTRATION_FOR_TRUE_HIGH:
            reasons.append('worst_slice_concentration_above_limit')
            overestimated = True
        if multi:
            reasons.append('mi_multiple_components_present')
            overestimated = True
        if topology in ABNORMAL_TOPOLOGIES:
            reasons.append('abnormal_myo_topology:' + topology)
            overestimated = True
        verdict = 'overestimated' if overestimated else 'true_high_mi'
    return {
        'case_id': case_id,
        'verdict': verdict,
        'reasons': tuple(reasons),
        'mi_score': mi_score,
        'worst_slice_fraction': worst,
        'source': source,
        'fallback_used': fallback_used,
    }

def _dist(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {'n': 0, 'mean': 0.0, 'median': 0.0, 'minimum': 0.0, 'maximum': 0.0}
    return {
        'n': len(values),
        'mean': float(statistics.fmean(values)),
        'median': float(statistics.median(values)),
        'minimum': float(min(values)),
        'maximum': float(max(values)),
    }

def _rate(flags: Sequence[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)

def audit_lge_mi_fraction_high(records: Iterable[Mapping[str, Any]], *, hash_lookup: Mapping[tuple[str, str], str] | None = None, positive_quota: int = 30, control_quota: int = 10) -> dict[str, Any]:
    ordered = dedupe_validate(records)
    cohort_counts = Counter(record['cohort'] for record in ordered)
    if cohort_counts[POSITIVE_COHORT] != positive_quota:
        raise CohortSizeError(f'expected {positive_quota} positive records, got {cohort_counts[POSITIVE_COHORT]}')
    if cohort_counts[CONTROL_COHORT] != control_quota:
        raise CohortSizeError(f'expected {control_quota} control records, got {cohort_counts[CONTROL_COHORT]}')
    resolved = []
    for record in ordered:
        summary, source, fallback_used, provenance = _resolve(record, hash_lookup)
        resolved.append((record, summary, source, fallback_used, provenance))
    fallback_provenance = tuple(item[4] for item in resolved if item[4] is not None)
    reused_count = sum(1 for item in resolved if not item[3])
    positive_items = [item for item in resolved if item[0]['cohort'] == POSITIVE_COHORT]
    control_items = [item for item in resolved if item[0]['cohort'] == CONTROL_COHORT]
    adjudications = tuple(adjudicate_positive(item[1], item[2], item[3], item[0]['case_id']) for item in positive_items)
    verdict_counts = dict(Counter(item['verdict'] for item in adjudications))
    positive_mi = [item[1]['mi_score'] for item in positive_items]
    control_mi = [item[1]['mi_score'] for item in control_items]
    positive_worst = [item[1]['worst_slice_fraction'] for item in positive_items]
    control_worst = [item[1]['worst_slice_fraction'] for item in control_items]
    positive_multi = [item[1]['mi_multiple_components'] for item in positive_items]
    control_multi = [item[1]['mi_multiple_components'] for item in control_items]
    positive_topology = dict(Counter(item[1]['myo_topology'] for item in positive_items))
    control_topology = dict(Counter(item[1]['myo_topology'] for item in control_items))
    positive_mi_dist = _dist(positive_mi)
    control_mi_dist = _dist(control_mi)
    positive_worst_dist = _dist(positive_worst)
    control_worst_dist = _dist(control_worst)
    per_case_modality_counts = dict(Counter(record['case_id'] + '|' + record['modality'] for record in ordered))
    total = len(ordered)
    return {
        'positive_count': len(positive_items),
        'control_count': len(control_items),
        'cache_reuse_rate': (reused_count / total) if total else 0.0,
        'cache_reused_records': reused_count,
        'hash_mismatch_fallbacks': len(fallback_provenance),
        'duplicate_case_modality_records': 0,
        'adjudications': adjudications,
        'positive_verdict_counts': verdict_counts,
        'positive_mi_score': positive_mi_dist,
        'control_mi_score': control_mi_dist,
        'positive_worst_slice_concentration': positive_worst_dist,
        'control_worst_slice_concentration': control_worst_dist,
        'worst_slice_concentration_delta': positive_worst_dist['mean'] - control_worst_dist['mean'],
        'positive_mi_multiple_components_rate': _rate(positive_multi),
        'control_mi_multiple_components_rate': _rate(control_multi),
        'positive_myo_topology_counts': positive_topology,
        'control_myo_topology_counts': control_topology,
        'fallback_provenance': fallback_provenance,
        'per_case_modality_counts': per_case_modality_counts,
    }
