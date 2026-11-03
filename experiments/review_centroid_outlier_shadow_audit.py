from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ALLOWED_CACHE_RUNS = (4, 5, 9, 13, 14, 15, 16, 17, 18)
EXCLUDED_RUNS = (30,)
TARGET_POSITIVES = 30
TARGET_CONTROLS = 10
ADDRESSABILITY_MIN = 0.95
REVIEW_FLAG = 'review_centroid_outlier'
CENTROID_DRIFT_FLAG = 'cmr_centroid_drift'
MYO_TOPOLOGY_FLAG = 'myo_topology'
EXISTING_EXPLANATORY_FLAGS = frozenset([CENTROID_DRIFT_FLAG, MYO_TOPOLOGY_FLAG])
IMPORT_EXTRACTOR_RUN = 9
IMPORT_EXTRACTOR_STAGE = 'B1'
MAX_IMPORT_EXTRACTOR_CALLS = 1
STRATUM_DRIFT_PRESENT = 'drift_present'
STRATUM_DRIFT_ABSENT = 'drift_absent'


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    modality: str
    run_id: int
    flags: frozenset = frozenset()
    slice_index: Optional[int] = None
    frame_index: Optional[int] = None

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def is_addressable(self) -> bool:
        return self.slice_index is not None and self.frame_index is not None


@dataclass(frozen=True)
class CachedSummary:
    run_id: int
    digest: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HashMismatchReport:
    source_run_id: int
    expected_digest: Optional[str]
    observed_digest: str
    import_extractor_called: bool
    import_extractor_run: Optional[int] = None
    import_extractor_stage: Optional[str] = None
    replacement_payload: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CacheAudit:
    eligible_runs: Tuple[int, ...]
    checked_count: int
    matched_count: int
    mismatched_count: int
    import_extractor_calls: int
    hash_checked_rate: float
    hash_match_rate: float
    mismatches: Tuple[HashMismatchReport, ...] = ()


@dataclass(frozen=True)
class ShadowAuditReport:
    positive_selected_count: int
    control_selected_count: int
    positive_addressable_count: int
    positive_addressability_rate: float
    unique_case_modality_count: int
    duplicate_case_modality_count: int
    centroid_drift_cooccurrence_count: int
    centroid_drift_cooccurrence_rate: float
    myo_topology_cooccurrence_count: int
    myo_topology_cooccurrence_rate: float
    unexplained_by_existing_flags_count: int
    unexplained_by_existing_flags_rate: float
    cache_audit: CacheAudit
    selected_positives: Tuple[CaseRecord, ...] = ()
    selected_controls: Tuple[CaseRecord, ...] = ()
    notes: Tuple[str, ...] = ()


def _coerce_flags(value: Any) -> frozenset:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset([value])
    return frozenset(str(item) for item in value)


def as_case(record: Any) -> CaseRecord:
    if isinstance(record, CaseRecord):
        return record
    if isinstance(record, Mapping):
        return CaseRecord(
            case_id=str(record['case_id']),
            modality=str(record['modality']),
            run_id=int(record['run_id']),
            flags=_coerce_flags(record.get('flags')),
            slice_index=record.get('slice_index'),
            frame_index=record.get('frame_index'),
        )
    return CaseRecord(str(record[0]), str(record[1]), int(record[2]))


def as_cached_summary(record: Any) -> CachedSummary:
    if isinstance(record, CachedSummary):
        return record
    return CachedSummary(
        run_id=int(record['run_id']),
        digest=str(record['digest']),
        payload=dict(record.get('payload') or {}),
    )


def _prefer(candidate: CaseRecord, current: CaseRecord) -> bool:
    if candidate.is_addressable() != current.is_addressable():
        return candidate.is_addressable()
    return candidate.run_id < current.run_id


def dedupe_case_modality(records: Iterable[Any]) -> List[CaseRecord]:
    seen: Dict[Tuple[str, str], CaseRecord] = {}
    order: List[Tuple[str, str]] = []
    for raw in records:
        case = as_case(raw)
        key = (case.case_id, case.modality)
        if key not in seen:
            seen[key] = case
            order.append(key)
            continue
        if _prefer(case, seen[key]):
            seen[key] = case
    return [seen[key] for key in order]


def stratum_key(case: CaseRecord) -> str:
    if case.has_flag(CENTROID_DRIFT_FLAG):
        return STRATUM_DRIFT_PRESENT
    return STRATUM_DRIFT_ABSENT


def _stratified_take(pool: Sequence[CaseRecord], limit: int) -> List[CaseRecord]:
    buckets: Dict[str, List[CaseRecord]] = {STRATUM_DRIFT_PRESENT: [], STRATUM_DRIFT_ABSENT: []}
    for case in sorted(pool, key=lambda item: (item.case_id, item.modality, item.run_id)):
        buckets[stratum_key(case)].append(case)
    ordered: List[CaseRecord] = []
    keys = sorted(buckets.keys())
    while len(ordered) < limit:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if bucket and len(ordered) < limit:
                ordered.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


def select_cohort(cases: Iterable[Any]) -> Tuple[List[CaseRecord], List[CaseRecord]]:
    eligible = [c for c in dedupe_case_modality(cases) if c.run_id not in EXCLUDED_RUNS]
    positives = [c for c in eligible if c.has_flag(REVIEW_FLAG)]
    controls = [c for c in eligible if not c.has_flag(REVIEW_FLAG)]
    return _stratified_take(positives, TARGET_POSITIVES), _stratified_take(controls, TARGET_CONTROLS)


def audit_cache(
    cached_summaries: Iterable[Any],
    observed_digests: Mapping[int, str],
    import_extractor: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
) -> CacheAudit:
    summaries = [as_cached_summary(item) for item in cached_summaries]
    summaries = [s for s in summaries if s.run_id in ALLOWED_CACHE_RUNS]
    summaries.sort(key=lambda item: item.run_id)
    eligible_runs = tuple(s.run_id for s in summaries)
    checked = 0
    matched = 0
    calls = 0
    mismatches: List[HashMismatchReport] = []
    for summary in summaries:
        observed = observed_digests.get(summary.run_id)
        if observed is None:
            continue
        checked += 1
        if str(observed) == summary.digest:
            matched += 1
            continue
        called = False
        replacement = None
        call_run = None
        call_stage = None
        if summary.run_id == IMPORT_EXTRACTOR_RUN and calls < MAX_IMPORT_EXTRACTOR_CALLS and import_extractor is not None:
            replacement = import_extractor(summary.run_id, IMPORT_EXTRACTOR_STAGE)
            calls += 1
            called = True
            call_run = IMPORT_EXTRACTOR_RUN
            call_stage = IMPORT_EXTRACTOR_STAGE
        mismatches.append(
            HashMismatchReport(
                source_run_id=summary.run_id,
                expected_digest=summary.digest,
                observed_digest=str(observed),
                import_extractor_called=called,
                import_extractor_run=call_run,
                import_extractor_stage=call_stage,
                replacement_payload=dict(replacement) if replacement is not None else None,
            )
        )
    eligible_count = len(summaries)
    hash_checked_rate = 1.0 if eligible_count == 0 else checked / eligible_count
    hash_match_rate = 1.0 if checked == 0 else matched / checked
    return CacheAudit(
        eligible_runs=eligible_runs,
        checked_count=checked,
        matched_count=matched,
        mismatched_count=len(mismatches),
        import_extractor_calls=calls,
        hash_checked_rate=hash_checked_rate,
        hash_match_rate=hash_match_rate,
        mismatches=tuple(mismatches),
    )


def run_shadow_audit(
    cases: Iterable[Any],
    cached_summaries: Iterable[Any] = (),
    observed_digests: Optional[Mapping[int, str]] = None,
    import_extractor: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
    explanatory_flags: Iterable[str] = EXISTING_EXPLANATORY_FLAGS,
) -> ShadowAuditReport:
    observed = dict(observed_digests or {})
    cache_audit = audit_cache(cached_summaries, observed, import_extractor)
    raw_cases = [as_case(item) for item in cases]
    raw_keys = [(c.case_id, c.modality) for c in raw_cases]
    duplicate_count = len(raw_keys) - len(set(raw_keys))
    deduped = dedupe_case_modality(raw_cases)
    eligible = [c for c in deduped if c.run_id not in EXCLUDED_RUNS]
    positives = _stratified_take([c for c in eligible if c.has_flag(REVIEW_FLAG)], TARGET_POSITIVES)
    controls = _stratified_take([c for c in eligible if not c.has_flag(REVIEW_FLAG)], TARGET_CONTROLS)
    positive_count = len(positives)
    addressable = sum(1 for c in positives if c.is_addressable())
    drift_count = sum(1 for c in positives if c.has_flag(CENTROID_DRIFT_FLAG))
    topology_count = sum(1 for c in positives if c.has_flag(MYO_TOPOLOGY_FLAG))
    explained = frozenset(str(flag) for flag in explanatory_flags)
    unexplained_count = sum(1 for c in positives if not (c.flags & explained))
    denom = float(positive_count) if positive_count else 0.0
    addressability_rate = 0.0 if not denom else addressable / denom
    drift_rate = 0.0 if not denom else drift_count / denom
    topology_rate = 0.0 if not denom else topology_count / denom
    unexplained_rate = 0.0 if not denom else unexplained_count / denom
    notes: List[str] = []
    if positive_count and addressability_rate < ADDRESSABILITY_MIN:
        notes.append('addressability_below_threshold')
    if duplicate_count:
        notes.append('duplicate_case_modality_collapsed')
    if cache_audit.hash_checked_rate < 1.0:
        notes.append('cache_hash_check_incomplete')
    if cache_audit.mismatched_count:
        notes.append('cache_hash_mismatch_reported')
    if cache_audit.import_extractor_calls > MAX_IMPORT_EXTRACTOR_CALLS:
        notes.append('import_extractor_call_budget_exceeded')
    return ShadowAuditReport(
        positive_selected_count=positive_count,
        control_selected_count=len(controls),
        positive_addressable_count=addressable,
        positive_addressability_rate=addressability_rate,
        unique_case_modality_count=len(deduped),
        duplicate_case_modality_count=duplicate_count,
        centroid_drift_cooccurrence_count=drift_count,
        centroid_drift_cooccurrence_rate=drift_rate,
        myo_topology_cooccurrence_count=topology_count,
        myo_topology_cooccurrence_rate=topology_rate,
        unexplained_by_existing_flags_count=unexplained_count,
        unexplained_by_existing_flags_rate=unexplained_rate,
        cache_audit=cache_audit,
        selected_positives=tuple(positives),
        selected_controls=tuple(controls),
        notes=tuple(notes),
    )


def report_to_dict(report: ShadowAuditReport) -> Dict[str, Any]:
    return {
        'positive_selected_count': report.positive_selected_count,
        'control_selected_count': report.control_selected_count,
        'positive_addressable_count': report.positive_addressable_count,
        'positive_addressability_rate': report.positive_addressability_rate,
        'unique_case_modality_count': report.unique_case_modality_count,
        'duplicate_case_modality_count': report.duplicate_case_modality_count,
        'centroid_drift_cooccurrence_count': report.centroid_drift_cooccurrence_count,
        'centroid_drift_cooccurrence_rate': report.centroid_drift_cooccurrence_rate,
        'myo_topology_cooccurrence_count': report.myo_topology_cooccurrence_count,
        'myo_topology_cooccurrence_rate': report.myo_topology_cooccurrence_rate,
        'unexplained_by_existing_flags_count': report.unexplained_by_existing_flags_count,
        'unexplained_by_existing_flags_rate': report.unexplained_by_existing_flags_rate,
        'cache_eligible_runs': list(report.cache_audit.eligible_runs),
        'cache_hash_checked_rate': report.cache_audit.hash_checked_rate,
        'cache_hash_match_rate': report.cache_audit.hash_match_rate,
        'cache_mismatch_count': report.cache_audit.mismatched_count,
        'import_extractor_calls': report.cache_audit.import_extractor_calls,
        'notes': list(report.notes),
    }


def mismatch_reports(report: ShadowAuditReport) -> Tuple[HashMismatchReport, ...]:
    return report.cache_audit.mismatches
