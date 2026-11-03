'''Bounded, read-only shadow audit for the ``cmr_myo_touches_image_border`` finding.

The audit is deliberately small and side-effect free:

* **Read-only.** Nothing in this module writes production labels, thresholds or
  rules - write_counts() is a constant zero map and no other code path opens a
  file for writing. Caller supplied candidate payloads are never mutated (they
  are copied before use).
* **Bounded.** At most POSITIVE_BUDGET (50) positive cases and CONTROL_BUDGET
  (10) cmr controls are audited, selected by a stable case_id sort so two runs
  over the same input produce identical reports.
* **Offline.** No network access, no dataset download; every payload comes from
  the caller (candidate index, frozen run caches) or an explicitly injected
  extractor callable.
* **Cache reuse with hash guard.** Payloads are resolved from the frozen
  run-4 / run-5 / run-9 / run-13 .. run-18 caches first. A cached payload whose
  stored payload_hash does not match its recomputed hash - or a case that is
  absent from every cache - is not trusted: it is re-extracted through the
  run-9 B1 extractor only, and both the mismatch and the fallback are reported.

Positive cases must carry the ``cmr_myo_touches_image_border`` finding; controls
are ``cmr`` cases that do not carry it. Geometry may arrive inline on the
candidate, from a cache hit, or from the fallback extractor.
'''

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    'FINDING_KEY',
    'POSITIVE_BUDGET',
    'CONTROL_BUDGET',
    'CACHE_RUN_IDS',
    'FALLBACK_SOURCE',
    'INLINE_SOURCE',
    'CO_OCCURRENCE_KEYS',
    'WRITE_TARGETS',
    'TRUNCATION_FOV',
    'TRUNCATION_CROP',
    'TRUNCATION_NONE',
    'ShadowAuditError',
    'MissingExtractorError',
    'MalformedCasePayload',
    'CaseRecord',
    'CacheResolver',
    'BoundedShadowAuditReport',
    'payload_hash',
    'with_payload_hash',
    'record_from_payload',
    'touched_sides',
    'classify_truncation',
    'select_bounded_sample',
    'run_bounded_shadow_audit',
    'write_counts',
]

FINDING_KEY = 'cmr_myo_touches_image_border'

POSITIVE_BUDGET = 50
CONTROL_BUDGET = 10

# Frozen caches that may be reused; order defines resolution precedence.
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

# Hash mismatches / cache misses may only be re-extracted through run-9 B1.
FALLBACK_RUN_ID = 'run-9'
FALLBACK_EXTRACTOR_ID = 'B1'
FALLBACK_SOURCE = FALLBACK_RUN_ID + '/' + FALLBACK_EXTRACTOR_ID
INLINE_SOURCE = 'candidate_inline'

HASH_FIELD = 'payload_hash'
GEOMETRY_KEYS = ('rows', 'cols', 'mask_rows', 'mask_cols')

SIDES = ('left', 'right', 'top', 'bottom')
TRUNCATION_FOV = 'field_of_view'
TRUNCATION_CROP = 'crop'
TRUNCATION_NONE = 'not_touching'

# Co-occurring findings whose overlap with the border-touch finding is reported.
CO_OCCURRENCE_KEYS = ('small_area', 'excess_components', 'empty')

# Production write targets; the audit must never touch any of them.
WRITE_TARGETS = ('production_labels', 'thresholds', 'rules')


class ShadowAuditError(RuntimeError):
    '''Base error for the bounded border-touch shadow audit.'''


class MissingExtractorError(ShadowAuditError):
    '''Raised when a case needs the run-9 B1 fallback but none was injected.'''


class MalformedCasePayload(ShadowAuditError):
    '''Raised when a payload does not carry usable mask geometry.'''


def payload_hash(payload: Mapping[str, Any]) -> str:
    '''Stable sha256 over a payload body, excluding the stored hash field.'''
    body = {k: v for k, v in payload.items() if k != HASH_FIELD}
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def with_payload_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    '''Return a copy of ``payload`` carrying its freshly computed hash.'''
    body = {k: v for k, v in payload.items() if k != HASH_FIELD}
    return {**body, HASH_FIELD: payload_hash(body)}


def _findings_of(payload: Mapping[str, Any]) -> frozenset[str]:
    raw = payload.get('findings') or ()
    if isinstance(raw, str):
        return frozenset([raw])
    try:
        return frozenset(str(v) for v in raw)
    except TypeError:
        return frozenset([str(raw)])


def _modality_of(payload: Mapping[str, Any]) -> str:
    return str(payload.get('modality') or '').strip().lower()


@dataclass(frozen=True)
class CaseRecord:
    '''Verified geometry of one audited case.'''

    case_id: str
    modality: str
    findings: frozenset[str]
    slice_index: int
    n_slices: int
    rows: int
    cols: int
    mask_rows: tuple[int, int]
    mask_cols: tuple[int, int]
    source: str = INLINE_SOURCE


def record_from_payload(
    case_id: str,
    payload: Mapping[str, Any],
    *,
    modality: Any = None,
    findings: Any = None,
    source: str = INLINE_SOURCE,
) -> CaseRecord:
    '''Build a CaseRecord from a payload, validating its mask geometry.'''
    if not isinstance(payload, Mapping):
        raise MalformedCasePayload('case %r: payload is not a mapping' % (case_id,))
    try:
        rows = int(payload['rows'])
        cols = int(payload['cols'])
        mask_rows = tuple(int(v) for v in payload['mask_rows'])
        mask_cols = tuple(int(v) for v in payload['mask_cols'])
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedCasePayload(
            'case %r: incomplete geometry (%s)' % (case_id, exc)
        ) from exc
    if len(mask_rows) != 2 or len(mask_cols) != 2:
        raise MalformedCasePayload('case %r: mask bounds must be 2-tuples' % (case_id,))
    if mask_rows[0] > mask_rows[1] or mask_cols[0] > mask_cols[1]:
        raise MalformedCasePayload('case %r: inverted mask bounds' % (case_id,))
    if rows <= 0 or cols <= 0:
        raise MalformedCasePayload('case %r: non-positive matrix size' % (case_id,))

    n_slices = max(1, int(payload.get('n_slices', 1) or 1))
    slice_index = int(payload.get('slice_index', 0) or 0)

    resolved_modality = str(
        modality if modality is not None else payload.get('modality', 'unknown')
    )
    if findings is None:
        resolved_findings = _findings_of(payload)
    else:
        resolved_findings = _findings_of({'findings': findings})

    return CaseRecord(
        case_id=str(case_id),
        modality=resolved_modality,
        findings=resolved_findings,
        slice_index=slice_index,
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        mask_rows=(mask_rows[0], mask_rows[1]),
        mask_cols=(mask_cols[0], mask_cols[1]),
        source=source,
    )


def touched_sides(record: CaseRecord) -> tuple[str, ...]:
    '''Return the image-border sides the mask touches, in SIDES order.'''
    sides: list[str] = []
    if record.mask_cols[0] <= 0:
        sides.append('left')
    if record.mask_cols[1] >= record.cols - 1:
        sides.append('right')
    if record.mask_rows[0] <= 0:
        sides.append('top')
    if record.mask_rows[1] >= record.rows - 1:
        sides.append('bottom')
    return tuple(sides)


def classify_truncation(record: CaseRecord, sides: Sequence[str] | None = None) -> str:
    '''Classify a border touch as field-of-view coverage vs in-plane crop.

    Deterministic, geometry-only heuristic:

    * a touch on the first or last slice of the cine stack is a through-plane
      coverage boundary -> TRUNCATION_FOV;
    * a mask that spans a whole in-plane matrix dimension on an interior slice is
      a reconstruction / reformat crop boundary -> TRUNCATION_FOV;
    * any other interior touch -> TRUNCATION_CROP;
    * a mask that touches nothing -> TRUNCATION_NONE.
    '''
    sides = touched_sides(record) if sides is None else tuple(sides)
    if not sides:
        return TRUNCATION_NONE
    if record.slice_index <= 0 or record.slice_index >= record.n_slices - 1:
        return TRUNCATION_FOV
    spans_rows = record.mask_rows[0] <= 0 and record.mask_rows[1] >= record.rows - 1
    spans_cols = record.mask_cols[0] <= 0 and record.mask_cols[1] >= record.cols - 1
    if spans_rows or spans_cols:
        return TRUNCATION_FOV
    return TRUNCATION_CROP


def write_counts() -> dict[str, int]:
    '''Production write counters; always zero because the audit never writes.'''
    return {target: 0 for target in WRITE_TARGETS}


class CacheResolver:
    '''Resolve case payloads from frozen run caches with hash verification.

    Resolution order per case:

    1. first hit across CACHE_RUN_IDS (then any extra run id, sorted); the stored
       payload_hash must match the recomputed hash, otherwise the cache entry is
       discarded and recorded as a hash mismatch;
    2. the inline geometry shipped by the candidate index;
    3. the run-9 B1 fallback extractor (required, never silent).
    '''

    def __init__(
        self,
        caches: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        extractor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.caches = {
            str(run_id): dict(payloads or {})
            for run_id, payloads in (caches or {}).items()
        }
        self.extractor = extractor

        index: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for run_id in CACHE_RUN_IDS:
            for case_id, payload in self.caches.get(run_id, {}).items():
                index.setdefault(str(case_id), (run_id, payload))
        for run_id in sorted(self.caches):
            if run_id in CACHE_RUN_IDS:
                continue
            for case_id, payload in self.caches[run_id].items():
                index.setdefault(str(case_id), (run_id, payload))
        self._index = index

        self.reuse_counts: dict[str, int] = {run_id: 0 for run_id in CACHE_RUN_IDS}
        self.cache_miss_case_ids: list[str] = []
        self.hash_mismatch_case_ids: list[str] = []
        self.fallback_case_ids: list[str] = []
        self.fallback_reasons: dict[str, str] = {}
        self.inline_case_ids: list[str] = []
        self.unhashed_reuse_count = 0

    def resolve(
        self, case_id: str, inline: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], str]:
        '''Return (payload, source) for case_id.'''
        hit = self._index.get(case_id)
        if hit is not None:
            run_id, cached = hit
            expected = cached.get(HASH_FIELD)
            if expected is not None and expected != payload_hash(cached):
                self.hash_mismatch_case_ids.append(case_id)
                self.fallback_reasons[case_id] = 'hash_mismatch:' + run_id
                return self._extract(case_id), FALLBACK_SOURCE
            self.reuse_counts[run_id] = self.reuse_counts.get(run_id, 0) + 1
            if expected is None:
                self.unhashed_reuse_count += 1
            return {k: v for k, v in cached.items() if k != HASH_FIELD}, run_id

        if inline is not None:
            self.inline_case_ids.append(case_id)
            return dict(inline), INLINE_SOURCE

        self.cache_miss_case_ids.append(case_id)
        self.fallback_reasons[case_id] = 'cache_miss'
        return self._extract(case_id), FALLBACK_SOURCE

    def _extract(self, case_id: str) -> dict[str, Any]:
        if self.extractor is None:
            raise MissingExtractorError(
                'case %r needs the %s re-extraction but no extractor callable was '
                'injected' % (case_id, FALLBACK_SOURCE)
            )
        payload = dict(self.extractor(case_id))
        if not all(key in payload for key in ('mask_rows', 'mask_cols')):
            raise MalformedCasePayload(
                'case %r: %s re-extraction returned no mask bounds'
                % (case_id, FALLBACK_SOURCE)
            )
        if case_id not in self.fallback_case_ids:
            self.fallback_case_ids.append(case_id)
        return payload


def select_bounded_sample(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    '''Return (positives, controls) capped at the audit budgets.'''
    indexed: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or 'case_id' not in candidate:
            raise ShadowAuditError('candidate without case_id: %r' % (candidate,))
        indexed.append(candidate)

    positives = sorted(
        (c for c in indexed if FINDING_KEY in _findings_of(c)),
        key=lambda c: str(c['case_id']),
    )[:POSITIVE_BUDGET]
    controls = sorted(
        (
            c
            for c in indexed
            if _modality_of(c) == 'cmr' and FINDING_KEY not in _findings_of(c)
        ),
        key=lambda c: str(c['case_id']),
    )[:CONTROL_BUDGET]
    return positives, controls


def _materialise_case(
    candidate: Mapping[str, Any], resolver: CacheResolver
) -> tuple[CaseRecord, str]:
    case_id = str(candidate['case_id'])
    geometry = {key: candidate[key] for key in GEOMETRY_KEYS if key in candidate}
    for optional in ('slice_index', 'n_slices'):
        if optional in candidate:
            geometry[optional] = candidate[optional]
    inline = geometry if all(key in geometry for key in GEOMETRY_KEYS) else None

    payload, source = resolver.resolve(case_id, inline=inline)
    record = record_from_payload(
        case_id,
        payload,
        modality=candidate.get('modality', payload.get('modality')),
        findings=candidate.get('findings', payload.get('findings')),
        source=source,
    )
    return record, source


def _co_occurrence(
    positives: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    n_pos = len(positives)
    n_ctl = len(controls)
    out: dict[str, dict[str, float]] = {}
    for key in CO_OCCURRENCE_KEYS:
        p = sum(1 for row in positives if key in row['co_occurring'])
        c = sum(1 for row in controls if key in row['co_occurring'])
        out[key] = {
            'positives': p,
            'controls': c,
            'positive_rate': (p / n_pos) if n_pos else 0.0,
            'control_rate': (c / n_ctl) if n_ctl else 0.0,
        }
    return out


@dataclass
class BoundedShadowAuditReport:
    '''Result of one bounded, read-only border-touch shadow audit run.'''

    finding_key: str
    positives: list[dict[str, Any]]
    controls: list[dict[str, Any]]
    control_false_positive_case_ids: list[str]
    control_false_positive_rate: float
    co_occurrence: dict[str, dict[str, float]]
    cache_reuse_counts: dict[str, int]
    cache_reuse_rate: float
    cache_miss_case_ids: list[str]
    hash_mismatch_case_ids: list[str]
    fallback_case_ids: list[str]
    fallback_reasons: dict[str, str]
    fallback_extractor: str
    inline_case_ids: list[str]
    unhashed_reuse_count: int
    positive_label_without_geometry_touch: list[str]
    completion_rate: float
    bounded: dict[str, int]
    write_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        '''JSON-serialisable view of the report.'''
        return asdict(self)


def run_bounded_shadow_audit(
    candidates: Sequence[Mapping[str, Any]],
    *,
    caches: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    extractor: Callable[[str], Mapping[str, Any]] | None = None,
) -> BoundedShadowAuditReport:
    '''Run the bounded, read-only border-touch shadow audit.

    ``caches`` maps run_id -> {case_id: payload}; ``extractor`` is the only
    permitted re-extraction path for hash mismatches and cache misses (run-9 B1).
    Neither the candidates nor the cache payloads are mutated.
    '''
    candidate_list = list(candidates)
    positives_meta, controls_meta = select_bounded_sample(candidate_list)

    available_positives = sum(1 for c in candidate_list if FINDING_KEY in _findings_of(c))
    available_controls = sum(
        1
        for c in candidate_list
        if _modality_of(c) == 'cmr' and FINDING_KEY not in _findings_of(c)
    )

    resolver = CacheResolver(caches=caches, extractor=extractor)

    positives: list[dict[str, Any]] = []
    for candidate in positives_meta:
        record, source = _materialise_case(candidate, resolver)
        sides = touched_sides(record)
        positives.append(
            {
                'case_id': record.case_id,
                'sides': list(sides),
                'truncation': classify_truncation(record, sides),
                'slice_index': record.slice_index,
                'n_slices': record.n_slices,
                'source': source,
                'co_occurring': sorted(k for k in CO_OCCURRENCE_KEYS if k in record.findings),
            }
        )

    controls: list[dict[str, Any]] = []
    for candidate in controls_meta:
        record, source = _materialise_case(candidate, resolver)
        sides = touched_sides(record)
        controls.append(
            {
                'case_id': record.case_id,
                'sides': list(sides),
                'truncation': classify_truncation(record, sides),
                'slice_index': record.slice_index,
                'n_slices': record.n_slices,
                'source': source,
                'co_occurring': sorted(k for k in CO_OCCURRENCE_KEYS if k in record.findings),
            }
        )

    control_false_positives = [row['case_id'] for row in controls if row['sides']]
    audited = len(positives) + len(controls)
    selected = len(positives_meta) + len(controls_meta)
    reused = sum(resolver.reuse_counts.values())

    return BoundedShadowAuditReport(
        finding_key=FINDING_KEY,
        positives=positives,
        controls=controls,
        control_false_positive_case_ids=control_false_positives,
        control_false_positive_rate=(
            len(control_false_positives) / len(controls) if controls else 0.0
        ),
        co_occurrence=_co_occurrence(positives, controls),
        cache_reuse_counts=dict(resolver.reuse_counts),
        cache_reuse_rate=(reused / audited) if audited else 0.0,
        cache_miss_case_ids=list(resolver.cache_miss_case_ids),
        hash_mismatch_case_ids=list(resolver.hash_mismatch_case_ids),
        fallback_case_ids=list(resolver.fallback_case_ids),
        fallback_reasons=dict(resolver.fallback_reasons),
        fallback_extractor=FALLBACK_SOURCE,
        inline_case_ids=list(resolver.inline_case_ids),
        unhashed_reuse_count=resolver.unhashed_reuse_count,
        positive_label_without_geometry_touch=[
            row['case_id'] for row in positives if not row['sides']
        ],
        completion_rate=(audited / selected) if selected else 1.0,
        bounded={
            'positive_budget': POSITIVE_BUDGET,
            'positives_available': available_positives,
            'positives_audited': len(positives),
            'control_budget': CONTROL_BUDGET,
            'controls_available': available_controls,
            'controls_audited': len(controls),
        },
        write_counts=write_counts(),
    )
