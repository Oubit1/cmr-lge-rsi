"""Write-side hard gate for ``(case_id, modality)`` de-duplication.

Isolated, offline experiment module for the CMR/LGE QC orchestrator.  It adds a
single enforcement point on the *write* side of the pipeline:

* exactly one surviving row per ``(case_id, modality)`` key: duplicates are
  merged into the first-observed row instead of being appended, so the
  persisted duplicate count is always ``0``;
* the ``issue_type`` set and the slice indices of every contributing row are
  unioned, so problem recall cannot shrink during a merge;
* label fields are copied verbatim from the first-observed row, therefore a
  merge can never rewrite a ground-truth label (zero label modification);
* an extractor cache keyed by the payload hash: when the hash is already valid
  the cached extractor is reused and building a new one is refused;
* a terminal ``DONE`` state: after validation passed the gate refuses further
  writes/proposals instead of regenerating duplicate work.

Pure Python, deterministic, no network, no dataset and no third-party
dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    'CASE_ID_FIELD',
    'MODALITY_FIELD',
    'ISSUE_TYPE_FIELD',
    'SLICE_INDEX_FIELD',
    'LABEL_FIELDS',
    'CacheMissError',
    'CacheStats',
    'CaseModalityDedupError',
    'CaseModalityWriteGate',
    'ExtractorCache',
    'GateClosedError',
    'GateStateError',
    'MergeResult',
    'ValidationFailedError',
    'ValidationReport',
    'case_modality_key',
    'collect_label_signatures',
    'count_duplicate_rows',
    'label_signature',
    'merge_case_modality_rows',
    'normalize_issue_types',
    'normalize_slice_indices',
    'payload_hash',
]

CASE_ID_FIELD = 'case_id'
MODALITY_FIELD = 'modality'
ISSUE_TYPE_FIELD = 'issue_type'
SLICE_INDEX_FIELD = 'slice_indices'
LABEL_FIELDS: Tuple[str, ...] = ('label', 'labels', 'gt_label', 'ground_truth')
MAX_RANGE_WIDTH = 4096


class CaseModalityDedupError(RuntimeError):
    """Base class for every hard-gate violation."""


class GateClosedError(CaseModalityDedupError):
    """Raised when a write/proposal is attempted after the gate is DONE."""


class GateStateError(CaseModalityDedupError):
    """Raised when the gate is used out of order."""


class ValidationFailedError(CaseModalityDedupError):
    """Raised when ``mark_done`` is called on a failing validation report."""


class CacheMissError(CaseModalityDedupError):
    """Raised when an extractor would have to be built for an already-valid hash."""


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------


def _as_iterable(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.keys())
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def normalize_issue_types(value: Any) -> Tuple[str, ...]:
    """Return the de-duplicated, sorted ``issue_type`` set encoded by ``value``."""
    cleaned = {str(item).strip() for item in _as_iterable(value)}
    return tuple(sorted(item for item in cleaned if item))


def normalize_slice_indices(
    value: Any, *, max_range_width: int = MAX_RANGE_WIDTH
) -> Tuple[int, ...]:
    """Return sorted, de-duplicated slice indices; ``'3-6'`` expands to 3..6."""
    indices = set()
    for item in _as_iterable(value):
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            indices.add(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        if '-' in text:
            low_text, _, high_text = text.partition('-')
            try:
                low, high = int(low_text), int(high_text)
            except ValueError:
                continue
            if high < low:
                low, high = high, low
            if high - low + 1 > max_range_width:
                continue
            indices.update(range(low, high + 1))
            continue
        try:
            indices.add(int(text))
        except ValueError:
            continue
    return tuple(sorted(indices))


def case_modality_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Return the canonical ``(case_id, modality)`` key of ``row``.

    Rows without a usable key are rejected outright: a write-side gate must
    never guess a key, otherwise duplicates can slip through.
    """
    try:
        raw_case = row[CASE_ID_FIELD]
        raw_modality = row[MODALITY_FIELD]
    except (TypeError, KeyError) as exc:
        raise ValueError(
            'row must carry {0} and {1}'.format(CASE_ID_FIELD, MODALITY_FIELD)
        ) from exc
    if raw_case is None or raw_modality is None:
        raise ValueError(
            'row must carry non-null {0} and {1}'.format(CASE_ID_FIELD, MODALITY_FIELD)
        )
    case_id = str(raw_case).strip()
    modality = str(raw_modality).strip()
    if not case_id or not modality:
        raise ValueError(
            'row must carry non-empty {0} and {1}'.format(CASE_ID_FIELD, MODALITY_FIELD)
        )
    return case_id, modality


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(',', ':'))


def label_signature(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Order-independent canonical view of every label field present in ``row``."""
    return tuple(
        sorted((name, _canonical(row[name])) for name in LABEL_FIELDS if name in row)
    )


def collect_label_signatures(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]]:
    """Snapshot label signatures per key, taken *before* any merge happens."""
    signatures: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {}
    for row in rows:
        key = case_modality_key(row)
        signatures.setdefault(key, label_signature(row))
    return signatures


def count_duplicate_rows(rows: Iterable[Mapping[str, Any]]) -> int:
    """Count rows whose ``(case_id, modality)`` key was already seen."""
    seen = set()
    duplicates = 0
    for row in rows:
        key = case_modality_key(row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeResult:
    """Outcome of one write-side merge: duplicate-free rows plus audit data."""

    rows: Tuple[Dict[str, Any], ...]
    collapsed_existing: int = 0
    collapsed_incoming: int = 0
    label_conflicts: Tuple[Tuple[str, str], ...] = ()
    issue_types: Dict[Tuple[str, str], Tuple[str, ...]] = field(default_factory=dict)
    slice_indices: Dict[Tuple[str, str], Tuple[int, ...]] = field(default_factory=dict)
    labels: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = field(default_factory=dict)

    @property
    def duplicate_rows(self) -> int:
        return count_duplicate_rows(self.rows)

    @property
    def keys(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(case_modality_key(row) for row in self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rows': [dict(row) for row in self.rows],
            'collapsed_existing': self.collapsed_existing,
            'collapsed_incoming': self.collapsed_incoming,
            'label_conflicts': [list(key) for key in self.label_conflicts],
            'issue_types': {
                '{0}|{1}'.format(key[0], key[1]): list(value)
                for key, value in self.issue_types.items()
            },
        }


def merge_case_modality_rows(
    existing_rows: Sequence[Mapping[str, Any]] = (),
    incoming_rows: Sequence[Mapping[str, Any]] = (),
) -> MergeResult:
    """Collapse ``(case_id, modality)`` duplicates while preserving recall.

    The first observed row of a key becomes the survivor; every later row with
    the same key is merged into it (never appended).  ``issue_type`` and
    ``slice_indices`` are unioned, label fields are left untouched.
    """
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    issue_types: Dict[Tuple[str, str], set] = {}
    slice_indices: Dict[Tuple[str, str], set] = {}
    labels: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {}
    label_conflicts: List[Tuple[str, str]] = []
    collapsed_existing = 0
    collapsed_incoming = 0

    def ingest(row: Mapping[str, Any], is_incoming: bool) -> None:
        nonlocal collapsed_existing, collapsed_incoming
        key = case_modality_key(row)
        record = dict(row)
        record[CASE_ID_FIELD] = key[0]
        record[MODALITY_FIELD] = key[1]
        signature = label_signature(record)
        found_issue_types = set(normalize_issue_types(record.get(ISSUE_TYPE_FIELD)))
        found_slices = set(normalize_slice_indices(record.get(SLICE_INDEX_FIELD)))

        if key not in rows:
            rows[key] = record
            labels[key] = signature
            issue_types[key] = found_issue_types
            slice_indices[key] = found_slices
            return

        if is_incoming:
            collapsed_incoming += 1
        else:
            collapsed_existing += 1
        if signature != labels[key]:
            label_conflicts.append(key)
        issue_types[key].update(found_issue_types)
        slice_indices[key].update(found_slices)

    for row in existing_rows:
        ingest(row, False)
    for row in incoming_rows:
        ingest(row, True)

    merged_rows: List[Dict[str, Any]] = []
    for key in sorted(rows):
        record = dict(rows[key])
        record[ISSUE_TYPE_FIELD] = list(sorted(issue_types[key]))
        record[SLICE_INDEX_FIELD] = list(sorted(slice_indices[key]))
        merged_rows.append(record)

    return MergeResult(
        rows=tuple(merged_rows),
        collapsed_existing=collapsed_existing,
        collapsed_incoming=collapsed_incoming,
        label_conflicts=tuple(label_conflicts),
        issue_types={key: tuple(sorted(value)) for key, value in issue_types.items()},
        slice_indices={key: tuple(sorted(value)) for key, value in slice_indices.items()},
        labels=dict(labels),
    )


# ---------------------------------------------------------------------------
# extractor cache
# ---------------------------------------------------------------------------


def payload_hash(payload: Any, *, salt: Optional[str] = None) -> str:
    """Stable sha256 of a JSON-canonicalised payload."""
    body = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':'))
    if salt:
        body = '{0}|{1}'.format(salt, body)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    created: int = 0
    reused: int = 0
    violations: int = 0

    @property
    def hits(self) -> int:
        return self.reused

    @property
    def total(self) -> int:
        return self.created + self.reused

    @property
    def reuse_rate(self) -> float:
        return self.reused / self.total if self.total else 0.0


class ExtractorCache:
    """Payload-hash keyed extractor cache.

    ``resolve(payload, hash_is_valid=True)`` encodes the hard part of the
    contract: the caller asserts the payload hash is valid, therefore an
    extractor must already be cached.  If it is not, the gate refuses to build
    a new one and raises :class:`CacheMissError` (counted as a violation).
    """

    def __init__(self, factory: Optional[Callable[[Any], Any]] = None) -> None:
        self._factory = factory
        self._entries: Dict[str, Any] = {}
        self._created = 0
        self._reused = 0
        self._violations = 0

    def resolve(
        self,
        payload: Any,
        *,
        factory: Optional[Callable[[Any], Any]] = None,
        hash_is_valid: bool = False,
    ) -> Tuple[Any, bool]:
        """Return ``(extractor, reused)``; never rebuild a cached hash."""
        digest = payload_hash(payload)
        if digest in self._entries:
            self._reused += 1
            return self._entries[digest], True
        if hash_is_valid:
            self._violations += 1
            raise CacheMissError(
                'valid hash {0} is not cached; refusing to build a new extractor'.format(
                    digest[:12]
                )
            )
        builder = factory or self._factory
        if builder is None:
            self._violations += 1
            raise CacheMissError('no extractor factory available for a cache miss')
        extractor = builder(payload)
        if extractor is None:
            self._violations += 1
            raise CacheMissError('extractor factory returned None')
        self._entries[digest] = extractor
        self._created += 1
        return extractor, False

    def contains(self, payload: Any) -> bool:
        return payload_hash(payload) in self._entries

    def prime(self, payload: Any, extractor: Any) -> str:
        digest = payload_hash(payload)
        self._entries[digest] = extractor
        return digest

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            created=self._created, reused=self._reused, violations=self._violations
        )

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# validation + terminal DONE state
# ---------------------------------------------------------------------------


def _normalise_signature(signature: Any) -> Tuple[Tuple[str, str], ...]:
    if isinstance(signature, Mapping):
        return tuple(sorted((str(key), _canonical(value)) for key, value in signature.items()))
    return tuple((str(pair[0]), str(pair[1])) for pair in signature)


def _label_expectations(
    labels_before: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]]:
    if not labels_before:
        return {}
    expected: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {}
    for key, signature in labels_before.items():
        if isinstance(key, (tuple, list)) and len(key) == 2:
            expected[(str(key[0]), str(key[1]))] = _normalise_signature(signature)
    return expected


@dataclass(frozen=True)
class ValidationReport:
    duplicate_rows: int = 0
    missing_issue_types: Tuple[str, ...] = ()
    labels_modified: int = 0
    cache_violations: int = 0
    extractors_created: int = 0
    cache_reuse_rate: float = 0.0
    collapsed_existing: int = 0
    collapsed_incoming: int = 0

    @property
    def issue_types_complete(self) -> bool:
        return not self.missing_issue_types

    @property
    def passed(self) -> bool:
        return (
            self.duplicate_rows == 0
            and self.issue_types_complete
            and self.labels_modified == 0
            and self.cache_violations == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'duplicate_rows': self.duplicate_rows,
            'missing_issue_types': list(self.missing_issue_types),
            'labels_modified': self.labels_modified,
            'cache_violations': self.cache_violations,
            'extractors_created': self.extractors_created,
            'cache_reuse_rate': self.cache_reuse_rate,
            'issue_types_complete': self.issue_types_complete,
            'passed': self.passed,
        }


class CaseModalityWriteGate:
    """Write-side gate: dedup + extractor cache + terminal DONE state.

    Lifecycle: ``write`` -> ``validate`` -> ``mark_done``.  Once ``DONE`` the
    gate refuses further writes/proposals (:class:`GateClosedError`) so the same
    work is never proposed again.
    """

    STATE_PENDING = 'PENDING'
    STATE_MERGED = 'MERGED'
    STATE_DONE = 'DONE'

    def __init__(
        self,
        *,
        name: str = 'case_modality_write_gate',
        extractor_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.name = name
        self.state = self.STATE_PENDING
        self.cache = ExtractorCache(extractor_factory)
        self.rejected_proposals = 0
        self.last_merge: Optional[MergeResult] = None
        self.last_report: Optional[ValidationReport] = None

    @property
    def is_done(self) -> bool:
        return self.state == self.STATE_DONE

    def _ensure_open(self) -> None:
        if self.is_done:
            self.rejected_proposals += 1
            raise GateClosedError(
                '{0} is DONE; further writes are refused'.format(self.name)
            )

    def write(
        self,
        existing_rows: Sequence[Mapping[str, Any]] = (),
        incoming_rows: Sequence[Mapping[str, Any]] = (),
    ) -> MergeResult:
        self._ensure_open()
        merged = merge_case_modality_rows(existing_rows, incoming_rows)
        self.last_merge = merged
        self.state = self.STATE_MERGED
        return merged

    def resolve_extractor(
        self,
        payload: Any,
        *,
        factory: Optional[Callable[[Any], Any]] = None,
        hash_is_valid: bool = False,
    ) -> Tuple[Any, bool]:
        self._ensure_open()
        return self.cache.resolve(payload, factory=factory, hash_is_valid=hash_is_valid)

    def validate(
        self,
        merge_result: Optional[MergeResult] = None,
        *,
        labels_before: Optional[Mapping[Any, Any]] = None,
        cache_stats: Optional[CacheStats] = None,
    ) -> ValidationReport:
        merged = merge_result if merge_result is not None else self.last_merge
        if merged is None:
            raise GateStateError('no merged write available to validate')

        duplicate_rows = count_duplicate_rows(merged.rows)
        missing: List[str] = []
        labels_modified = 0
        expected_labels = _label_expectations(labels_before)

        for row in merged.rows:
            key = case_modality_key(row)
            expected_types = set(merged.issue_types.get(key, ()))
            actual_types = set(normalize_issue_types(row.get(ISSUE_TYPE_FIELD)))
            missing.extend(
                sorted(
                    '{0}|{1}:{2}'.format(key[0], key[1], item)
                    for item in expected_types - actual_types
                )
            )
            signature = label_signature(row)
            baseline = merged.labels.get(key)
            if baseline is not None and signature != baseline:
                labels_modified += 1
            if key in expected_labels and signature != expected_labels[key]:
                labels_modified += 1

        stats = cache_stats if cache_stats is not None else self.cache.stats
        report = ValidationReport(
            duplicate_rows=duplicate_rows,
            missing_issue_types=tuple(missing),
            labels_modified=labels_modified,
            cache_violations=stats.violations,
            extractors_created=stats.created,
            cache_reuse_rate=stats.reuse_rate,
            collapsed_existing=merged.collapsed_existing,
            collapsed_incoming=merged.collapsed_incoming,
        )
        self.last_report = report
        return report

    def mark_done(self, report: Optional[ValidationReport] = None) -> ValidationReport:
        final = report if report is not None else self.last_report
        if final is None:
            raise GateStateError('validate() must run before mark_done()')
        if not final.passed:
            raise ValidationFailedError(
                '{0} validation failed: {1}'.format(self.name, final.to_dict())
            )
        self.last_report = final
        self.state = self.STATE_DONE
        return final
