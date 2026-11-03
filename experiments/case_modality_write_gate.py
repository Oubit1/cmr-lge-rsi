"""Write-enforcing ``(case_id, modality)`` de-duplication gate.

Upgrades the previously advisory ``(case_id, modality)`` de-duplication proposal
into an enforced write gate:

1. A row whose ``(case_id, modality)`` key already exists is rejected outright
   and the raw row is kept verbatim in the audit log (one entry per rejected
   write attempt); it is never stored as a second record.
2. ``issue_type`` sets from every row sharing a key are merged (union) and the
   per-issue slice indices are preserved, sorted and de-duplicated, so the
   merged record can never lose recall relative to any single incoming row and
   the review queue never grows because of a duplicate row.
3. While the cached hash for a key is still valid no new extractor is
   constructed. Extractor construction happens at most once per key, and only
   when the cached hash is missing or mismatched.

Only the standard library is used, so the module runs fully offline.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    'CaseModalityWriteGate',
    'MergedRecord',
    'WriteDecision',
    'cache_hash_is_valid',
    'case_modality_key',
    'issue_index_map',
    'make_key',
    'merge_issue_index_maps',
    'normalize_issue_types',
    'normalize_slice_indices',
    'run_write_gate',
]

Row = Dict[str, Any]
Key = Tuple[str, str]
IssueIndexMap = Dict[str, Tuple[int, ...]]

CASE_ID_FIELDS: Tuple[str, ...] = ('case_id', 'case', 'caseId', 'caseID')
MODALITY_FIELDS: Tuple[str, ...] = ('modality', 'mod', 'modality_name')


def _first_present(row, fields):
    '''Return the first non-empty value of ``fields`` present in ``row``.'''
    for name in fields:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def make_key(case_id, modality) -> Key:
    '''Normalise a ``(case_id, modality)`` pair into the canonical gate key.'''
    if case_id is None or modality is None:
        raise KeyError('(case_id, modality) key requires both fields')
    case = str(case_id).strip()
    mod = str(modality).strip().lower()
    if not case or not mod:
        raise KeyError('(case_id, modality) key fields must be non-empty')
    return (case, mod)


def case_modality_key(row) -> Key:
    '''Extract and normalise the de-duplication key of ``row``.'''
    return make_key(_first_present(row, CASE_ID_FIELDS), _first_present(row, MODALITY_FIELDS))


def normalize_issue_types(value) -> Tuple[str, ...]:
    '''Return issue type labels de-duplicated with input order preserved.'''
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    seen = set()
    out: List[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def normalize_slice_indices(value) -> Tuple[int, ...]:
    '''Return slice indices as a sorted, de-duplicated tuple of ints.'''
    if value is None or isinstance(value, bool):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        parts = [piece.strip() for piece in stripped.split(',')] if ',' in stripped else [stripped]
    elif isinstance(value, (int, float)):
        parts = [value]
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    out = set()
    for item in parts:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str) and ',' in item:
            out.update(normalize_slice_indices(item))
            continue
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(out))


def issue_index_map(row) -> IssueIndexMap:
    '''Extract ``{issue_type: (slice_index, ...)}`` from a review row.

    Several legacy shapes are accepted and merged when present:

    * ``issue_slice_indices`` as an issue-to-indices mapping
    * ``issue_slice_indices`` as a list of per-issue dicts
    * ``issues`` as a list of per-issue dicts or plain issue-type strings
    * flat ``issue_types`` / ``issue_type`` plus ``slice_indices`` / ``slice_index``

    Malformed entries are ignored rather than failing the write; no issue that is
    present in the row is ever dropped from the returned map.
    '''
    mapping: Dict[str, set] = {}

    def add(issue_value, index_value) -> None:
        names = normalize_issue_types(issue_value)
        if not names:
            return
        indices = normalize_slice_indices(index_value)
        for name in names:
            mapping.setdefault(name, set()).update(indices)

    container = row.get('issue_slice_indices')
    if isinstance(container, MappingABC):
        for issue, indices in container.items():
            add(issue, indices)
    elif isinstance(container, (list, tuple)):
        for entry in container:
            if isinstance(entry, MappingABC):
                add(
                    entry.get('issue_type', entry.get('issue')),
                    entry.get('slice_indices', entry.get('indices')),
                )

    issues = row.get('issues')
    if isinstance(issues, (list, tuple)):
        for entry in issues:
            if isinstance(entry, MappingABC):
                add(
                    entry.get('issue_type', entry.get('issue')),
                    entry.get('slice_indices', entry.get('indices', entry.get('slices'))),
                )
            else:
                add(entry, None)

    flat_names = normalize_issue_types(row.get('issue_types')) or normalize_issue_types(
        row.get('issue_type')
    )
    if flat_names:
        flat_indices = row.get('slice_indices')
        if flat_indices is None:
            flat_indices = row.get('slice_index')
        if flat_indices is None:
            flat_indices = row.get('slices')
        for name in flat_names:
            add(name, flat_indices)

    return {name: tuple(sorted(mapping[name])) for name in sorted(mapping)}


def merge_issue_index_maps(base, incoming) -> IssueIndexMap:
    '''Union two issue maps, preserving every slice index from both sides.'''
    merged: Dict[str, set] = {}
    for source in (base, incoming):
        for name, indices in (source or {}).items():
            key_name = str(name).strip()
            if not key_name:
                continue
            merged.setdefault(key_name, set()).update(normalize_slice_indices(indices))
    return {name: tuple(sorted(merged[name])) for name in sorted(merged)}


def cache_hash_is_valid(stored_hash, fresh_hash) -> bool:
    '''True only when both hashes are present and identical.'''
    if stored_hash is None or fresh_hash is None:
        return False
    stored = str(stored_hash).strip()
    fresh = str(fresh_hash).strip()
    if not stored or not fresh:
        return False
    return stored == fresh


@dataclass
class MergedRecord:
    '''The single enforced record per ``(case_id, modality)`` key.'''

    case_id: str
    modality: str
    key: Key
    issue_slice_indices: IssueIndexMap = field(default_factory=dict)
    accepted_rows: int = 0
    rejected_rows: int = 0
    first_write_index: int = 0
    last_write_index: int = 0
    cache_hash: Optional[str] = None
    fresh_hash: Optional[str] = None
    cache_valid: bool = False
    extractor: Any = None
    extractor_built: bool = False
    extractor_refused: bool = False
    row: Optional[Row] = None

    @property
    def issue_types(self) -> Tuple[str, ...]:
        return tuple(sorted(self.issue_slice_indices))

    def recall_view(self) -> IssueIndexMap:
        return {name: tuple(indices) for name, indices in self.issue_slice_indices.items()}

    def as_row(self) -> Row:
        return {
            'case_id': self.case_id,
            'modality': self.modality,
            'issue_types': list(self.issue_types),
            'issue_slice_indices': {
                name: list(indices) for name, indices in sorted(self.issue_slice_indices.items())
            },
            'accepted_rows': self.accepted_rows,
            'rejected_rows': self.rejected_rows,
            'cache_hash': self.cache_hash,
            'cache_valid': self.cache_valid,
            'extractor_built': self.extractor_built,
        }


@dataclass
class WriteDecision:
    '''Outcome of a single write attempt against the gate.'''

    accepted: bool
    reason: str
    key: Key
    record: Optional[MergedRecord] = None
    audit_entry: Optional[Row] = None
    extractor_built: bool = False
    extractor_refused: bool = False


class CaseModalityWriteGate:
    '''Enforced, write-blocking ``(case_id, modality)`` de-duplication gate.'''

    ENFORCEMENT_MODE = 'write_blocking'
    ACCEPTED_REASON = 'accepted_new_key'
    DUPLICATE_REASON = 'duplicate_case_modality_rejected'
    MALFORMED_REASON = 'missing_case_modality_key'

    def __init__(self, cache_hashes=None, extractor_factory=None):
        self._records: Dict[Key, MergedRecord] = {}
        self._audit_log: List[Row] = []
        self._cache_hashes: Dict[Key, str] = {}
        for raw_key, value in (cache_hashes or {}).items():
            if value is None:
                continue
            self._cache_hashes[make_key(raw_key[0], raw_key[1])] = str(value)
        self._extractor_factory = extractor_factory
        self._write_attempts = 0
        self._extractor_builds = 0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def records(self) -> Dict[Key, MergedRecord]:
        return dict(self._records)

    @property
    def audit_log(self) -> List[Row]:
        '''Deep copy of every rejected write attempt, raw row included.'''
        return copy.deepcopy(self._audit_log)

    @property
    def rejected_rows(self) -> int:
        return len(self._audit_log)

    @property
    def duplicate_rejections(self) -> int:
        return sum(1 for entry in self._audit_log if entry.get('reason') == self.DUPLICATE_REASON)

    @property
    def write_attempts(self) -> int:
        return self._write_attempts

    @property
    def extractor_builds(self) -> int:
        return self._extractor_builds

    def queue_size(self) -> int:
        '''Number of review-queue records actually written.'''
        return len(self._records)

    def cached_hash(self, case_id, modality):
        return self._cache_hashes.get(make_key(case_id, modality))

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def write(self, row, fresh_hash=None, extractor_factory=None) -> WriteDecision:
        '''Attempt one write; duplicates are rejected and always audited.'''
        if not isinstance(row, MappingABC):
            raise TypeError('write() expects a mapping row')
        self._write_attempts += 1
        factory = extractor_factory or self._extractor_factory

        try:
            key = case_modality_key(row)
        except KeyError as exc:
            return self._reject_malformed(row, str(exc))

        incoming = issue_index_map(row)
        record = self._records.get(key)

        if record is None:
            record = MergedRecord(
                case_id=key[0],
                modality=key[1],
                key=key,
                first_write_index=self._write_attempts,
                last_write_index=self._write_attempts,
            )
            record.issue_slice_indices = merge_issue_index_maps({}, incoming)
            record.accepted_rows = 1
            record.row = copy.deepcopy(dict(row))
            self._records[key] = record
            built, refused = self._resolve_extractor(record, fresh_hash, factory)
            return WriteDecision(True, self.ACCEPTED_REASON, key, record, None, built, refused)

        # Duplicate key: block the second record, keep the evidence, keep the row.
        record.issue_slice_indices = merge_issue_index_maps(record.issue_slice_indices, incoming)
        record.rejected_rows += 1
        record.last_write_index = self._write_attempts
        built, refused = self._resolve_extractor(record, fresh_hash, factory)
        entry = self._duplicate_audit_entry(record, row, key)
        self._audit_log.append(entry)
        return WriteDecision(False, self.DUPLICATE_REASON, key, record, copy.deepcopy(entry), built, refused)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve_extractor(self, record, fresh_hash, factory):
        fresh = None if fresh_hash is None else str(fresh_hash)
        stored = self._cache_hashes.get(record.key)
        if stored is None:
            stored = record.cache_hash
        record.fresh_hash = fresh
        record.cache_hash = stored
        valid = cache_hash_is_valid(stored, fresh)
        record.cache_valid = valid

        if valid:
            # Cached hash still valid: no new extractor may be constructed.
            record.extractor_refused = True
            return False, True
        if record.extractor_built:
            # Hash mismatched but this key already has its extractor: reuse it.
            return False, False
        if factory is None:
            return False, False

        extractor = factory(record.key, fresh)
        record.extractor = extractor
        record.extractor_built = True
        record.cache_hash = fresh
        if fresh is not None:
            self._cache_hashes[record.key] = fresh
        self._extractor_builds += 1
        return True, False

    def _duplicate_audit_entry(self, record, row, key) -> Row:
        return {
            'audit_index': len(self._audit_log) + 1,
            'write_index': self._write_attempts,
            'enforcement_mode': self.ENFORCEMENT_MODE,
            'reason': self.DUPLICATE_REASON,
            'case_id': key[0],
            'modality': key[1],
            'key': [key[0], key[1]],
            'first_write_index': record.first_write_index,
            'accepted_rows': record.accepted_rows,
            'rejected_rows': record.rejected_rows,
            'row': copy.deepcopy(dict(row)),
        }

    def _reject_malformed(self, row, detail) -> WriteDecision:
        entry = {
            'audit_index': len(self._audit_log) + 1,
            'write_index': self._write_attempts,
            'enforcement_mode': self.ENFORCEMENT_MODE,
            'reason': self.MALFORMED_REASON,
            'detail': detail,
            'case_id': '',
            'modality': '',
            'key': ['', ''],
            'row': copy.deepcopy(dict(row)),
        }
        self._audit_log.append(entry)
        return WriteDecision(False, self.MALFORMED_REASON, ('', ''), None, copy.deepcopy(entry), False, False)


def run_write_gate(rows, cache_hashes=None, extractor_factory=None, fresh_hash_fn=None):
    '''Feed an iterable of rows through a fresh gate and return a summary.

    ``fresh_hash_fn(row)`` may supply the current content hash per row; when it
    is omitted no hash is supplied, so no extractor build can be justified by a
    valid cache and every new key is allowed at most one build.
    '''
    gate = CaseModalityWriteGate(cache_hashes=cache_hashes, extractor_factory=extractor_factory)
    decisions: List[WriteDecision] = []
    for row in rows:
        fresh = None
        if fresh_hash_fn is not None and isinstance(row, MappingABC):
            fresh = fresh_hash_fn(row)
        decisions.append(gate.write(row, fresh_hash=fresh))
    records = {key: record.as_row() for key, record in gate.records.items()}
    return {
        'gate': gate,
        'records': records,
        'record_count': len(records),
        'audit_log': gate.audit_log,
        'duplicate_rejections': gate.duplicate_rejections,
        'rejected_rows': gate.rejected_rows,
        'extractor_builds': gate.extractor_builds,
        'decisions': decisions,
    }
