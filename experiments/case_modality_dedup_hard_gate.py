'''Write-side (case_id, modality) dedup hard gate for the isolated QC run pipeline.

Dependency-free (standard library only) so it runs inside the isolated
experiment harness without downloading anything and without touching datasets,
patient images, production labels, credentials or runtime databases.

Three coupled guarantees:

1. One row per key: `admit` is the only write path. The first row for a
   (case_id, modality) key is canonical; every later row is merged into it
   (issue_type set union, slice_index set union) and never becomes a second
   ledger row, so duplicates per key are exactly 0 while unique issue_type
   coverage is monotonically non-decreasing.
2. No extractor fan-out behind a valid cache: `resolve_extractor` answers
   REUSE whenever a valid content hash is already cached for the key; a new
   extractor is only permitted when no valid hash exists.
3. DONE is terminal: `mark_validated` / `close` drive a key to DONE, after
   which `propose_dedup` never issues again and writes are rejected until an
   explicit `reopen` with a reason. `convergence_verdict` turns three clean
   consecutive runs into a stop-or-escalate decision for the efficiency auditor.

The gate is fail-closed: caching never weakens a safety check and a DONE key
cannot be silently mutated by a later run.
'''

from __future__ import annotations

import re
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

__all__ = [
    'STATE_OPEN',
    'STATE_VALIDATED',
    'STATE_DONE',
    'ACTION_CANONICAL',
    'ACTION_MERGED',
    'ACTION_REJECTED',
    'EXTRACTOR_REUSE',
    'EXTRACTOR_CREATE',
    'DEFAULT_CONVERGENCE_WINDOW',
    'GateDecision',
    'ExtractorDecision',
    'DedupProposal',
    'CaseModalityDedupHardGate',
    'normalize_case_id',
    'normalize_modality',
    'dedup_key',
    'normalize_issue_types',
    'normalize_slice_indices',
    'is_valid_hash',
    'normalize_cache_hash',
    'dedup_records',
    'duplicate_record_count',
    'issue_type_coverage',
    'dedup_invariants',
    'proposal_kinds',
    'dedup_proposals_in_history',
    'clean_run_streak',
    'convergence_verdict',
]

STATE_OPEN = 'OPEN'
STATE_VALIDATED = 'VALIDATED'
STATE_DONE = 'DONE'

ACTION_CANONICAL = 'CANONICAL'
ACTION_MERGED = 'MERGED'
ACTION_REJECTED = 'REJECTED'

EXTRACTOR_REUSE = 'REUSE'
EXTRACTOR_CREATE = 'CREATE'

DEFAULT_CONVERGENCE_WINDOW = 3

_HASH_RE = re.compile(r'^[0-9a-f]{8,128}$')
_DEDUP_MARKERS = ('dedup', 'de-dup', 'duplicate')
_INVALID_KEY = ('<invalid>', '<invalid>')


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------

def normalize_case_id(case_id):
    '''Trim a case id; raise when it is missing or empty.'''
    if case_id is None:
        raise ValueError('case_id is required')
    text = str(case_id).strip()
    if not text:
        raise ValueError('case_id is required')
    return text


def normalize_modality(modality):
    '''Trim and upper-case a modality; raise when it is missing or empty.'''
    if modality is None:
        raise ValueError('modality is required')
    text = str(modality).strip().upper()
    if not text:
        raise ValueError('modality is required')
    return text


def dedup_key(case_id, modality):
    '''Canonical (case_id, modality) key used by every guard below.'''
    return (normalize_case_id(case_id), normalize_modality(modality))


def _as_sequence(value):
    '''Coerce scalars, mappings and iterables into a tuple without losing 0.'''
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bool, int, float)):
        return (value,)
    if isinstance(value, MappingABC):
        return (value,)
    if isinstance(value, IterableABC):
        return tuple(value)
    return (value,)


def normalize_issue_types(values):
    '''Trim, drop empties and de-duplicate issue types, keeping first-seen order.'''
    ordered = []
    seen = set()
    for value in _as_sequence(values):
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def normalize_slice_indices(values):
    '''Coerce slice indices to a sorted unique tuple, ignoring unusable values.'''
    indices = set()
    for value in _as_sequence(values):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            indices.add(value)
            continue
        if isinstance(value, float):
            if value.is_integer():
                indices.add(int(value))
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                indices.add(int(text))
            except ValueError:
                continue
    return tuple(sorted(indices))


def is_valid_hash(value):
    '''A cache hash is valid only when it is a hex string of usable length.'''
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    return bool(_HASH_RE.match(text))


def normalize_cache_hash(value):
    '''Return a lower-cased valid hash, or None when the value is unusable.'''
    if not is_valid_hash(value):
        return None
    return str(value).strip().lower()


def _record_key(item):
    '''Best-effort key extraction for externally supplied records.'''
    if isinstance(item, MappingABC):
        case_id = item.get('case_id')
        modality = item.get('modality')
    else:
        case_id = getattr(item, 'case_id', None)
        modality = getattr(item, 'modality', None)
    try:
        return dedup_key(case_id, modality)
    except ValueError:
        return _INVALID_KEY


def _issue_types_of(item):
    if isinstance(item, MappingABC):
        values = item.get('issue_types')
        if values is None:
            values = item.get('issue_type')
    else:
        values = getattr(item, 'issue_types', None)
        if values is None:
            values = getattr(item, 'issue_type', None)
    return values


def _slice_indices_of(item):
    if isinstance(item, MappingABC):
        values = item.get('slice_indices')
        if values is None:
            values = item.get('slice_index')
    else:
        values = getattr(item, 'slice_indices', None)
        if values is None:
            values = getattr(item, 'slice_index', None)
    return values


# ---------------------------------------------------------------------------
# decision records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDecision:
    key: Tuple[str, str]
    action: str
    canonical: bool
    merged: bool
    closed: bool
    issue_types: Tuple[str, ...]
    slice_indices: Tuple[int, ...]
    duplicate_records_after: int
    reason: str


@dataclass(frozen=True)
class ExtractorDecision:
    key: Tuple[str, str]
    action: str
    reuse: bool
    created: bool
    adopted: bool
    cache_hash: Optional[str]
    reason: str


@dataclass(frozen=True)
class DedupProposal:
    key: Tuple[str, str]
    issued: bool
    reason: str
    kind: str = 'dedup'
    issue_types: Tuple[str, ...] = ()
    slice_indices: Tuple[int, ...] = ()


@dataclass
class _CanonicalRow:
    case_id: str
    modality: str
    issue_types: Set[str] = field(default_factory=set)
    slice_indices: Set[int] = field(default_factory=set)
    state: str = STATE_OPEN
    merged_records: int = 0
    duplicate_records_suppressed: int = 0
    proposals_issued: int = 0
    validation_passes: int = 0
    validation_failures: int = 0
    reopens: int = 0

    @property
    def key(self):
        return (self.case_id, self.modality)

    def snapshot(self):
        return {
            'key': self.key,
            'case_id': self.case_id,
            'modality': self.modality,
            'issue_types': tuple(sorted(self.issue_types)),
            'slice_indices': tuple(sorted(self.slice_indices)),
            'state': self.state,
            'merged_records': self.merged_records,
            'duplicate_records_suppressed': self.duplicate_records_suppressed,
            'proposals_issued': self.proposals_issued,
            'validation_passes': self.validation_passes,
            'validation_failures': self.validation_failures,
            'reopens': self.reopens,
        }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

class CaseModalityDedupHardGate:
    '''Fail-closed write-side gate keyed on (case_id, modality).

    Call `admit` on the write path, `resolve_extractor` before any extraction
    work, and `propose_dedup` when the auditor considers a dedup proposal.
    Validation passes drive the key to DONE.
    '''

    def __init__(self):
        self._rows = {}
        self._cache = {}
        self._extractors_created = 0
        self._extractor_reuses = 0
        self._proposals_issued = 0
        self._proposals_suppressed = 0
        self._rejected_writes = 0
        self._reopens = 0

    # -- telemetry ---------------------------------------------------------

    @property
    def extractors_created(self):
        return self._extractors_created

    @property
    def cache_reuses(self):
        return self._extractor_reuses

    @property
    def proposals_issued(self):
        return self._proposals_issued

    @property
    def proposals_suppressed(self):
        return self._proposals_suppressed

    @property
    def rejected_writes(self):
        return self._rejected_writes

    @property
    def reopens(self):
        return self._reopens

    def duplicate_record_count(self, rows=None):
        '''Duplicate (case_id, modality) rows in the ledger (0 once gated).'''
        items = list(self.rows() if rows is None else rows)
        return duplicate_record_count(items)

    def issue_type_coverage(self, rows=None):
        items = list(self.rows() if rows is None else rows)
        return issue_type_coverage(items)

    # -- ledger views ------------------------------------------------------

    def rows(self):
        snapshots = []
        for key, row in self._rows.items():
            snapshot = row.snapshot()
            snapshot['cache_hash'] = self._cache.get(key)
            snapshots.append(snapshot)
        return snapshots

    def record(self, case_id, modality):
        key = dedup_key(case_id, modality)
        row = self._rows.get(key)
        if row is None:
            return None
        snapshot = row.snapshot()
        snapshot['cache_hash'] = self._cache.get(key)
        return snapshot

    def state(self, case_id, modality):
        record = self.record(case_id, modality)
        return None if record is None else record['state']

    def write_plan(self, case_id, modality):
        record = self.record(case_id, modality)
        if record is None:
            raise KeyError('no canonical record for key')
        return {
            'case_id': record['case_id'],
            'modality': record['modality'],
            'issue_types': record['issue_types'],
            'slice_indices': record['slice_indices'],
            'state': record['state'],
            'accepts_writes': record['state'] != STATE_DONE,
        }

    def pending_dedup_keys(self):
        '''Open keys that already absorbed duplicate writes and still need review.'''
        return tuple(
            key
            for key, row in self._rows.items()
            if row.state == STATE_OPEN and row.duplicate_records_suppressed > 0
        )

    # -- write-side hard gate ---------------------------------------------

    def admit(self, case_id, modality, issue_types=(), slice_indices=()):
        '''Admit one evidence row, merging it into the canonical row for the key.'''
        key = dedup_key(case_id, modality)
        new_issue_types = normalize_issue_types(issue_types)
        new_slices = normalize_slice_indices(slice_indices)
        row = self._rows.get(key)

        if row is None:
            row = _CanonicalRow(case_id=key[0], modality=key[1])
            row.issue_types.update(new_issue_types)
            row.slice_indices.update(new_slices)
            self._rows[key] = row
            return GateDecision(
                key=key,
                action=ACTION_CANONICAL,
                canonical=True,
                merged=False,
                closed=False,
                issue_types=tuple(sorted(row.issue_types)),
                slice_indices=tuple(sorted(row.slice_indices)),
                duplicate_records_after=self.duplicate_record_count(),
                reason='first row for key is canonical',
            )

        if row.state == STATE_DONE:
            self._rejected_writes += 1
            return GateDecision(
                key=key,
                action=ACTION_REJECTED,
                canonical=False,
                merged=False,
                closed=True,
                issue_types=tuple(sorted(row.issue_types)),
                slice_indices=tuple(sorted(row.slice_indices)),
                duplicate_records_after=self.duplicate_record_count(),
                reason='key is DONE; reopen with a reason before accepting evidence',
            )

        row.issue_types.update(new_issue_types)
        row.slice_indices.update(new_slices)
        row.merged_records += 1
        row.duplicate_records_suppressed += 1
        return GateDecision(
            key=key,
            action=ACTION_MERGED,
            canonical=False,
            merged=True,
            closed=False,
            issue_types=tuple(sorted(row.issue_types)),
            slice_indices=tuple(sorted(row.slice_indices)),
            duplicate_records_after=self.duplicate_record_count(),
            reason='merged into canonical row instead of writing a duplicate',
        )

    def admit_many(self, records):
        '''Batch write-path entry point: one-shot dedup for a stream of records.'''
        decisions = []
        for item in records or ():
            if isinstance(item, MappingABC):
                case_id = item.get('case_id')
                modality = item.get('modality')
            else:
                case_id = getattr(item, 'case_id', None)
                modality = getattr(item, 'modality', None)
            decisions.append(
                self.admit(case_id, modality, _issue_types_of(item), _slice_indices_of(item))
            )
        return decisions

    # -- extractor cache guard --------------------------------------------

    def register_cache_hash(self, case_id, modality, content_hash):
        '''Store a cache hash; invalid hashes never replace a valid one.'''
        key = dedup_key(case_id, modality)
        normalized = normalize_cache_hash(content_hash)
        if normalized is None:
            return False
        self._cache[key] = normalized
        return True

    def cache_hash(self, case_id, modality):
        return self._cache.get(dedup_key(case_id, modality))

    def resolve_extractor(self, case_id, modality, content_hash=None):
        '''Reuse cached artifacts whenever a valid content hash is available.'''
        key = dedup_key(case_id, modality)
        normalized = normalize_cache_hash(content_hash)
        if normalized is not None:
            self.register_cache_hash(key[0], key[1], normalized)
        cached = self._cache.get(key)
        if cached is not None:
            self._extractor_reuses += 1
            return ExtractorDecision(
                key=key,
                action=EXTRACTOR_REUSE,
                reuse=True,
                created=False,
                adopted=normalized is not None and normalized == cached,
                cache_hash=cached,
                reason='valid content hash cached for key; extractor reuse required',
            )
        self._extractors_created += 1
        return ExtractorDecision(
            key=key,
            action=EXTRACTOR_CREATE,
            reuse=False,
            created=True,
            adopted=False,
            cache_hash=None,
            reason='no valid content hash cached for key; new extractor permitted',
        )

    # -- lifecycle ---------------------------------------------------------

    def mark_validated(self, case_id, modality, passed=True):
        '''Record a validation outcome. A passed run moves the key to VALIDATED.'''
        key = dedup_key(case_id, modality)
        row = self._rows.get(key)
        if row is None:
            row = _CanonicalRow(case_id=key[0], modality=key[1])
            self._rows[key] = row
        if row.state == STATE_DONE:
            return False
        if passed:
            row.validation_passes += 1
            row.state = STATE_VALIDATED
            return True
        row.validation_failures += 1
        row.state = STATE_OPEN
        return False

    def close(self, case_id, modality):
        '''Mark DONE once validation passed; idempotent and fail-closed.'''
        key = dedup_key(case_id, modality)
        row = self._rows.get(key)
        if row is None:
            return False
        if row.state == STATE_DONE:
            return False
        if row.validation_passes < 1:
            return False
        row.state = STATE_DONE
        return True

    def reopen(self, case_id, modality, reason):
        '''Explicitly leave DONE with a reason; otherwise DONE stays terminal.'''
        key = dedup_key(case_id, modality)
        row = self._rows.get(key)
        if row is None:
            return False
        text = str(reason).strip() if reason is not None else ''
        if not text:
            return False
        if row.state != STATE_DONE:
            return False
        row.state = STATE_OPEN
        row.reopens += 1
        self._reopens += 1
        return True

    def propose_dedup(self, case_id, modality):
        '''Issue a dedup proposal at most once per key, never after DONE.'''
        key = dedup_key(case_id, modality)
        row = self._rows.get(key)
        if row is None:
            self._proposals_suppressed += 1
            return DedupProposal(
                key=key,
                issued=False,
                reason='no canonical record for key',
            )
        if row.state == STATE_DONE:
            self._proposals_suppressed += 1
            return DedupProposal(
                key=key,
                issued=False,
                reason='key is DONE; dedup proposal permanently suppressed',
                issue_types=tuple(sorted(row.issue_types)),
                slice_indices=tuple(sorted(row.slice_indices)),
            )
        if row.duplicate_records_suppressed > 0 and row.proposals_issued == 0:
            row.proposals_issued += 1
            self._proposals_issued += 1
            return DedupProposal(
                key=key,
                issued=True,
                reason='merged duplicates on this key need a single review pass',
                issue_types=tuple(sorted(row.issue_types)),
                slice_indices=tuple(sorted(row.slice_indices)),
            )
        self._proposals_suppressed += 1
        return DedupProposal(
            key=key,
            issued=False,
            reason='dedup already handled at the write gate; no new proposal',
            issue_types=tuple(sorted(row.issue_types)),
            slice_indices=tuple(sorted(row.slice_indices)),
        )


# ---------------------------------------------------------------------------
# record-level rollup helpers (usable on raw pre-gate records)
# ---------------------------------------------------------------------------

def duplicate_record_count(records):
    '''Count rows beyond the first for every (case_id, modality) key.'''
    counts = {}
    for item in records or ():
        key = _record_key(item)
        counts[key] = counts.get(key, 0) + 1
    return sum(count - 1 for count in counts.values())


def issue_type_coverage(records):
    '''Union of issue types across records, normalized.'''
    coverage = set()
    for item in records or ():
        coverage.update(normalize_issue_types(_issue_types_of(item)))
    return frozenset(coverage)


def dedup_records(records):
    '''Fold raw records into one canonical entry per (case_id, modality) key.'''
    merged = {}
    order = []
    for item in records or ():
        key = dedup_key(
            item.get('case_id') if isinstance(item, MappingABC) else getattr(item, 'case_id', None),
            item.get('modality') if isinstance(item, MappingABC) else getattr(item, 'modality', None),
        )
        if key not in merged:
            merged[key] = {
                'case_id': key[0],
                'modality': key[1],
                'issue_types': [],
                'slice_indices': [],
                'records': 0,
            }
            order.append(key)
        entry = merged[key]
        for text in normalize_issue_types(_issue_types_of(item)):
            if text not in entry['issue_types']:
                entry['issue_types'].append(text)
        merged_slices = set(entry['slice_indices'])
        merged_slices.update(normalize_slice_indices(_slice_indices_of(item)))
        entry['slice_indices'] = sorted(merged_slices)
        entry['records'] = entry['records'] + 1
    return dict((key, merged[key]) for key in order)


def dedup_invariants(records):
    '''Audit report: duplicate count, unique keys and preserved coverage.'''
    items = list(records or ())
    merged = dedup_records(items)
    duplicates = duplicate_record_count(items)
    return {
        'records': len(items),
        'unique_keys': len(merged),
        'duplicate_records': duplicates,
        'no_duplicates': duplicates == 0,
        'issue_type_coverage': issue_type_coverage(items),
        'slice_indices': dict(
            (key, tuple(entry['slice_indices'])) for key, entry in merged.items()
        ),
    }


# ---------------------------------------------------------------------------
# iteration-history convergence check
# ---------------------------------------------------------------------------

def _normalized_kind(proposal):
    '''Normalize a proposal kind; suppressed proposals (issued=False) count as none.'''
    if isinstance(proposal, MappingABC):
        issued = proposal.get('issued')
        value = proposal.get('kind')
        if value is None:
            value = proposal.get('type')
        if value is None:
            value = proposal.get('action')
    else:
        issued = getattr(proposal, 'issued', None)
        value = getattr(proposal, 'kind', None)
        if value is None:
            value = getattr(proposal, 'type', None)
        if value is None:
            value = getattr(proposal, 'action', None)
    if issued is False:
        return ''
    text = str(value).strip().lower() if value is not None else ''
    if not text:
        return ''
    for marker in _DEDUP_MARKERS:
        if marker in text:
            return 'dedup'
    return text


def proposal_kinds(proposals):
    '''Sorted unique normalized kinds, with every dedup variant collapsed to dedup.'''
    kinds = set()
    for proposal in proposals or ():
        kind = _normalized_kind(proposal)
        if kind:
            kinds.add(kind)
    return tuple(sorted(kinds))


def _proposals_of_run(run):
    if isinstance(run, MappingABC):
        proposals = run.get('proposals')
    else:
        proposals = getattr(run, 'proposals', None)
    return tuple(proposals or ())


def _run_has_dedup(run):
    for proposal in _proposals_of_run(run):
        if 'dedup' in proposal_kinds((proposal,)):
            return True
    return False


def dedup_proposals_in_history(iteration_history):
    '''Total issued dedup proposals across all recorded runs.'''
    total = 0
    for run in iteration_history or ():
        if _run_has_dedup(run):
            total += len([
                p for p in _proposals_of_run(run) if 'dedup' in proposal_kinds((p,))
            ])
    return total


def clean_run_streak(iteration_history):
    '''Number of trailing consecutive runs with no issued dedup proposal.'''
    streak = 0
    for run in reversed(list(iteration_history or ())):
        if _run_has_dedup(run):
            break
        streak += 1
    return streak


def convergence_verdict(iteration_history, window=DEFAULT_CONVERGENCE_WINDOW):
    '''Decide STOP_CLASS versus CONTINUE/ escalate after a clean-run window.'''
    if window < 1:
        raise ValueError('window must be >= 1')
    streak = clean_run_streak(iteration_history)
    converged = streak >= window
    if converged:
        reason = 'no dedup proposal in the last %d runs; class converged' % window
    else:
        reason = 'only %d clean run(s) in a row; keep iterating before escalation' % streak
    return {
        'converged': converged,
        'clean_streak': streak,
        'window': window,
        'action': 'STOP_CLASS' if converged else 'CONTINUE',
        'escalate_to_human': not converged,
        'reason': reason,
    }
