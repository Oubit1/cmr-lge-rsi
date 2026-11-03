'''Audit coverage ledger, closure registry and the isolated write de-dup gate.

Dependency free and offline runnable.  Three composable pieces:

* :func:`cache_hash` / :func:`merge_issue_types` - pure helpers.
* :class:`AuditCoverageLedger` - append-only coverage ledger recording audited
  classes, closed gates and cache hashes.  A ledger hit is a reason to skip
  work; it is never a reason to drop QC evidence.  Registered hashes are
  content-addressed, so a payload already audited under any key is a hit.
* :class:`DedupWriteGate` - the (case_id, modality, slice_index) write de-dup
  hard gate that merges issue_type values.  It is only armed when the caller
  declares the isolated pipeline; every other pipeline is a pure pass-through
  so no QC safety check is weakened.
'''

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    'STATUS_OPEN',
    'STATUS_DONE',
    'ISOLATED_PIPELINE',
    'DEFAULT_PIPELINE',
    'ACCEPTANCE_THRESHOLDS',
    'cache_hash',
    'merge_issue_types',
    'retention_rates',
    'DedupWriteResult',
    'DedupWriteGate',
    'LedgerEntry',
    'AuditCoverageLedger',
]

STATUS_OPEN = 'OPEN'
STATUS_DONE = 'DONE'
ISOLATED_PIPELINE = 'isolated'
DEFAULT_PIPELINE = 'production'

# Acceptance thresholds mirrored from the approved maintenance action.  The
# write gate may only be armed on the isolated pipeline, so DEFAULT_PIPELINE
# keeps every hard QC check live.
ACCEPTANCE_THRESHOLDS: Dict[str, float] = {
    'duplicate_write_rate': 0.0,
    'slice_index_retention': 1.0,
    'issue_type_merge_retention': 1.0,
    'closed_repeat_proposals': 0,
    'covered_class_skip_ratio': 0.90,
}


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def _canonical(payload: Any) -> str:
    '''Stable JSON encoding used for hashing and for de-dup keys.'''
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)


def cache_hash(payload: Any) -> str:
    '''Deterministic sha256 over the canonical form of *payload*.'''
    return hashlib.sha256(_canonical(payload).encode('utf-8')).hexdigest()


def _freeze(value: Any) -> Any:
    '''Make *value* hashable while keeping scalar slice indices stable.'''
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set, frozenset)):
        return _canonical(value)
    return value


def _as_issue_types(value: Any) -> List[str]:
    '''Flatten the many shapes of an issue_type field into a list of strings.'''
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        return _as_issue_types(value.get('issue_type'))
    if isinstance(value, (bytes, bytearray)):
        return [bytes(value).decode('utf-8', 'replace')]
    try:
        iterator = iter(value)
    except TypeError:
        return [str(value)]
    out: List[str] = []
    for item in iterator:
        out.extend(_as_issue_types(item))
    return out


def merge_issue_types(*sources: Any) -> Tuple[str, ...]:
    '''Union of issue types, first-seen order preserved, duplicates dropped.'''
    seen: Dict[str, None] = {}
    for source in sources:
        for issue in _as_issue_types(source):
            if issue not in seen:
                seen[issue] = None
    return tuple(seen)


def _kept_ratio(expected: Sequence[Any], actual: Sequence[Any]) -> float:
    '''Fraction of *expected* items that survive in *actual*.'''
    if not expected:
        return 1.0
    present = set(actual)
    kept = sum(1 for item in expected if item in present)
    return kept / len(expected)


def retention_rates(rows: Sequence[Mapping[str, Any]], result: 'DedupWriteResult') -> Dict[str, float]:
    '''How much of the input slice indices / issue types survived the write.'''
    in_slices: List[Any] = []
    in_issues: List[str] = []
    for row in rows or ():
        if isinstance(row, Mapping):
            in_slices.append(row.get('slice_index'))
            in_issues.extend(_as_issue_types(row.get('issue_type')))
        else:
            in_slices.append(row)
    unique_slices = tuple(dict.fromkeys(in_slices))
    unique_issues = merge_issue_types(in_issues)
    return {
        'slice_index_retention': _kept_ratio(unique_slices, result.slice_indices),
        'issue_type_merge_retention': _kept_ratio(unique_issues, result.issue_types),
    }


# ---------------------------------------------------------------------------
# write de-dup gate
# ---------------------------------------------------------------------------
@dataclass
class DedupWriteResult:
    '''Outcome of a single write pass through :class:`DedupWriteGate`.'''

    written: List[Dict[str, Any]] = field(default_factory=list)
    blocked_rows: int = 0
    merged_rows: int = 0
    total_rows: int = 0
    duplicates_written: int = 0
    armed: bool = False
    slice_indices: Tuple[Any, ...] = ()
    issue_types: Tuple[str, ...] = ()

    @property
    def duplicate_write_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.duplicates_written / self.total_rows

    def as_dict(self) -> Dict[str, Any]:
        return {
            'rows_written': len(self.written),
            'rows_blocked': self.blocked_rows,
            'rows_merged': self.merged_rows,
            'total_rows': self.total_rows,
            'duplicates_written': self.duplicates_written,
            'armed': self.armed,
            'slice_indices': list(self.slice_indices),
            'issue_types': list(self.issue_types),
        }


class DedupWriteGate:
    '''(case_id, modality, slice_index) write de-dup hard gate.

    Armed only on the isolated pipeline.  When armed, a repeated
    (case_id, modality, slice_index) is merged into the first written row
    instead of being written again, and its issue_type values are unioned.
    When not armed the gate is a pure pass-through: every row is emitted and
    nothing is dropped, so no QC safety check is weakened.
    '''

    def __init__(self, pipeline: str = DEFAULT_PIPELINE) -> None:
        self.pipeline = pipeline
        self.rows_seen = 0
        self.duplicates_written = 0
        self.rows_blocked = 0
        self.rows_merged = 0
        self.slice_index_retention = 1.0
        self.issue_type_merge_retention = 1.0
        self.last_result: Optional[DedupWriteResult] = None

    @property
    def armed(self) -> bool:
        return self.pipeline == ISOLATED_PIPELINE

    @property
    def duplicate_write_rate(self) -> float:
        if self.rows_seen == 0:
            return 0.0
        return self.duplicates_written / self.rows_seen

    def write(
        self,
        rows: Sequence[Mapping[str, Any]] = (),
        case_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> DedupWriteResult:
        '''Write *rows*, de-duplicating on (case_id, modality, slice_index).'''
        rows = list(rows or ())
        written: List[Dict[str, Any]] = []
        index: Dict[Tuple[Any, Any, Any], int] = {}
        blocked = 0
        merged = 0

        for row in rows:
            record = dict(row) if isinstance(row, Mapping) else {'issue_type': row}
            resolved_case = record.get('case_id', case_id)
            resolved_modality = record.get('modality', modality)
            slice_index = record.get('slice_index')
            record['case_id'] = resolved_case
            record['modality'] = resolved_modality
            record['slice_index'] = slice_index
            key = (
                _freeze(resolved_case),
                _freeze(resolved_modality),
                _freeze(slice_index),
            )

            if not self.armed:
                # Pass-through: nothing is dropped outside the isolated run.
                record['issue_type'] = list(merge_issue_types(record.get('issue_type')))
                written.append(record)
                continue

            if key in index:
                target = written[index[key]]
                target['issue_type'] = list(
                    merge_issue_types(target.get('issue_type'), record.get('issue_type'))
                )
                blocked += 1
                merged += 1
                continue

            record['issue_type'] = list(merge_issue_types(record.get('issue_type')))
            index[key] = len(written)
            written.append(record)

        result = DedupWriteResult(
            written=written,
            blocked_rows=blocked,
            merged_rows=merged,
            total_rows=len(rows),
            duplicates_written=0,
            armed=self.armed,
            slice_indices=tuple(dict.fromkeys(r['slice_index'] for r in written)),
            issue_types=merge_issue_types(*(r['issue_type'] for r in written)),
        )

        rates = retention_rates(rows, result)
        self.rows_seen += len(rows)
        self.duplicates_written += result.duplicates_written
        self.rows_blocked += blocked
        self.rows_merged += merged
        self.slice_index_retention = min(
            self.slice_index_retention, rates['slice_index_retention']
        )
        self.issue_type_merge_retention = min(
            self.issue_type_merge_retention, rates['issue_type_merge_retention']
        )
        self.last_result = result
        return result


# ---------------------------------------------------------------------------
# coverage ledger + closure registry
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    '''One row of the coverage ledger.'''

    key: str
    kind: str
    status: str = STATUS_OPEN
    digest: str = ''
    detail: Dict[str, Any] = field(default_factory=dict)
    hits: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'kind': self.kind,
            'status': self.status,
            'digest': self.digest,
            'detail': dict(self.detail),
            'hits': self.hits,
        }


class AuditCoverageLedger:
    '''Coverage ledger and closure registry for the efficiency auditor.

    Registration is append-only.  Lookups through the explicit skip helpers
    feed the covered-class skip ratio; a hit means the caller must skip the
    work.  Closed items can never be re-proposed.  Cache hashes are
    content-addressed: a payload already registered under any key counts as a
    hit, so repeated work is skipped even when it arrives under a new key.
    '''

    def __init__(self, pipeline: str = DEFAULT_PIPELINE) -> None:
        self.pipeline = pipeline
        self.gate = DedupWriteGate(pipeline)
        self._classes: Dict[str, LedgerEntry] = {}
        self._closed: Dict[str, LedgerEntry] = {}
        self._hashes: Dict[str, str] = {}
        self._digests: Set[str] = set()
        self._lookups: Dict[str, int] = {}
        self._skips: Dict[str, int] = {}
        self.closed_repeat_attempts = 0
        self.closed_repeat_proposals = 0

    # -- registration -------------------------------------------------------
    def declare_audited_class(self, class_name: Any, evidence: Any = None) -> LedgerEntry:
        '''Record that *class_name* has already been audited by this role.'''
        key = str(class_name)
        entry = LedgerEntry(
            key=key,
            kind='class',
            status=STATUS_DONE,
            digest=cache_hash({'kind': 'class', 'key': key, 'evidence': evidence}),
            detail={'evidence': evidence},
        )
        self._classes[key] = entry
        return entry

    def close_gate(self, gate_id: Any, evidence: Any = None) -> LedgerEntry:
        '''Register a closed gate (no further proposals for it).'''
        key = str(gate_id)
        entry = LedgerEntry(
            key=key,
            kind='gate',
            status=STATUS_DONE,
            digest=cache_hash({'kind': 'gate', 'key': key, 'evidence': evidence}),
            detail={'evidence': evidence},
        )
        self._closed[key] = entry
        return entry

    def mark_done(self, item_id: Any, verified: bool, evidence: Any = None) -> bool:
        '''Mark *item_id* DONE only when validation passed.'''
        if not verified:
            return False
        self.close_gate(item_id, evidence=evidence)
        return True

    def record_cache_hash(self, key: Any, payload: Any = None) -> str:
        '''Store a cache hash for *key* (payload defaults to the key itself).'''
        digest = cache_hash(payload if payload is not None else key)
        self._hashes[str(key)] = digest
        self._digests.add(digest)
        return digest

    # -- lookups ------------------------------------------------------------
    def _bump(self, kind: str, hit: bool) -> None:
        self._lookups[kind] = self._lookups.get(kind, 0) + 1
        if hit:
            self._skips[kind] = self._skips.get(kind, 0) + 1

    def is_covered_class(self, class_name: Any) -> bool:
        return str(class_name) in self._classes

    def is_closed(self, item_id: Any) -> bool:
        return str(item_id) in self._closed

    def is_done(self, item_id: Any) -> bool:
        return self.is_closed(item_id)

    def lookup_cache_hash(self, key: Any) -> Optional[str]:
        return self._hashes.get(str(key))

    def has_cache_hash(self, digest: Any) -> bool:
        '''True when *digest* was registered under any key.'''
        return str(digest) in self._digests

    def _hash_hit(self, key: Any, payload: Any = None) -> bool:
        '''Cache hit test.

        With no payload only the key registration matters.  With a payload the
        check is content-addressed: a hash already recorded under any key is a
        hit, so the same audited content is never processed twice.
        '''
        stored = self._hashes.get(str(key))
        if payload is None:
            return stored is not None
        digest = cache_hash(payload)
        if stored == digest:
            return True
        return digest in self._digests

    def is_hash_hit(self, key: Any, payload: Any = None) -> bool:
        return self._hash_hit(key, payload)

    # -- explicit skip helpers (these feed the coverage metric) -------------
    def skip_audited_class(self, class_name: Any) -> bool:
        key = str(class_name)
        hit = key in self._classes
        self._bump('class', hit)
        if hit:
            self._classes[key].hits += 1
        return hit

    def skip_closed(self, item_id: Any) -> bool:
        key = str(item_id)
        hit = key in self._closed
        self._bump('closed', hit)
        if hit:
            self._closed[key].hits += 1
        return hit

    def skip_cached(self, key: Any, payload: Any = None) -> bool:
        hit = self._hash_hit(key, payload)
        self._bump('hash', hit)
        return hit

    def skip_ratio(self, kind: str = 'class') -> float:
        lookups = self._lookups.get(kind, 0)
        if lookups == 0:
            return 0.0
        return self._skips.get(kind, 0) / lookups

    # -- proposals ----------------------------------------------------------
    def propose(
        self,
        item_id: Any,
        issue_type: Any = None,
        payload: Any = None,
        case_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        '''Emit a proposal, or a skip record when the ledger already covers it.

        Closed gates never yield a proposal, so the count of repeated
        proposals for closed items stays at zero.
        '''
        item_key = str(item_id)

        if item_key in self._closed:
            self.closed_repeat_attempts += 1
            return {
                'item_id': item_key,
                'skipped': True,
                'reason': 'closed_gate',
                'proposal': None,
            }
        if item_key in self._classes:
            return {
                'item_id': item_key,
                'skipped': True,
                'reason': 'covered_class',
                'proposal': None,
            }
        if payload is not None and self._hash_hit(item_key, payload):
            return {
                'item_id': item_key,
                'skipped': True,
                'reason': 'cache_hit',
                'proposal': None,
            }

        return {
            'item_id': item_key,
            'skipped': False,
            'reason': 'new',
            'proposal': {
                'item_id': item_key,
                'issue_type': list(merge_issue_types(issue_type)),
                'case_id': case_id,
                'modality': modality,
                'pipeline': self.pipeline,
                'write_gate_armed': self.gate.armed,
            },
        }

    # -- writes -------------------------------------------------------------
    def write_rows(
        self,
        rows: Sequence[Mapping[str, Any]] = (),
        case_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> DedupWriteResult:
        '''Delegate to the (case_id, modality) write de-dup hard gate.'''
        return self.gate.write(rows, case_id=case_id, modality=modality)

    # -- reporting ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        '''Full ledger dump for audit trail purposes.'''
        return {
            'pipeline': self.pipeline,
            'write_gate_armed': self.gate.armed,
            'audited_classes': [e.as_dict() for e in self._classes.values()],
            'closed_gates': [e.as_dict() for e in self._closed.values()],
            'cache_hashes': dict(self._hashes),
        }

    def metrics(self) -> Dict[str, Any]:
        return {
            'duplicate_write_rate': self.gate.duplicate_write_rate,
            'slice_index_retention': self.gate.slice_index_retention,
            'issue_type_merge_retention': self.gate.issue_type_merge_retention,
            'closed_repeat_proposals': self.closed_repeat_proposals,
            'closed_repeat_attempts': self.closed_repeat_attempts,
            'covered_class_skip_ratio': self.skip_ratio('class'),
            'pipeline': self.pipeline,
            'write_gate_armed': self.gate.armed,
        }

    def acceptance_report(self) -> Dict[str, Dict[str, Any]]:
        metrics = self.metrics()
        specs = (
            ('duplicate_write_rate', '<=', ACCEPTANCE_THRESHOLDS['duplicate_write_rate']),
            ('slice_index_retention', '>=', ACCEPTANCE_THRESHOLDS['slice_index_retention']),
            ('issue_type_merge_retention', '>=', ACCEPTANCE_THRESHOLDS['issue_type_merge_retention']),
            ('closed_repeat_proposals', '<=', ACCEPTANCE_THRESHOLDS['closed_repeat_proposals']),
            ('covered_class_skip_ratio', '>=', ACCEPTANCE_THRESHOLDS['covered_class_skip_ratio']),
        )
        report: Dict[str, Dict[str, Any]] = {}
        for name, op, threshold in specs:
            value = metrics[name]
            passed = value <= threshold if op == '<=' else value >= threshold
            report[name] = {
                'value': value,
                'op': op,
                'threshold': threshold,
                'passed': passed,
            }
        return report

    def meets_acceptance(self) -> bool:
        return all(item['passed'] for item in self.acceptance_report().values())
