'''Isolated coverage ledger with DONE marks and write-side (case_id, modality) dedup.

This module is an offline experiment. It does not touch datasets, patient images,
production labels, credentials, or runtime databases.

Key behaviours:
- Coverage ledger keyed by (issue_type, modality, cache_hash, rule_version).
- A class marked DONE with an unchanged cache_hash is skipped on replay.
- Write-side rows are deduplicated by (case_id, modality); the first slice_index
  is preserved and issue_type sets are merged so recall does not drop.
- Duplicate rows are rejected explicitly.
- After validation the ledger refuses to propose new audits.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    'CoverageKey',
    'CoverageRecord',
    'CoverageLedger',
    'WriteRecord',
    'DuplicateCaseModalityError',
    'merge_case_modality',
    'merge_case_modality_strict',
    'replay_runs',
]

CoverageKey = Tuple[str, str, str, str]


@dataclass(frozen=True)
class CoverageRecord:
    '''A single ledger entry for an audited class.'''

    issue_type: str
    modality: str
    cache_hash: str
    rule_version: str
    status: str = 'DONE'

    def key(self) -> CoverageKey:
        return (self.issue_type, self.modality, self.cache_hash, self.rule_version)


class CoverageLedger:
    '''In-memory coverage ledger for audit-class DONE marks.'''

    def __init__(self) -> None:
        self._records: Dict[CoverageKey, CoverageRecord] = {}
        self._validated = False

    def mark_done(
        self,
        issue_type: str,
        modality: str,
        cache_hash: str,
        rule_version: str,
    ) -> CoverageRecord:
        '''Record a class as DONE and return the stored record.'''
        record = CoverageRecord(issue_type, modality, cache_hash, rule_version)
        self._records[record.key()] = record
        return record

    def is_covered(
        self,
        issue_type: str,
        modality: str,
        cache_hash: str,
        rule_version: str,
    ) -> bool:
        '''Return True when the exact class is DONE.'''
        record = self._records.get((issue_type, modality, cache_hash, rule_version))
        return record is not None and record.status == 'DONE'

    def should_skip_audit(
        self,
        issue_type: str,
        modality: str,
        cache_hash: str,
        rule_version: str,
    ) -> bool:
        '''Skip audit only for an exact DONE hit with unchanged hash/rule version.'''
        return self.is_covered(issue_type, modality, cache_hash, rule_version)

    def mark_validated(self) -> None:
        '''Freeze the ledger: no new audit proposals may be created.'''
        self._validated = True

    @property
    def validated(self) -> bool:
        return self._validated

    def propose_audit(
        self,
        issue_type: str,
        modality: str,
        cache_hash: str,
        rule_version: str,
    ) -> Optional[CoverageRecord]:
        '''Return a new DONE record, or None if validated or already covered.'''
        if self._validated:
            return None
        if self.is_covered(issue_type, modality, cache_hash, rule_version):
            return None
        return self.mark_done(issue_type, modality, cache_hash, rule_version)


@dataclass(frozen=True)
class WriteRecord:
    '''Write-side row from the isolated pipeline.'''

    case_id: str
    modality: str
    issue_types: Tuple[str, ...]
    slice_index: int

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError('case_id must be non-empty')
        if not self.modality:
            raise ValueError('modality must be non-empty')
        if self.slice_index < 0:
            raise ValueError('slice_index must be non-negative')
        object.__setattr__(self, 'issue_types', tuple(sorted(set(self.issue_types))))


class DuplicateCaseModalityError(ValueError):
    '''Raised when duplicate (case_id, modality) rows are present in strict mode.'''


def merge_case_modality(
    records: Iterable[WriteRecord],
) -> Tuple[List[WriteRecord], List[WriteRecord]]:
    '''Merge records by (case_id, modality).

    Returns ``(merged, rejected)`` where ``merged`` contains one record per
    (case_id, modality). The first record's ``slice_index`` is preserved and
    all ``issue_types`` are unioned so recall does not drop. Later duplicate
    rows are returned in ``rejected``.
    '''
    seen: Dict[Tuple[str, str], WriteRecord] = {}
    order: List[Tuple[str, str]] = []
    rejected: List[WriteRecord] = []

    for record in records:
        key = (record.case_id, record.modality)
        if key in seen:
            rejected.append(record)
            existing = seen[key]
            merged_types = tuple(sorted(set(existing.issue_types) | set(record.issue_types)))
            seen[key] = WriteRecord(
                case_id=existing.case_id,
                modality=existing.modality,
                issue_types=merged_types,
                slice_index=existing.slice_index,
            )
        else:
            seen[key] = record
            order.append(key)

    merged = [seen[key] for key in order]
    return merged, rejected


def merge_case_modality_strict(records: Iterable[WriteRecord]) -> List[WriteRecord]:
    '''Merge and reject duplicate rows by raising ``DuplicateCaseModalityError``.'''
    merged, rejected = merge_case_modality(records)
    if rejected:
        raise DuplicateCaseModalityError(
            f'rejected {len(rejected)} duplicate (case_id, modality) row(s)'
        )
    return merged


def replay_runs(runs: Iterable[dict], ledger: CoverageLedger) -> dict:
    '''Replay run records against a ledger.

    Each run dict must contain ``issue_type``, ``modality``, ``cache_hash`` and
    ``rule_version``. Covered runs are skipped; uncovered runs may be proposed
    unless the ledger is validated.
    '''
    skipped = 0
    proposed = 0
    skipped_keys: List[CoverageKey] = []

    for run in runs:
        issue_type = run['issue_type']
        modality = run['modality']
        cache_hash = run['cache_hash']
        rule_version = run['rule_version']
        key = (issue_type, modality, cache_hash, rule_version)

        if ledger.should_skip_audit(*key):
            skipped += 1
            skipped_keys.append(key)
            continue

        proposal = ledger.propose_audit(*key)
        if proposal is not None:
            proposed += 1

    return {
        'skipped': skipped,
        'proposed': proposed,
        'skipped_keys': skipped_keys,
    }
