'''Case-modality write-side dedup closure hard gate.

This module is intentionally self-contained and offline. It implements a single
closure pass for (case_id, modality) rows: duplicate rows are rejected, their
issue_type sets and slice indices are merged into the first representative row,
and a validated pass marks the gate DONE. Once DONE, an unchanged cache hash or
rule version produces no new proposal.
'''

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RULE_VERSION = 'case_modality_write_dedup_closure.v1'
DEFAULT_MIN_ISSUE_TYPE_RETENTION = 0.999


@dataclass(frozen=True)
class RejectedRow:
    index: int
    reason: str
    case_id: str
    modality: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class GateResult:
    rule_version: str
    deduped: Tuple[Mapping[str, Any], ...]
    rejected: Tuple[RejectedRow, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class ClosureDecision:
    status: str
    rule_version: str
    fingerprint: str
    proposal: Optional[Mapping[str, Any]]
    validation: ValidationResult
    metrics: Mapping[str, Any]
    deduped_rows: Tuple[Mapping[str, Any], ...]
    rejected_rows: Tuple[RejectedRow, ...]
    done_marked: bool
    skipped: bool


def _as_issue_types(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (value,)
    else:
        try:
            items = tuple(value)
        except TypeError:
            items = (value,)
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(sorted(set(out)))


def _as_slice_indices(value: Any) -> Tuple[int, ...]:
    if value is None or isinstance(value, bool):
        return ()
    if isinstance(value, int):
        items: Iterable[Any] = (value,)
    else:
        try:
            items = tuple(value)
        except TypeError:
            items = ()
    out: List[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        out.append(index)
    return tuple(sorted(set(out)))


def _key_from_row(row: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(row, Mapping):
        return None
    case_id = str(row.get('case_id', '')).strip()
    modality = str(row.get('modality', '')).strip()
    if not case_id or not modality:
        return None
    return (case_id, modality.lower())


def _canonical_row_for_hash(row: Any) -> Any:
    key = _key_from_row(row)
    if key is None:
        return {'invalid': True, 'raw': repr(row)}
    return {
        'case_id': key[0],
        'modality': key[1],
        'issue_types': list(_as_issue_types(row.get('issue_type'))),
        'slice_indices': list(_as_slice_indices(row.get('slice_index'))),
    }


def canonical_fingerprint(rows: Optional[Sequence[Mapping[str, Any]]], rule_version: str = RULE_VERSION) -> str:
    normalized = [_canonical_row_for_hash(row) for row in (rows or ())]
    payload = {'rule_version': str(rule_version), 'rows': normalized}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _compute_metrics(
    input_rows: Tuple[Mapping[str, Any], ...],
    deduped: Tuple[Mapping[str, Any], ...],
    rejected: Tuple[RejectedRow, ...],
) -> Dict[str, Any]:
    input_keys = set()
    before_issue_types = set()
    before_slice_pairs = set()
    invalid_rows = 0
    for row in input_rows:
        key = _key_from_row(row)
        if key is None:
            invalid_rows += 1
            continue
        input_keys.add(key)
        before_issue_types.update(_as_issue_types(row.get('issue_type')))
        for index in _as_slice_indices(row.get('slice_index')):
            before_slice_pairs.add((key, index))

    output_keys = set()
    after_issue_types = set()
    after_slice_pairs = set()
    for row in deduped:
        key = (str(row['case_id']), str(row['modality']).lower())
        output_keys.add(key)
        issue_types = row.get('issue_types', row.get('issue_type', ()))
        for issue_type in issue_types:
            after_issue_types.add(issue_type)
        slice_indices = row.get('slice_indices', row.get('slice_index', ()))
        for index in slice_indices:
            after_slice_pairs.add((key, int(index)))

    issue_retention = (
        len(before_issue_types & after_issue_types) / len(before_issue_types)
        if before_issue_types else 1.0
    )
    slice_retention = (
        len(before_slice_pairs & after_slice_pairs) / len(before_slice_pairs)
        if before_slice_pairs else 1.0
    )
    return {
        'input_rows': len(input_rows),
        'unique_input_keys': len(input_keys),
        'unique_output_keys': len(output_keys),
        'unique_key_loss': len(input_keys - output_keys),
        'duplicate_rows_rejected': sum(1 for row in rejected if row.reason == 'duplicate_case_modality'),
        'invalid_rows_rejected': invalid_rows,
        'positive_issue_type_retention': issue_retention,
        'slice_index_retention': slice_retention,
        'all_unique_after_gate': len(output_keys) == len(deduped),
    }


def apply_case_modality_write_dedup_gate(
    rows: Optional[Sequence[Mapping[str, Any]]],
    *,
    rule_version: str = RULE_VERSION,
) -> GateResult:
    input_rows = tuple(rows or ())
    deduped_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str]] = []
    rejected: List[RejectedRow] = []

    for index, row in enumerate(input_rows):
        if not isinstance(row, Mapping):
            rejected.append(RejectedRow(
                index=index,
                reason='row_not_mapping',
                case_id='',
                modality='',
                raw={'value': repr(row)},
            ))
            continue
        key = _key_from_row(row)
        if key is None:
            rejected.append(RejectedRow(
                index=index,
                reason='missing_case_id_or_modality',
                case_id=str(row.get('case_id', '')).strip(),
                modality=str(row.get('modality', '')).strip(),
                raw=dict(row),
            ))
            continue
        case_id, _modality_key = key
        modality_original = str(row.get('modality', '')).strip()
        issue_types = _as_issue_types(row.get('issue_type'))
        slice_indices = _as_slice_indices(row.get('slice_index'))
        if key not in deduped_by_key:
            deduped_by_key[key] = {
                'case_id': case_id,
                'modality': modality_original,
                'issue_type': issue_types,
                'issue_types': issue_types,
                'slice_index': slice_indices,
                'slice_indices': slice_indices,
            }
            order.append(key)
        else:
            existing = deduped_by_key[key]
            merged_issue_types = tuple(sorted(set(existing['issue_type']) | set(issue_types)))
            merged_slice_indices = tuple(sorted(set(existing['slice_index']) | set(slice_indices)))
            existing['issue_type'] = merged_issue_types
            existing['issue_types'] = merged_issue_types
            existing['slice_index'] = merged_slice_indices
            existing['slice_indices'] = merged_slice_indices
            rejected.append(RejectedRow(
                index=index,
                reason='duplicate_case_modality',
                case_id=case_id,
                modality=modality_original,
                raw=dict(row),
            ))

    deduped = tuple(deduped_by_key[key] for key in order)
    metrics = _compute_metrics(input_rows, deduped, tuple(rejected))
    return GateResult(rule_version=str(rule_version), deduped=deduped, rejected=tuple(rejected), metrics=metrics)


def validate_gate_result(
    result: GateResult,
    *,
    min_issue_type_retention: float = DEFAULT_MIN_ISSUE_TYPE_RETENTION,
) -> ValidationResult:
    failures: List[str] = []
    metrics = result.metrics
    if metrics.get('unique_key_loss') != 0:
        failures.append('unique_key_loss_nonzero')
    if metrics.get('positive_issue_type_retention', 0.0) + 1e-12 < min_issue_type_retention:
        failures.append('positive_issue_type_retention_below_threshold')
    if not metrics.get('all_unique_after_gate', False):
        failures.append('duplicate_keys_remain')
    if metrics.get('invalid_rows_rejected', 0) > 0:
        failures.append('invalid_rows_rejected')
    return ValidationResult(passed=not failures, failures=tuple(failures))


class CaseModalityWriteDedupClosure:
    '''Stateful write-side closure gate with idempotent DONE marker.'''

    def __init__(self, *, rule_version: str = RULE_VERSION) -> None:
        if not str(rule_version).strip():
            raise ValueError('rule_version must be non-empty')
        self.rule_version = str(rule_version)
        self._done_rule_version: Optional[str] = None
        self._done_fingerprint: Optional[str] = None

    @property
    def done(self) -> bool:
        return self._done_fingerprint is not None

    def reset(self) -> None:
        self._done_rule_version = None
        self._done_fingerprint = None

    def evaluate(
        self,
        rows: Optional[Sequence[Mapping[str, Any]]],
        *,
        rule_version: Optional[str] = None,
        cache_hash: Optional[str] = None,
        auto_commit: bool = True,
        min_issue_type_retention: float = DEFAULT_MIN_ISSUE_TYPE_RETENTION,
    ) -> ClosureDecision:
        version = str(rule_version or self.rule_version)
        fingerprint = str(cache_hash) if cache_hash is not None else canonical_fingerprint(rows, version)
        if (
            self._done_fingerprint is not None
            and self._done_rule_version == version
            and self._done_fingerprint == fingerprint
        ):
            return ClosureDecision(
                status='NOOP',
                rule_version=version,
                fingerprint=fingerprint,
                proposal=None,
                validation=ValidationResult(passed=True, failures=()),
                metrics={'skipped': True, 'reason': 'cache_hash_or_rule_version_unchanged'},
                deduped_rows=(),
                rejected_rows=(),
                done_marked=True,
                skipped=True,
            )

        gate_result = apply_case_modality_write_dedup_gate(rows, rule_version=version)
        validation = validate_gate_result(gate_result, min_issue_type_retention=min_issue_type_retention)
        status = 'APPLIED' if validation.passed else 'REJECTED'
        proposal: Optional[Dict[str, Any]] = None
        done_marked = False
        if validation.passed:
            proposal = {
                'action': 'case_modality_write_dedup_closure',
                'rule_version': version,
                'fingerprint': fingerprint,
                'deduped_rows': [dict(row) for row in gate_result.deduped],
                'rejected_count': len(gate_result.rejected),
            }
            if auto_commit:
                self._done_rule_version = version
                self._done_fingerprint = fingerprint
                done_marked = True
        return ClosureDecision(
            status=status,
            rule_version=version,
            fingerprint=fingerprint,
            proposal=proposal,
            validation=validation,
            metrics=gate_result.metrics,
            deduped_rows=gate_result.deduped,
            rejected_rows=gate_result.rejected,
            done_marked=done_marked,
            skipped=False,
        )
