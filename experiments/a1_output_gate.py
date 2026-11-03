"""A1 output-side hard gate: exactly one record per ``(case_id, modality)``.

Gate scope (deliberately narrow and read-only over QC payloads):

* one merged record per ``(case_id, modality)`` key;
* ``issue_type`` sets are unioned and every issue keeps its own slice indices,
  so each ``(case_id, modality, issue_type, slice_index)`` triple stays
  addressable after rollup;
* byte-identical duplicate rows are rejected outright and only counted, never
  emitted downstream;
* extractor creation is hash-gated: a still-valid cache hash forbids any new
  extractor, and only a hash mismatch admits the single ``b1_import`` extractor.

Flags, labels, rule ids and the false-repair cap are never dropped or rewritten.
When rows inside one ``(case_id, modality)`` group disagree on the false-repair
cap, only the most conservative (smallest) observed value is kept.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

KEY_FIELDS: Tuple[str, str] = ("case_id", "modality")
B1_IMPORT_EXTRACTOR: str = "b1_import"
UNATTRIBUTED_ISSUE: str = "__unattributed__"

ISSUE_FIELDS = ("issue_type", "issue_types", "issue")
FLAT_SLICE_FIELDS = ("slice_indices", "slices", "slice_index")
PER_ISSUE_SLICE_FIELDS = ("slice_indices_by_issue", "slices_by_issue")
FLAG_FIELDS = ("flags", "flag")
LABEL_FIELDS = ("labels", "label")
RULE_FIELDS = ("rule", "rule_id", "rule_ids")
CAP_FIELD = "false_repair_cap"

SEVERITY_RANKS: Mapping[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

KNOWN_FIELDS = frozenset(
    KEY_FIELDS
    + ISSUE_FIELDS
    + FLAT_SLICE_FIELDS
    + PER_ISSUE_SLICE_FIELDS
    + FLAG_FIELDS
    + LABEL_FIELDS
    + RULE_FIELDS
    + (CAP_FIELD, "severity")
)

__all__ = [
    "B1_IMPORT_EXTRACTOR",
    "CAP_FIELD",
    "KEY_FIELDS",
    "SEVERITY_RANKS",
    "UNATTRIBUTED_ISSUE",
    "ExtractorDecision",
    "ExtractorGateError",
    "HighRiskSummary",
    "MergeReport",
    "collect_positive_slice_index",
    "enforce_extractor_gate",
    "evaluate_extractor_gate",
    "merge_a1_records",
    "row_fingerprint",
    "summarize_high_risk",
]


class ExtractorGateError(RuntimeError):
    """Raised when a caller tries to bypass the extractor hash gate."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), default=str)


def row_fingerprint(row: Mapping[str, Any]) -> str:
    """Stable content hash of one A1 output row (used for duplicate rejection)."""
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _iter_strs(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        token = value.strip()
        if token:
            yield token
        return
    if isinstance(value, Mapping):
        for key in value:
            yield str(key)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield str(value)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strs(item)
        return
    yield str(value)


def _str_set(*values: Any) -> set:
    out: set = set()
    for value in values:
        out.update(_iter_strs(value))
    return out


def _iter_ints(value: Any):
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        yield int(value)
        return
    if isinstance(value, float):
        if float(value).is_integer():
            yield int(value)
        return
    if isinstance(value, str):
        token = value.strip()
        if token.lstrip("+-").isdigit():
            yield int(token)
        return
    if isinstance(value, Mapping):
        for inner in value.values():
            yield from _iter_ints(inner)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_ints(item)


def _int_set(*values: Any) -> set:
    out: set = set()
    for value in values:
        out.update(_iter_ints(value))
    return out


def _row_issue_types(row: Mapping[str, Any]) -> set:
    return _str_set(*[row.get(field) for field in ISSUE_FIELDS])


def _row_flat_slices(row: Mapping[str, Any]) -> set:
    return _int_set(*[row.get(field) for field in FLAT_SLICE_FIELDS])


def _row_slices_by_issue(row: Mapping[str, Any]) -> Dict[str, set]:
    issues = _row_issue_types(row)
    per_issue: Dict[str, set] = {}
    for field in PER_ISSUE_SLICE_FIELDS:
        raw = row.get(field)
        if isinstance(raw, Mapping):
            for issue, slices in raw.items():
                per_issue.setdefault(str(issue), set()).update(_iter_ints(slices))
    flat = _row_flat_slices(row)
    if flat:
        targets = issues or {UNATTRIBUTED_ISSUE}
        for issue in targets:
            per_issue.setdefault(issue, set()).update(flat)
    for issue in issues:
        per_issue.setdefault(issue, set())
    return per_issue


def _severity_rank(severity: Any) -> int:
    if severity is None:
        return -1
    token = str(severity).strip().lower()
    if not token:
        return -1
    return SEVERITY_RANKS.get(token, 0)


def _merge_cap(current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if current is None:
        return candidate
    try:
        return min(current, candidate)
    except TypeError:
        return current


def _empty_bucket(key: Tuple[Any, Any]) -> Dict[str, Any]:
    case_id, modality = key
    return {
        "case_id": case_id,
        "modality": modality,
        "issue_types": set(),
        "slices_by_issue": {},
        "slice_indices": set(),
        "flags": set(),
        "labels": set(),
        "rule_ids": set(),
        "severity": None,
        "false_repair_cap": None,
        "source_row_count": 0,
        "extras": {},
    }


def _absorb_row(bucket: Dict[str, Any], row: Mapping[str, Any]) -> None:
    bucket["source_row_count"] += 1
    bucket["issue_types"].update(_row_issue_types(row))
    for issue, slices in _row_slices_by_issue(row).items():
        bucket["slices_by_issue"].setdefault(issue, set()).update(slices)
    bucket["slice_indices"].update(_row_flat_slices(row))
    for field in FLAG_FIELDS:
        bucket["flags"].update(_iter_strs(row.get(field)))
    for field in LABEL_FIELDS:
        bucket["labels"].update(_iter_strs(row.get(field)))
    for field in RULE_FIELDS:
        bucket["rule_ids"].update(_iter_strs(row.get(field)))
    severity = row.get("severity")
    if _severity_rank(severity) > _severity_rank(bucket["severity"]):
        bucket["severity"] = severity
    bucket["false_repair_cap"] = _merge_cap(bucket["false_repair_cap"], row.get(CAP_FIELD))
    for key, value in row.items():
        if key in KNOWN_FIELDS:
            continue
        bucket["extras"].setdefault(key, value)


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    by_issue: Dict[str, Tuple[int, ...]] = {}
    all_slices = set(bucket["slice_indices"])
    for issue, slices in bucket["slices_by_issue"].items():
        ordered = tuple(sorted(int(item) for item in slices))
        by_issue[issue] = ordered
        all_slices.update(ordered)
    record: Dict[str, Any] = {
        "case_id": bucket["case_id"],
        "modality": bucket["modality"],
        "issue_types": tuple(sorted(bucket["issue_types"])),
        "slice_indices_by_issue": by_issue,
        "slice_indices": tuple(sorted(all_slices)),
        "flags": tuple(sorted(bucket["flags"])),
        "labels": tuple(sorted(bucket["labels"])),
        "rule_ids": tuple(sorted(bucket["rule_ids"])),
        "severity": bucket["severity"],
        "source_row_count": bucket["source_row_count"],
    }
    if bucket["false_repair_cap"] is not None:
        record[CAP_FIELD] = bucket["false_repair_cap"]
    record.update(bucket["extras"])
    return record


def collect_positive_slice_index(records) -> Tuple[Tuple[Any, Any, str, int], ...]:
    """Every addressable ``(case_id, modality, issue_type, slice_index)`` triple."""
    triples = set()
    for record in records:
        by_issue = record.get("slice_indices_by_issue") or {}
        for issue, slices in by_issue.items():
            for slice_index in slices or ():
                triples.add(
                    (
                        record.get("case_id"),
                        record.get("modality"),
                        str(issue),
                        int(slice_index),
                    )
                )
    return tuple(
        sorted(
            triples,
            key=lambda item: (str(item[0]), str(item[1]), str(item[2]), int(item[3])),
        )
    )


@dataclass(frozen=True)
class MergeReport:
    records: Tuple[Mapping[str, Any], ...]
    input_row_count: int
    duplicate_rows_rejected: int
    merged_group_count: int
    payload_chars_before: int
    payload_chars_after: int
    positive_slice_index: Tuple[Tuple[Any, Any, str, int], ...]

    @property
    def row_count(self) -> int:
        return len(self.records)

    @property
    def row_reduction(self) -> int:
        return self.input_row_count - len(self.records)

    @property
    def unique_key_rate(self) -> float:
        if not self.records:
            return 1.0
        keys = {(record.get("case_id"), record.get("modality")) for record in self.records}
        return len(keys) / float(len(self.records))

    @property
    def token_reduction_ratio(self) -> float:
        if self.payload_chars_before <= 0:
            return 1.0
        return 1.0 - (float(self.payload_chars_after) / float(self.payload_chars_before))


def merge_a1_records(rows) -> MergeReport:
    """Consolidate A1 output rows into one record per ``(case_id, modality)``.

    Byte-identical rows are rejected outright and counted in
    ``duplicate_rows_rejected``; distinct rows sharing a key are merged so that
    no issue type, flag, label, rule id or slice index is lost.
    """
    rows = list(rows)
    seen_fingerprints = set()
    buckets: "OrderedDict[Tuple[Any, Any], Dict[str, Any]]" = OrderedDict()
    duplicates = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(
                "A1 rows must be mappings, got {0!r}".format(type(row).__name__)
            )
        fingerprint = row_fingerprint(row)
        if fingerprint in seen_fingerprints:
            duplicates += 1
            continue
        seen_fingerprints.add(fingerprint)
        key = (row.get(KEY_FIELDS[0]), row.get(KEY_FIELDS[1]))
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _empty_bucket(key)
            buckets[key] = bucket
        _absorb_row(bucket, row)
    records = tuple(_finalize_bucket(bucket) for bucket in buckets.values())
    before_chars = sum(len(_canonical_json(row)) for row in rows)
    after_chars = sum(len(_canonical_json(record)) for record in records)
    return MergeReport(
        records=records,
        input_row_count=len(rows),
        duplicate_rows_rejected=duplicates,
        merged_group_count=len(buckets),
        payload_chars_before=before_chars,
        payload_chars_after=after_chars,
        positive_slice_index=collect_positive_slice_index(records),
    )


@dataclass(frozen=True)
class HighRiskSummary:
    total_records: int
    positive_records: int
    positive_rate: float
    issue_counts: Mapping[str, int]
    issue_slice_counts: Mapping[str, int]
    flag_counts: Mapping[str, int]
    severity_counts: Mapping[str, int]
    worst_slice_index: Optional[int]
    worst_slice_index_by_severity: Mapping[str, Optional[int]]
    positive_slice_index: Tuple[Tuple[Any, Any, str, int], ...]


def summarize_high_risk(records) -> HighRiskSummary:
    """Count-level high-risk rollup that keeps every positive slice addressable."""
    records = list(records)
    issue_counts: Counter = Counter()
    issue_slice_counts: Counter = Counter()
    flag_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    worst_slice_index: Optional[int] = None
    worst_by_severity: Dict[str, Optional[int]] = {}
    positive_records = 0
    for record in records:
        by_issue = record.get("slice_indices_by_issue") or {}
        positives = {
            str(issue): tuple(slices or ())
            for issue, slices in by_issue.items()
            if slices
        }
        if positives:
            positive_records += 1
        severity = record.get("severity") or "unrated"
        severity = str(severity)
        severity_counts[severity] += 1
        for flag in record.get("flags") or ():
            flag_counts[str(flag)] += 1
        for issue, slices in positives.items():
            issue_counts[issue] += 1
            issue_slice_counts[issue] += len(slices)
            for slice_index in slices:
                slice_index = int(slice_index)
                if worst_slice_index is None or slice_index > worst_slice_index:
                    worst_slice_index = slice_index
                current = worst_by_severity.get(severity)
                if current is None or slice_index > current:
                    worst_by_severity[severity] = slice_index
    total = len(records)
    return HighRiskSummary(
        total_records=total,
        positive_records=positive_records,
        positive_rate=(float(positive_records) / float(total)) if total else 0.0,
        issue_counts=dict(issue_counts),
        issue_slice_counts=dict(issue_slice_counts),
        flag_counts=dict(flag_counts),
        severity_counts=dict(severity_counts),
        worst_slice_index=worst_slice_index,
        worst_slice_index_by_severity=dict(worst_by_severity),
        positive_slice_index=collect_positive_slice_index(records),
    )


@dataclass(frozen=True)
class ExtractorDecision:
    allowed: bool
    reason: str
    extractor_kind: Optional[str] = None


def cache_hash_matches(cache_hash: Any, expected_hash: Any) -> bool:
    if expected_hash in (None, ""):
        return False
    if cache_hash in (None, ""):
        return False
    return str(cache_hash).strip() == str(expected_hash).strip()


def evaluate_extractor_gate(
    *,
    cache_hash: Any,
    expected_hash: Any,
    requested_extractor: Any = None,
    cached_extractors=(),
) -> ExtractorDecision:
    """Decide whether a new extractor may be created for this payload.

    A valid cache hash forbids every new extractor. Only a hash mismatch admits
    the single ``b1_import`` extractor kind, and only if it is not already
    cached.
    """
    if expected_hash in (None, ""):
        return ExtractorDecision(False, "missing_expected_hash", None)
    if cache_hash_matches(cache_hash, expected_hash):
        return ExtractorDecision(False, "cache_hash_valid", None)
    if requested_extractor in (None, ""):
        return ExtractorDecision(False, "no_extractor_requested", None)
    if str(requested_extractor) != B1_IMPORT_EXTRACTOR:
        return ExtractorDecision(False, "extractor_not_allowed_on_hash_mismatch", None)
    existing = {str(item) for item in (cached_extractors or ())}
    if B1_IMPORT_EXTRACTOR in existing:
        return ExtractorDecision(False, "b1_import_extractor_already_cached", None)
    return ExtractorDecision(True, "hash_mismatch_b1_import_only", B1_IMPORT_EXTRACTOR)


def enforce_extractor_gate(**kwargs) -> ExtractorDecision:
    decision = evaluate_extractor_gate(**kwargs)
    if not decision.allowed:
        raise ExtractorGateError(
            "extractor gate refused creation: {0}".format(decision.reason)
        )
    return decision
