"""Write-side hard gate enforcing one row per ``(case_id, modality)``.

This module is the write-side counterpart of the existing case/modality dedup
experiments.  It is deliberately side-effect free: it never writes label or
rule payloads, never touches the datasets, never reaches the network, never
mutates the caller's rows, and never changes any segmentation or dice metric.
It only computes the canonical, deduplicated set of rows an isolated pipeline
is allowed to persist and reports what it rejected.

Contract
--------
* One record per ``(case_id, modality)``.
* The first row for a key is accepted.  Every later row with the same key is a
  *duplicate*: it is rejected as a row of its own, but its ``issue_type``
  values and slice indices are folded into the surviving record so no evidence
  is lost (issue_type sets are merged, slice indices stay complete).
* The duplicate rejection rate (rejected duplicate rows / duplicate rows) must
  stay at or above 95%.
* When the cache hash is valid no new extractor may be created; a stale or
  missing hash takes the explicit fallback path instead.  A mismatching cache
  hash makes ``ingest`` fall back without committing anything.
* Once validation passes the gate is marked ``DONE`` and emits no further
  proposals; later ingest calls become no-ops.
* Label and rule write counters must stay at zero.

Note for dynamic loaders: this module uses ``from __future__ import
annotations``, so CPython's ``dataclasses`` resolves the annotation strings
through ``sys.modules[cls.__module__]``.  A loader that builds the module with
``importlib.util.module_from_spec`` must register it in ``sys.modules`` before
calling ``exec_module``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CaseModalityWriteDedupGate",
    "DedupRecord",
    "ExtractorDecision",
    "IngestReport",
    "ValidationReport",
    "WriteGateError",
    "MIN_DUPLICATE_REJECTION_RATE",
    "STATUS_DONE",
    "STATUS_PENDING",
    "compute_cache_hash",
    "normalize_row",
]

STATUS_PENDING = "PENDING"
STATUS_DONE = "DONE"

MIN_DUPLICATE_REJECTION_RATE = 0.95

_ISSUE_TYPE_KEYS: Sequence[str] = ("issue_types", "issue_type", "issues")
_SLICE_KEYS: Sequence[str] = (
    "slice_indices",
    "slice_index",
    "slices",
    "slice_idx",
)


class WriteGateError(ValueError):
    """Raised when the gate is asked to do something it cannot do safely."""


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #
def compute_cache_hash(payload: Any) -> str:
    """Return a stable sha256 digest for ``payload`` (order insensitive)."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten_values(row: Mapping[str, Any], keys: Sequence[str]) -> list:
    values: list = []
    for name in keys:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            values.extend(value)
        else:
            values.append(value)
    return values


def _normalize_issue_types(values: Iterable[Any]) -> tuple:
    out: list = []
    seen: set = set()
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(sorted(out))


def _normalize_slice_indices(values: Iterable[Any]) -> tuple:
    out: set = set()
    for item in values:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.add(item)
            continue
        if isinstance(item, float):
            if item.is_integer():
                out.add(int(item))
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                out.add(int(text))
            except ValueError:
                continue
            continue
    return tuple(sorted(out))


@dataclass(frozen=True)
class DedupRecord:
    """One canonical row per ``(case_id, modality)``."""

    case_id: str
    modality: str
    issue_types: tuple = ()
    slice_indices: tuple = ()

    @property
    def key(self) -> tuple:
        return (self.case_id, self.modality)

    def merged_with(self, other: "DedupRecord") -> "DedupRecord":
        """Union issue_type sets and slice indices of the same key."""

        if not isinstance(other, DedupRecord):
            raise WriteGateError("merged_with expects a DedupRecord")
        if other.key != self.key:
            raise WriteGateError(f"cannot merge {other.key!r} into {self.key!r}")
        return DedupRecord(
            case_id=self.case_id,
            modality=self.modality,
            issue_types=tuple(sorted(set(self.issue_types) | set(other.issue_types))),
            slice_indices=tuple(sorted(set(self.slice_indices) | set(other.slice_indices))),
        )

    def has_complete_evidence(self) -> bool:
        return bool(self.issue_types) and bool(self.slice_indices)

    def as_proposal(self) -> dict:
        return {
            "case_id": self.case_id,
            "modality": self.modality,
            "issue_types": list(self.issue_types),
            "slice_indices": list(self.slice_indices),
        }

    def to_dict(self) -> dict:
        return self.as_proposal()


def normalize_row(row: Any) -> DedupRecord:
    """Normalise one incoming row into a :class:`DedupRecord`.

    Accepts both the singular (``issue_type`` / ``slice_index``) and the plural
    (``issue_types`` / ``slice_indices``) spellings and unions them.
    """

    if isinstance(row, DedupRecord):
        return DedupRecord(
            case_id=row.case_id,
            modality=row.modality,
            issue_types=_normalize_issue_types(row.issue_types),
            slice_indices=_normalize_slice_indices(row.slice_indices),
        )
    if not isinstance(row, Mapping):
        raise WriteGateError(f"row must be a mapping, got {type(row).__name__}")

    case_id = row.get("case_id")
    modality = row.get("modality")
    if case_id is None or not str(case_id).strip():
        raise WriteGateError("row is missing a non-empty case_id")
    if modality is None or not str(modality).strip():
        raise WriteGateError("row is missing a non-empty modality")

    return DedupRecord(
        case_id=str(case_id).strip(),
        modality=str(modality).strip(),
        issue_types=_normalize_issue_types(_flatten_values(row, _ISSUE_TYPE_KEYS)),
        slice_indices=_normalize_slice_indices(_flatten_values(row, _SLICE_KEYS)),
    )


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtractorDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class IngestReport:
    total_rows: int
    accepted_rows: int
    duplicate_rows: int
    rejected_duplicate_rows: int
    merged_rows: int
    new_proposals: int
    duplicate_rejection_rate: float
    cache_fallback: bool = False
    no_op: bool = False
    record_count: int = 0


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    status: str
    duplicate_rejection_rate: float
    record_count: int
    problems: tuple = ()


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
class CaseModalityWriteDedupGate:
    """Idempotent write-side dedup gate for isolated QC pipelines."""

    def __init__(
        self,
        cache_hash: str | None = None,
        min_duplicate_rejection_rate: float = MIN_DUPLICATE_REJECTION_RATE,
    ) -> None:
        rate = float(min_duplicate_rejection_rate)
        if not 0.0 <= rate <= 1.0:
            raise WriteGateError("min_duplicate_rejection_rate must be within [0, 1]")
        self._min_rate = rate
        self._cache_hash = str(cache_hash) if cache_hash else None
        self._records: dict = {}
        self._status = STATUS_PENDING
        # Counters that must stay at zero: the gate never writes labels/rules.
        self._label_writes = 0
        self._rule_writes = 0
        self._total_rows = 0
        self._accepted_rows = 0
        self._duplicate_rows = 0
        self._rejected_duplicate_rows = 0
        self._merged_rows = 0
        self._new_proposals = 0

    # -- introspection ------------------------------------------------------ #
    @property
    def status(self) -> str:
        return self._status

    @property
    def is_done(self) -> bool:
        return self._status == STATUS_DONE

    @property
    def cache_hash(self) -> str | None:
        return self._cache_hash

    @property
    def min_duplicate_rejection_rate(self) -> float:
        return self._min_rate

    @property
    def label_writes(self) -> int:
        return self._label_writes

    @property
    def rule_writes(self) -> int:
        return self._rule_writes

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def accepted_rows(self) -> int:
        return self._accepted_rows

    @property
    def duplicate_rows(self) -> int:
        return self._duplicate_rows

    @property
    def rejected_duplicate_rows(self) -> int:
        return self._rejected_duplicate_rows

    @property
    def merged_rows(self) -> int:
        return self._merged_rows

    @property
    def new_proposals(self) -> int:
        return self._new_proposals

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def duplicate_rejection_rate(self) -> float:
        if not self._duplicate_rows:
            return 1.0
        return self._rejected_duplicate_rows / self._duplicate_rows

    def records(self) -> tuple:
        return tuple(self._records[key] for key in sorted(self._records))

    def state_snapshot(self) -> dict:
        return {
            "status": self._status,
            "cache_hash": self._cache_hash,
            "records": [record.to_dict() for record in self.records()],
            "counters": {
                "total_rows": self._total_rows,
                "accepted_rows": self._accepted_rows,
                "duplicate_rows": self._duplicate_rows,
                "rejected_duplicate_rows": self._rejected_duplicate_rows,
                "merged_rows": self._merged_rows,
                "new_proposals": self._new_proposals,
                "label_writes": self._label_writes,
                "rule_writes": self._rule_writes,
            },
        }

    # -- cache hash --------------------------------------------------------- #
    def set_cache_hash(self, value: Any) -> str:
        """Register the cache hash; accepts a raw hash string or a payload."""

        if isinstance(value, str):
            self._cache_hash = value
        else:
            self._cache_hash = compute_cache_hash(value)
        return self._cache_hash

    def cache_hash_valid(self, candidate: Any) -> bool:
        if self._cache_hash is None or candidate is None:
            return False
        return str(candidate) == self._cache_hash

    def request_extractor(self, cache_hash: Any = None) -> ExtractorDecision:
        """New extractors are forbidden while the cache hash is valid."""

        if self._status == STATUS_DONE:
            return ExtractorDecision(False, "status_done")
        if self.cache_hash_valid(cache_hash):
            return ExtractorDecision(False, "cache_hash_valid")
        return ExtractorDecision(True, "cache_hash_missing_or_stale")

    # -- ingestion ---------------------------------------------------------- #
    def _noop_report(self, row_count: int, *, cache_fallback: bool) -> IngestReport:
        return IngestReport(
            total_rows=row_count,
            accepted_rows=0,
            duplicate_rows=0,
            rejected_duplicate_rows=0,
            merged_rows=0,
            new_proposals=0,
            duplicate_rejection_rate=self.duplicate_rejection_rate,
            cache_fallback=cache_fallback,
            no_op=True,
            record_count=len(self._records),
        )

    def ingest(self, rows: Iterable[Any], cache_hash: Any = None) -> IngestReport:
        """Commit rows through the hard gate and report what was rejected.

        The commit is staged: if the supplied cache hash does not match the
        registered hash the gate falls back and leaves the state untouched.
        """

        materialised = list(rows)

        if self._status == STATUS_DONE:
            return self._noop_report(len(materialised), cache_fallback=False)

        if (
            cache_hash is not None
            and self._cache_hash is not None
            and str(cache_hash) != self._cache_hash
        ):
            return self._noop_report(len(materialised), cache_fallback=True)

        staged = dict(self._records)
        accepted = duplicates = rejected = merged = proposals = 0

        for raw in materialised:
            record = normalize_row(raw)
            existing = staged.get(record.key)
            if existing is None:
                staged[record.key] = record
                accepted += 1
                proposals += 1
                continue

            # Duplicate row: rejected as a row of its own, evidence merged.
            duplicates += 1
            rejected += 1
            combined = existing.merged_with(record)
            if combined != existing:
                staged[record.key] = combined
                merged += 1
                proposals += 1

        self._records = staged
        self._total_rows += len(materialised)
        self._accepted_rows += accepted
        self._duplicate_rows += duplicates
        self._rejected_duplicate_rows += rejected
        self._merged_rows += merged
        self._new_proposals += proposals

        rate = rejected / duplicates if duplicates else 1.0
        return IngestReport(
            total_rows=len(materialised),
            accepted_rows=accepted,
            duplicate_rows=duplicates,
            rejected_duplicate_rows=rejected,
            merged_rows=merged,
            new_proposals=proposals,
            duplicate_rejection_rate=rate,
            cache_fallback=False,
            no_op=(accepted == 0 and merged == 0),
            record_count=len(staged),
        )

    def _clone(self) -> "CaseModalityWriteDedupGate":
        clone = CaseModalityWriteDedupGate(
            cache_hash=self._cache_hash,
            min_duplicate_rejection_rate=self._min_rate,
        )
        clone._records = dict(self._records)
        clone._status = self._status
        clone._label_writes = self._label_writes
        clone._rule_writes = self._rule_writes
        clone._total_rows = self._total_rows
        clone._accepted_rows = self._accepted_rows
        clone._duplicate_rows = self._duplicate_rows
        clone._rejected_duplicate_rows = self._rejected_duplicate_rows
        clone._merged_rows = self._merged_rows
        clone._new_proposals = self._new_proposals
        return clone

    def dry_run(self, rows: Iterable[Any], cache_hash: Any = None) -> IngestReport:
        """Return what ``ingest`` would do without mutating this gate."""

        return self._clone().ingest(rows, cache_hash=cache_hash)

    # -- proposals ---------------------------------------------------------- #
    def propose(self) -> tuple:
        """Emit one proposal per canonical record; nothing once DONE."""

        if self._status == STATUS_DONE:
            return ()
        return tuple(record.as_proposal() for record in self.records())

    # -- validation --------------------------------------------------------- #
    def validate(self) -> ValidationReport:
        problems: list = []

        for record in self.records():
            if not record.issue_types:
                problems.append(
                    f"missing issue_types for {record.case_id}/{record.modality}"
                )
            if not record.slice_indices:
                problems.append(
                    f"missing slice_indices for {record.case_id}/{record.modality}"
                )
            elif list(record.slice_indices) != sorted(set(record.slice_indices)):
                problems.append(
                    f"slice indices not canonical for {record.case_id}/{record.modality}"
                )

        if not self._records:
            problems.append("no accepted records to validate")

        if self._duplicate_rows and self.duplicate_rejection_rate < self._min_rate:
            problems.append(
                "duplicate rejection rate "
                f"{self.duplicate_rejection_rate:.4f} below {self._min_rate:.4f}"
            )

        if self._label_writes or self._rule_writes:
            problems.append("safety violation: label/rule writes must stay at zero")

        passed = not problems
        if passed:
            self._status = STATUS_DONE

        return ValidationReport(
            passed=passed,
            status=self._status,
            duplicate_rejection_rate=self.duplicate_rejection_rate,
            record_count=len(self._records),
            problems=tuple(problems),
        )
