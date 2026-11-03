"""A1 output-side write-time hard gate (isolated experiment).

Approved maintenance action (owner: ``efficiency_auditor``)
----------------------------------------------------------
Enforce the A1 output contract at write time:

1. exactly one record per ``(case_id, modality)``;
2. ``issue_type`` sets are merged and per-slice indices are preserved;
3. redundant / duplicate rows are rejected outright instead of being written;
4. when a valid cache hash already exists, creating a *new* extractor is
   forbidden -- the cached extractor must be reused (an explicit, audited
   ``invalidate`` call is the only way to unblock creation);
5. the result is emitted as a single case-level handoff artifact.

The module is offline and side-effect free: it never touches production
stores, datasets, patient images, labels or credentials.  ``apply_write_gate``
only writes into a caller supplied in-memory mapping / callback and can never
emit more than one record per identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

SCHEMA_VERSION = "a1-case-handoff-v1"
DEFAULT_MIN_STEP_REDUCTION = 0.30
LEGACY_STEPS_PER_ROW = 3  # extractor call + handoff step + write step per raw row

SEVERITY_RANK: Dict[str, int] = {
    "info": 0,
    "low": 0,
    "warn": 1,
    "warning": 1,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MIN_STEP_REDUCTION",
    "GateError",
    "InvalidRowError",
    "DuplicateRowError",
    "RecallRegression",
    "ExtractorCreationBlocked",
    "A1Row",
    "CaseModalityRecord",
    "RejectedRow",
    "GateResult",
    "CacheEntry",
    "ExtractorHandle",
    "ExtractorReuseGuard",
    "EffortLedger",
    "GateRun",
    "apply_hard_gate",
    "build_case_handoff",
    "handoff_to_json",
    "validate_handoff",
    "duplicate_record_rate",
    "legacy_facts",
    "recall_coverage",
    "ensure_recall_not_decreased",
    "effort_ledger",
    "apply_write_gate",
    "run_a1_output_gate",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class GateError(ValueError):
    """Base class for A1 output hard-gate failures."""


class InvalidRowError(GateError):
    """A raw A1 row cannot be normalised."""


class DuplicateRowError(GateError):
    """A duplicate row reached the write layer."""


class RecallRegression(GateError):
    """Merging dropped an issue/slice fact the legacy audit reported."""


class ExtractorCreationBlocked(GateError):
    """A new extractor was requested while a valid cache hash exists."""


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------
_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "case_id": ("case_id", "caseId", "case"),
    "modality": ("modality", "mod", "modality_name"),
    "issue_types": ("issue_types", "issue_type", "issues"),
    "slice_indices": ("slice_indices", "slice_index", "slices", "slice"),
    "sources": ("sources", "source"),
    "severity": ("severity", "level", "risk"),
}


def _pick(raw: Mapping[str, Any], canonical: str) -> Any:
    for alias in _FIELD_ALIASES[canonical]:
        if alias in raw and raw[alias] is not None:
            return raw[alias]
    return None


def _as_text(value: Any, field_name: str) -> str:
    if value is None:
        raise InvalidRowError(f"{field_name} is required and must be a non-empty string")
    if isinstance(value, bool):
        raise InvalidRowError(f"{field_name} must be a non-empty string, got bool")
    text = str(value).strip()
    if not text:
        raise InvalidRowError(f"{field_name} must be a non-empty string")
    return text


def _norm_text_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Iterable[Any] = (value,)
    else:
        try:
            candidates = tuple(value)
        except TypeError:
            raise InvalidRowError(f"{field_name} must be a string or an iterable of strings") from None
    seen: List[str] = []
    for item in candidates:
        if item is None or isinstance(item, bool):
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return tuple(sorted(seen))


def _norm_slice_indices(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise InvalidRowError("slice index must be an integer, got bool")
    if isinstance(value, int):
        candidates: Iterable[Any] = (value,)
    elif isinstance(value, str):
        candidates = tuple(part for part in value.replace(";", ",").split(",") if part.strip())
    else:
        try:
            candidates = tuple(value)
        except TypeError:
            raise InvalidRowError(
                "slice_indices must be an int, a comma separated string or an iterable of ints"
            ) from None
    resolved: Set[int] = set()
    for item in candidates:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                index = int(text)
            except ValueError:
                raise InvalidRowError(f"slice index {item!r} is not an integer") from None
        else:
            raise InvalidRowError(f"slice index {item!r} is not an integer")
        if index < 0:
            raise InvalidRowError(f"slice index {index} must be >= 0")
        resolved.add(index)
    return tuple(sorted(resolved))


def _norm_severity(value: Any) -> str:
    if value is None:
        return "info"
    text = str(value).strip().lower()
    return text or "info"


def _normalise_identity(key: Any) -> Tuple[str, str]:
    if isinstance(key, A1Row):
        return key.identity
    if isinstance(key, Mapping):
        return (
            _as_text(_pick(key, "case_id"), "case_id"),
            _as_text(_pick(key, "modality"), "modality"),
        )
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return (_as_text(key[0], "case_id"), _as_text(key[1], "modality"))
    raise GateError(f"cannot interpret {key!r} as a (case_id, modality) identity")


def _default_extractor_id(identity: Tuple[str, str]) -> str:
    return f"cached::{identity[0]}::{identity[1]}"


# ---------------------------------------------------------------------------
# rows and records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class A1Row:
    """A single normalised A1 output row (one audit observation)."""

    case_id: str
    modality: str
    issue_types: Tuple[str, ...] = ()
    slice_indices: Tuple[int, ...] = ()
    sources: Tuple[str, ...] = ()
    severity: str = "info"

    @classmethod
    def from_mapping(cls, raw: Any) -> "A1Row":
        if isinstance(raw, A1Row):
            return raw
        if not isinstance(raw, Mapping):
            raise InvalidRowError(f"row must be a mapping or A1Row, got {type(raw).__name__}")
        return cls(
            case_id=_as_text(_pick(raw, "case_id"), "case_id"),
            modality=_as_text(_pick(raw, "modality"), "modality"),
            issue_types=_norm_text_tuple(_pick(raw, "issue_types"), "issue_types"),
            slice_indices=_norm_slice_indices(_pick(raw, "slice_indices")),
            sources=_norm_text_tuple(_pick(raw, "sources"), "sources"),
            severity=_norm_severity(_pick(raw, "severity")),
        )

    @property
    def identity(self) -> Tuple[str, str]:
        return (self.case_id, self.modality)

    @property
    def payload(self) -> Tuple[str, str, Tuple[str, ...], Tuple[int, ...]]:
        return (self.case_id, self.modality, self.issue_types, self.slice_indices)

    def covers(self, other: "A1Row") -> bool:
        """True when this row already carries everything ``other`` reports."""
        return (
            self.identity == other.identity
            and set(other.issue_types).issubset(self.issue_types)
            and set(other.slice_indices).issubset(self.slice_indices)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "modality": self.modality,
            "issue_types": list(self.issue_types),
            "slice_indices": list(self.slice_indices),
            "sources": list(self.sources),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class CaseModalityRecord:
    """The single merged record kept for one ``(case_id, modality)``."""

    case_id: str
    modality: str
    issue_types: Tuple[str, ...] = ()
    slice_indices: Tuple[int, ...] = ()
    sources: Tuple[str, ...] = ()
    severity: str = "info"
    row_count: int = 1

    @property
    def identity(self) -> Tuple[str, str]:
        return (self.case_id, self.modality)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "modality": self.modality,
            "issue_types": list(self.issue_types),
            "slice_indices": list(self.slice_indices),
            "sources": list(self.sources),
            "severity": self.severity,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class RejectedRow:
    row: A1Row
    reason: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"row": self.row.as_dict(), "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class GateResult:
    records: Tuple[CaseModalityRecord, ...]
    rejected: Tuple[RejectedRow, ...] = ()
    input_rows: int = 0
    absorbed_rows: int = 0

    @property
    def output_rows(self) -> int:
        return len(self.records)

    @property
    def duplicate_rows(self) -> int:
        return sum(1 for item in self.rejected if item.reason == "duplicate_row")

    @property
    def duplicate_record_rate(self) -> float:
        return duplicate_record_rate(self.records)

    def stats(self) -> Dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "absorbed_rows": self.absorbed_rows,
            "rejected_rows": len(self.rejected),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_record_rate": self.duplicate_record_rate,
        }


def duplicate_record_rate(records: Iterable[CaseModalityRecord]) -> float:
    """Share of output records that repeat an already seen (case_id, modality)."""
    materialised = list(records)
    if not materialised:
        return 0.0
    seen: Set[Tuple[str, str]] = set()
    duplicates = 0
    for record in materialised:
        key = record.identity
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates / len(materialised)


# ---------------------------------------------------------------------------
# the hard gate
# ---------------------------------------------------------------------------
def apply_hard_gate(rows: Iterable[Any], *, reject_redundant: bool = True) -> GateResult:
    """Collapse rows to one record per (case_id, modality) and reject duplicates."""
    state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rejections: List[RejectedRow] = []
    input_rows = 0
    absorbed = 0

    for raw in rows:
        row = A1Row.from_mapping(raw)
        input_rows += 1
        entry = state.get(row.identity)
        if entry is None:
            state[row.identity] = {
                "issue_types": set(row.issue_types),
                "slice_indices": set(row.slice_indices),
                "sources": set(row.sources),
                "severity": row.severity,
                "row_count": 1,
            }
            continue

        covered = set(row.issue_types).issubset(entry["issue_types"]) and set(
            row.slice_indices
        ).issubset(entry["slice_indices"])
        if covered and reject_redundant:
            rejections.append(
                RejectedRow(
                    row=row,
                    reason="duplicate_row",
                    detail=f"payload for {row.identity} is already covered by the kept record",
                )
            )
            continue

        entry["issue_types"].update(row.issue_types)
        entry["slice_indices"].update(row.slice_indices)
        entry["sources"].update(row.sources)
        entry["severity"] = max(
            entry["severity"], row.severity, key=lambda name: SEVERITY_RANK.get(name, 0)
        )
        entry["row_count"] += 1
        absorbed += 1

    records = tuple(
        CaseModalityRecord(
            case_id=identity[0],
            modality=identity[1],
            issue_types=tuple(sorted(payload["issue_types"])),
            slice_indices=tuple(sorted(payload["slice_indices"])),
            sources=tuple(sorted(payload["sources"])),
            severity=payload["severity"],
            row_count=payload["row_count"],
        )
        for identity, payload in sorted(state.items())
    )
    return GateResult(
        records=records,
        rejected=tuple(rejections),
        input_rows=input_rows,
        absorbed_rows=absorbed,
    )


# ---------------------------------------------------------------------------
# case-level handoff artifact
# ---------------------------------------------------------------------------
def build_case_handoff(
    result: GateResult, *, generated_by: str = "efficiency_auditor"
) -> Dict[str, Any]:
    """Build the single case-level handoff artifact for a gate result."""
    per_case: Dict[str, List[CaseModalityRecord]] = {}
    for record in result.records:
        per_case.setdefault(record.case_id, []).append(record)

    cases: List[Dict[str, Any]] = []
    for case_id in sorted(per_case):
        records = sorted(per_case[case_id], key=lambda item: item.modality)
        cases.append(
            {
                "case_id": case_id,
                "record_count": len(records),
                "issue_types": sorted({issue for record in records for issue in record.issue_types}),
                "slice_indices": sorted({index for record in records for index in record.slice_indices}),
                "modalities": [record.as_dict() for record in records],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "a1_case_level_handoff",
        "generated_by": generated_by,
        "case_count": len(cases),
        "record_count": len(result.records),
        "duplicate_record_rate": duplicate_record_rate(result.records),
        "rejected_row_count": len(result.rejected),
        "cases": cases,
    }


def handoff_to_json(handoff: Mapping[str, Any]) -> str:
    return json.dumps(handoff, sort_keys=True, indent=2)


def validate_handoff(handoff: Mapping[str, Any]) -> List[str]:
    """Return the list of invariant violations of a handoff artifact."""
    problems: List[str] = []
    if not isinstance(handoff, Mapping):
        return [f"handoff must be a mapping, got {type(handoff).__name__}"]
    if handoff.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION!r}")

    seen: Set[Tuple[str, str]] = set()
    record_count = 0
    for case in handoff.get("cases", []):
        case_id = case.get("case_id")
        modalities = case.get("modalities", [])
        for modality in modalities:
            record_count += 1
            key = (case_id, modality.get("modality"))
            if key in seen:
                problems.append(f"duplicate record for {key} in handoff")
            seen.add(key)
        if case.get("record_count") != len(modalities):
            problems.append(f"record_count mismatch for case {case_id!r}")

    if handoff.get("record_count") != record_count:
        problems.append("top level record_count does not match the emitted modality records")
    if handoff.get("case_count") != len(handoff.get("cases", [])):
        problems.append("top level case_count does not match the emitted cases")
    if float(handoff.get("duplicate_record_rate", 1.0)) != 0.0:
        problems.append("duplicate_record_rate must be 0.0 on the output side")
    return problems


# ---------------------------------------------------------------------------
# recall preservation versus the historical per-issue-type audit
# ---------------------------------------------------------------------------
Fact = Tuple[str, str, str, Optional[int]]


def legacy_facts(rows: Iterable[Any]) -> Set[Fact]:
    """(case_id, modality, issue_type, slice_index) tuples of the legacy audit."""
    facts: Set[Fact] = set()
    for raw in rows:
        row = A1Row.from_mapping(raw)
        if not row.issue_types:
            continue
        indices: Tuple[Optional[int], ...] = tuple(row.slice_indices) or (None,)
        for issue in row.issue_types:
            for index in indices:
                facts.add((row.case_id, row.modality, issue, index))
    return facts


def _uncovered_facts(
    facts: Iterable[Fact], records: Iterable[CaseModalityRecord]
) -> Set[Fact]:
    lookup: Dict[Tuple[str, str], CaseModalityRecord] = {}
    for record in records:
        lookup.setdefault(record.identity, record)
    missing: Set[Fact] = set()
    for fact in facts:
        case_id, modality, issue, index = fact
        record = lookup.get((case_id, modality))
        if record is None or issue not in record.issue_types:
            missing.add(fact)
        elif index is not None and index not in record.slice_indices:
            missing.add(fact)
    return missing


def recall_coverage(
    facts: Iterable[Fact], records: Iterable[CaseModalityRecord]
) -> float:
    materialised = set(facts)
    if not materialised:
        return 1.0
    missing = _uncovered_facts(materialised, records)
    return (len(materialised) - len(missing)) / len(materialised)


def ensure_recall_not_decreased(
    facts: Iterable[Fact],
    records: Iterable[CaseModalityRecord],
    *,
    minimum: float = 1.0,
) -> float:
    materialised = set(facts)
    coverage = recall_coverage(materialised, records)
    if coverage < minimum:
        missing = sorted(_uncovered_facts(materialised, records))[:5]
        raise RecallRegression(
            f"recall regression: coverage={coverage:.4f} < {minimum:.4f}; missing examples={missing}"
        )
    return coverage


# ---------------------------------------------------------------------------
# extractor reuse / cache hash gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheEntry:
    key: Tuple[str, str]
    cache_hash: str
    extractor_id: str
    valid: bool = True


@dataclass(frozen=True)
class ExtractorHandle:
    key: Tuple[str, str]
    extractor_id: str
    reused: bool
    cache_hash: Optional[str] = None


class ExtractorReuseGuard:
    """Forbid new extractors while a valid cache hash exists."""

    def __init__(self, cache: Optional[Mapping[Any, Any]] = None) -> None:
        self._cache: Dict[Tuple[str, str], CacheEntry] = {}
        self.extractor_calls = 0
        self.cache_reuses = 0
        self.blocked_creations = 0
        self.invalidations: List[Tuple[Tuple[str, str], str, str]] = []
        self.conflicts: List[Tuple[Tuple[str, str], str, str]] = []
        for key, spec in (cache or {}).items():
            self.register_cache(key, spec)

    # -- registration ---------------------------------------------------
    @staticmethod
    def is_valid_hash(cache_hash: Any) -> bool:
        return isinstance(cache_hash, str) and bool(cache_hash.strip())

    def register_cache(
        self,
        key: Any,
        spec: Any,
        extractor_id: Optional[str] = None,
        valid: bool = True,
    ) -> CacheEntry:
        identity = _normalise_identity(key)
        if isinstance(spec, CacheEntry):
            entry = CacheEntry(
                key=identity,
                cache_hash=spec.cache_hash,
                extractor_id=spec.extractor_id or _default_extractor_id(identity),
                valid=bool(spec.valid),
            )
        elif isinstance(spec, Mapping):
            entry = CacheEntry(
                key=identity,
                cache_hash=str(spec.get("cache_hash") or spec.get("hash") or ""),
                extractor_id=str(spec.get("extractor_id") or _default_extractor_id(identity)),
                valid=bool(spec.get("valid", True)),
            )
        elif isinstance(spec, str):
            entry = CacheEntry(
                key=identity,
                cache_hash=spec,
                extractor_id=extractor_id or _default_extractor_id(identity),
                valid=bool(valid),
            )
        else:
            raise GateError(f"unsupported cache specification for {identity}: {spec!r}")
        if not self.is_valid_hash(entry.cache_hash):
            raise GateError(f"cache hash for {identity} must be a non-empty string")
        self._cache[identity] = entry
        return entry

    def invalidate(self, key: Any, reason: str = "") -> Optional[CacheEntry]:
        """Explicit, audited cache invalidation -- the only way to unblock creation."""
        identity = _normalise_identity(key)
        entry = self._cache.pop(identity, None)
        if entry is not None:
            self.invalidations.append((identity, entry.cache_hash, reason))
        return entry

    # -- lookups --------------------------------------------------------
    def valid_cache_entry(self, key: Any) -> Optional[CacheEntry]:
        identity = _normalise_identity(key)
        entry = self._cache.get(identity)
        if entry is None or not entry.valid or not self.is_valid_hash(entry.cache_hash):
            return None
        return entry

    def has_valid_cache(self, key: Any) -> bool:
        return self.valid_cache_entry(key) is not None

    # -- resolution -----------------------------------------------------
    def create_extractor(
        self,
        key: Any,
        cache_hash: Optional[str] = None,
        extractor_id: Optional[str] = None,
    ) -> ExtractorHandle:
        identity = _normalise_identity(key)
        entry = self.valid_cache_entry(identity)
        if entry is not None:
            self.blocked_creations += 1
            if self.is_valid_hash(cache_hash) and cache_hash != entry.cache_hash:
                self.conflicts.append((identity, entry.cache_hash, str(cache_hash)))
            raise ExtractorCreationBlocked(
                f"valid cache hash {entry.cache_hash!r} already exists for {identity}; "
                "reuse the cached extractor instead of creating a new one"
            )
        self.extractor_calls += 1
        new_id = extractor_id or _default_extractor_id(identity)
        if self.is_valid_hash(cache_hash):
            self.register_cache(identity, cache_hash, new_id)
            return ExtractorHandle(identity, new_id, reused=False, cache_hash=cache_hash)
        return ExtractorHandle(identity, new_id, reused=False, cache_hash=None)

    def resolve_extractor(
        self,
        key: Any,
        cache_hash: Optional[str] = None,
        extractor_id: Optional[str] = None,
    ) -> ExtractorHandle:
        identity = _normalise_identity(key)
        entry = self.valid_cache_entry(identity)
        if entry is not None:
            if self.is_valid_hash(cache_hash) and cache_hash != entry.cache_hash:
                self.blocked_creations += 1
                self.conflicts.append((identity, entry.cache_hash, str(cache_hash)))
                raise ExtractorCreationBlocked(
                    f"cache hash conflict for {identity}: cached={entry.cache_hash!r} "
                    f"requested={cache_hash!r}"
                )
            self.cache_reuses += 1
            return ExtractorHandle(
                identity, entry.extractor_id, reused=True, cache_hash=entry.cache_hash
            )
        return self.create_extractor(identity, cache_hash=cache_hash, extractor_id=extractor_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "cache_entries": len(self._cache),
            "valid_cache_entries": sum(1 for entry in self._cache.values() if entry.valid),
            "extractor_calls": self.extractor_calls,
            "cache_reuses": self.cache_reuses,
            "blocked_creations": self.blocked_creations,
            "invalidations": len(self.invalidations),
            "hash_conflicts": len(self.conflicts),
        }


# ---------------------------------------------------------------------------
# effort accounting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffortLedger:
    legacy_steps: int
    gated_steps: int

    @property
    def reduction(self) -> float:
        if self.legacy_steps <= 0:
            return 0.0
        return 1.0 - (self.gated_steps / self.legacy_steps)

    def meets(self, minimum: float = DEFAULT_MIN_STEP_REDUCTION) -> bool:
        return self.reduction >= minimum

    def as_dict(self) -> Dict[str, Any]:
        return {
            "legacy_steps": self.legacy_steps,
            "gated_steps": self.gated_steps,
            "reduction": self.reduction,
        }


def effort_ledger(
    row_count: int,
    *,
    record_count: int,
    cache_reuses: int = 0,
    case_count: Optional[int] = None,
) -> EffortLedger:
    """legacy = 3 steps per raw row; gated = extractor misses + writes + one handoff per case."""
    rows = max(0, int(row_count))
    records = max(0, int(record_count))
    reuses = max(0, min(int(cache_reuses), records))
    cases = records if case_count is None else max(0, int(case_count))
    legacy = rows * LEGACY_STEPS_PER_ROW
    gated = (records - reuses) + records + cases
    return EffortLedger(legacy_steps=legacy, gated_steps=gated)


# ---------------------------------------------------------------------------
# in-memory write gate
# ---------------------------------------------------------------------------
def apply_write_gate(
    result: Any, writer: Optional[Any] = None
) -> Dict[Tuple[str, str], CaseModalityRecord]:
    """Write exactly one record per (case_id, modality) into an in-memory store."""
    records = result.records if isinstance(result, GateResult) else tuple(result)
    store: Dict[Tuple[str, str], CaseModalityRecord] = {}
    for record in records:
        key = record.identity
        if key in store:
            raise DuplicateRowError(f"write blocked for duplicate identity {key}")
        store[key] = record
        if writer is not None:
            writer(record)
    return store


# ---------------------------------------------------------------------------
# single entry point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateRun:
    result: GateResult
    handoff: Dict[str, Any]
    store: Dict[Tuple[str, str], CaseModalityRecord]
    ledger: EffortLedger
    guard: ExtractorReuseGuard
    recall_coverage: float
    writes: int = 0


def run_a1_output_gate(
    rows: Iterable[Any],
    *,
    cache: Optional[Mapping[Any, Any]] = None,
    writer: Optional[Any] = None,
    generated_by: str = "efficiency_auditor",
) -> GateRun:
    """Run the full A1 output hard gate and emit one case-level handoff."""
    raw_rows = list(rows)
    result = apply_hard_gate(raw_rows)

    guard = ExtractorReuseGuard(cache or {})
    for record in result.records:
        guard.resolve_extractor(record.identity)

    facts = legacy_facts(raw_rows)
    coverage = ensure_recall_not_decreased(facts, result.records)

    writes: List[Tuple[str, str]] = []

    def _writer(record: CaseModalityRecord) -> None:
        writes.append(record.identity)
        if writer is not None:
            writer(record)

    store = apply_write_gate(result, writer=_writer)
    handoff = build_case_handoff(result, generated_by=generated_by)
    ledger = effort_ledger(
        result.input_rows,
        record_count=result.output_rows,
        cache_reuses=guard.cache_reuses,
        case_count=len({record.case_id for record in result.records}),
    )
    return GateRun(
        result=result,
        handoff=handoff,
        store=store,
        ledger=ledger,
        guard=guard,
        recall_coverage=coverage,
        writes=len(writes),
    )
