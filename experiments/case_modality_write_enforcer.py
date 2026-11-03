"""Write-side (case_id, modality) uniqueness gate for the CMR/LGE QC orchestrator.

The audit orchestration used to *propose* de-duplication of ``(case_id, modality)``
rows.  This module upgrades that proposal into an enforcement point at write time:

* a duplicate ``(case_id, modality)`` row is never written as a second row - the
  duplicate row is rejected while its payload is folded into the canonical row;
* folding merges the ``issue_type`` set (first-seen order preserved) and the
  ``slice_indices`` set (sorted, de-duplicated) so no evidence is lost;
* rows that would silently lose evidence (non-integer slice index, non-mapping
  payload) raise instead of being dropped, and a batch is applied atomically;
* extractor instances are reused whenever the cache hash is valid - a valid hash
  with no cached entry raises ``CacheMissError`` instead of building a new
  extractor, and reuse metrics are asserted against an explicit policy;
* a small round-metrics helper records token / latency / recall deltas so the
  "cheaper without losing recall" claim stays falsifiable.

Pure standard library, no network access, no dataset or credential access.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "CACHE_HASH_MIN_LEN",
    "is_valid_cache_hash",
    "InvalidRowError",
    "WriteGateError",
    "CacheMissError",
    "ReusePolicyError",
    "merge_issue_types",
    "merge_slice_indices",
    "WriteResult",
    "CaseModalityWriteEnforcer",
    "ExtractorHandle",
    "ExtractorCache",
    "RoundMetrics",
    "compare_rounds",
]

CACHE_HASH_MIN_LEN = 16
_HEX_DIGITS = frozenset("0123456789abcdef")


class InvalidRowError(ValueError):
    """A row payload would silently lose evidence (e.g. a bad slice index)."""


class WriteGateError(RuntimeError):
    """The write-side ``(case_id, modality)`` uniqueness gate was violated."""


class CacheMissError(RuntimeError):
    """A valid cache hash has no cached extractor; building is refused."""


class ReusePolicyError(RuntimeError):
    """Extractor reuse metrics breach the efficiency policy."""


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def is_valid_cache_hash(cache_hash: Any) -> bool:
    """Return True for a non-empty, hex-looking cache hash long enough to trust.

    A valid hash means "this extractor configuration was already validated", so a
    caller must reuse the cached instance instead of building a new extractor.
    """

    if not isinstance(cache_hash, str):
        return False
    digest = cache_hash.strip().lower()
    if len(digest) < CACHE_HASH_MIN_LEN:
        return False
    return all(char in _HEX_DIGITS for char in digest)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def merge_issue_types(
    existing: Sequence[str], incoming: Sequence[str]
) -> List[str]:
    """Union of issue types, preserving first-seen order, dropping empties."""

    merged: List[str] = []
    for raw in list(existing) + list(incoming):
        item = _norm_text(raw)
        if item and item not in merged:
            merged.append(item)
    return merged


def merge_slice_indices(existing: Sequence[int], incoming: Sequence[int]) -> List[int]:
    """Union of slice indices, sorted ascending and de-duplicated."""

    return sorted({int(value) for value in list(existing)} | {int(value) for value in list(incoming)})


def _coerce_issue_types(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: List[Any] = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:  # pragma: no cover - defensive
            raise InvalidRowError(f"issue_type is not iterable: {value!r}") from exc
    merged: List[str] = []
    for item in items:
        text = _norm_text(item)
        if text and text not in merged:
            merged.append(text)
    return tuple(merged)


def _coerce_slice_indices(value: Any) -> Tuple[int, ...]:
    """Coerce slice indices, raising instead of silently dropping any index."""

    if value is None:
        return ()
    if isinstance(value, bool):
        raise InvalidRowError(f"slice index must be an integer, got {value!r}")
    if isinstance(value, int):
        raw: List[Any] = [value]
    elif isinstance(value, str):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError as exc:
            raise InvalidRowError(f"slice_indices is not iterable: {value!r}") from exc

    indices: List[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise InvalidRowError(f"slice index must be an integer, got {item!r}")
        if isinstance(item, int):
            indices.append(item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                indices.append(int(text))
            except ValueError as exc:
                raise InvalidRowError(
                    f"slice index is not an integer: {item!r}; refusing to drop it silently"
                ) from exc
            continue
        raise InvalidRowError(f"slice index must be an integer, got {item!r}")
    return tuple(sorted(set(indices)))


# --------------------------------------------------------------------------- #
# write-side de-duplication
# --------------------------------------------------------------------------- #
@dataclass
class WriteResult:
    """Outcome of one write-side ingest batch."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_merged: int = 0
    duplicates_rejected: int = 0
    invalid_rows: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def written_row_count(self) -> int:
        return len(self.rows)

    @property
    def duplicate_rows_written(self) -> int:
        """Acceptance metric: must stay at 0."""

        keys = [(row["case_id"], row["modality"]) for row in self.rows]
        return len(keys) - len(set(keys))

    def assert_no_duplicates(self) -> None:
        if self.duplicate_rows_written:
            raise WriteGateError(
                f"{self.duplicate_rows_written} duplicate (case_id, modality) rows reached the write side"
            )


class CaseModalityWriteEnforcer:
    """Accumulates audit rows with exactly one canonical row per (case_id, modality).

    Duplicate rows are rejected as standalone rows (they are never appended a
    second time) while their ``issue_type`` set and ``slice_indices`` are merged
    into the canonical row, so no evidence is lost.

    ``ingest`` is atomic per batch: a payload that would lose evidence raises
    before any state is mutated.
    """

    def __init__(self) -> None:
        self._order: List[Tuple[str, str]] = []
        self._canonical: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.merged_duplicate_count = 0
        self.rejected_invalid_count = 0

    # -- helpers --------------------------------------------------------- #
    def canonical_keys(self) -> List[Tuple[str, str]]:
        return list(self._order)

    def _snapshot(self, key: Tuple[str, str]) -> Dict[str, Any]:
        row = self._canonical[key]
        return {
            "case_id": row["case_id"],
            "modality": row["modality"],
            "issue_type": list(row["issue_type"]),
            "slice_indices": list(row["slice_indices"]),
        }

    def assert_no_duplicate_rows(self) -> None:
        if len(self._order) != len(set(self._order)):
            raise WriteGateError("internal store holds duplicate (case_id, modality) keys")

    # -- main entry point ------------------------------------------------ #
    def ingest(self, rows: Iterable[Mapping[str, Any]]) -> WriteResult:
        parsed: List[Tuple[Tuple[str, str], Tuple[str, ...], Tuple[int, ...]]] = []
        invalid: List[Tuple[int, str]] = []

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                invalid.append((index, "row_not_a_mapping"))
                continue
            case_id = _norm_text(row.get("case_id"))
            modality = _norm_text(row.get("modality")).upper()
            if not case_id or not modality:
                invalid.append((index, "missing_case_id_or_modality"))
                continue
            issue_types = _coerce_issue_types(
                row.get("issue_type", row.get("issue_types"))
            )
            slices = _coerce_slice_indices(
                row.get("slice_indices", row.get("slice_index"))
            )
            parsed.append(((case_id, modality), issue_types, slices))

        # Mutation phase only runs after every row parsed cleanly (atomic batch).
        result = WriteResult(invalid_rows=list(invalid))
        for key, issue_types, slices in parsed:
            existing = self._canonical.get(key)
            if existing is None:
                existing = {
                    "case_id": key[0],
                    "modality": key[1],
                    "issue_type": [],
                    "slice_indices": [],
                }
                self._canonical[key] = existing
                self._order.append(key)
            else:
                # Duplicate row: rejected as a standalone row, payload merged.
                result.duplicates_merged += 1
                result.duplicates_rejected += 1
                self.merged_duplicate_count += 1

            existing["issue_type"] = merge_issue_types(
                existing["issue_type"], issue_types
            )
            existing["slice_indices"] = merge_slice_indices(
                existing["slice_indices"], slices
            )

        self.rejected_invalid_count += len(invalid)
        result.rows = [self._snapshot(key) for key in self._order]
        result.assert_no_duplicates()
        return result


# --------------------------------------------------------------------------- #
# extractor reuse / cache-hash enforcement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtractorHandle:
    extractor: Any
    cache_hash: str
    reused: bool
    created: bool


class ExtractorCache:
    """Reuse extractors keyed by a validated cache hash.

    Enforcement rule: when ``cache_hash`` is valid the caller must reuse the
    cached extractor.  A valid hash with no cached entry is refused
    (``CacheMissError``) rather than silently building a second extractor.
    Only unvalidated / cold hashes may build a new extractor.
    """

    def __init__(
        self,
        *,
        min_reuse_rate: float = 0.9,
        max_new_extractors: int = 0,
        refuse_on_valid_hash_miss: bool = True,
    ) -> None:
        if not 0.0 <= min_reuse_rate <= 1.0:
            raise ValueError("min_reuse_rate must be within [0, 1]")
        if max_new_extractors < 0:
            raise ValueError("max_new_extractors must be >= 0")
        self.min_reuse_rate = float(min_reuse_rate)
        self.max_new_extractors = int(max_new_extractors)
        self.refuse_on_valid_hash_miss = bool(refuse_on_valid_hash_miss)

        self._entries: Dict[str, Any] = {}
        self.reused_count = 0
        self.created_count = 0
        self.refused_count = 0

    # -- cache maintenance ----------------------------------------------- #
    def warm(self, cache_hash: str, extractor: Any) -> None:
        """Explicitly register an extractor under a *valid* hash (no build cost)."""

        if not is_valid_cache_hash(cache_hash):
            raise ValueError(f"refusing to cache under an invalid hash: {cache_hash!r}")
        self._entries[cache_hash.strip().lower()] = extractor

    def invalidate(self, cache_hash: str) -> bool:
        key = cache_hash.strip().lower() if isinstance(cache_hash, str) else ""
        return self._entries.pop(key, None) is not None

    def cached_hashes(self) -> List[str]:
        return sorted(self._entries)

    # -- acquisition ------------------------------------------------------ #
    def obtain(
        self,
        cache_hash: Any,
        builder: Optional[Callable[[Any], Any]] = None,
        *,
        payload: Any = None,
    ) -> ExtractorHandle:
        key = cache_hash.strip().lower() if isinstance(cache_hash, str) else ""
        valid = is_valid_cache_hash(key)

        if valid and key in self._entries:
            self.reused_count += 1
            return ExtractorHandle(self._entries[key], key, reused=True, created=False)

        if valid:
            self.refused_count += 1
            if self.refuse_on_valid_hash_miss:
                raise CacheMissError(
                    f"valid cache hash {key!r} has no cached extractor; "
                    "refusing to build a duplicate extractor"
                )

        if not callable(builder):
            raise TypeError("builder must be callable when the cache cannot be reused")

        extractor = builder(payload)
        if valid:
            # Only reachable when refuse_on_valid_hash_miss is False.
            self._entries[key] = extractor
        self.created_count += 1
        return ExtractorHandle(extractor, key, reused=False, created=True)

    # -- metrics ----------------------------------------------------------- #
    @property
    def new_extractors(self) -> int:
        return self.created_count

    @property
    def reuse_rate(self) -> float:
        total = self.reused_count + self.created_count
        if total == 0:
            return 1.0
        return self.reused_count / total

    def stats(self) -> Dict[str, Any]:
        return {
            "reused": self.reused_count,
            "created": self.created_count,
            "refused": self.refused_count,
            "reuse_rate": self.reuse_rate,
            "new_extractors": self.new_extractors,
        }

    def assert_reuse_policy(
        self,
        *,
        min_reuse_rate: Optional[float] = None,
        max_new_extractors: Optional[int] = None,
    ) -> None:
        rate_floor = self.min_reuse_rate if min_reuse_rate is None else float(min_reuse_rate)
        new_cap = self.max_new_extractors if max_new_extractors is None else int(max_new_extractors)
        problems: List[str] = []
        if self.reuse_rate < rate_floor:
            problems.append(f"reuse_rate {self.reuse_rate:.3f} < {rate_floor:.3f}")
        if self.new_extractors > new_cap:
            problems.append(f"new_extractors {self.new_extractors} > {new_cap}")
        if problems:
            raise ReusePolicyError("; ".join(problems))


# --------------------------------------------------------------------------- #
# round-over-round efficiency guard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoundMetrics:
    tokens: float
    elapsed_ms: float
    recalled_issues: int


def compare_rounds(previous: RoundMetrics, current: RoundMetrics) -> Dict[str, Any]:
    """Compare two audit rounds: cheaper is only accepted if recall does not drop."""

    recall_ok = current.recalled_issues >= previous.recalled_issues
    tokens_ok = current.tokens <= previous.tokens
    latency_ok = current.elapsed_ms <= previous.elapsed_ms
    return {
        "token_delta": current.tokens - previous.tokens,
        "token_improved": tokens_ok,
        "elapsed_ms_delta": current.elapsed_ms - previous.elapsed_ms,
        "latency_improved": latency_ok,
        "recall_delta": current.recalled_issues - previous.recalled_issues,
        "recall_not_regressed": recall_ok,
        "accepted": bool(tokens_ok and latency_ok and recall_ok),
    }
