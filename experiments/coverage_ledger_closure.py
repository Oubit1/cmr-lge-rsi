"""Isolated-side coverage ledger closure.

Pure in-memory bookkeeping for the isolated audit side: no dataset, no patient
image, no production label and no runtime database is touched.  The ledger
enforces the coverage acceptance properties:

1. exactly one merged row per (case_id, modality) carrying the union of
   issue_type values, so a (case_id, modality, issue_type) triple never appears
   on more than one row;
2. every slice index is preserved, per issue and as a row-level union;
3. exact duplicate rows are rejected, and an audit already marked DONE is
   skipped rather than re-proposed while its content hash and rule version are
   unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

__all__ = [
    "DONE",
    "OPEN",
    "CoverageLedger",
    "CoverageLedgerError",
    "canonical_rows",
    "canonical_work_item",
    "closure_metrics",
    "content_hash",
    "dedup_stats",
    "duplicate_issue_row_count",
    "duplicate_row_count",
    "issue_content_hash",
    "max_issue_multiplicity",
    "merge_by_case_modality",
    "reject_duplicate_rows",
    "work_item_from_entry",
    "work_items_for_open_issues",
]

OPEN = "OPEN"
DONE = "DONE"
_STATUSES = frozenset({OPEN, DONE})
_UNVERSIONED = "unversioned"


class CoverageLedgerError(ValueError):
    """Raised when a coverage row or work item cannot be normalised safely."""


def _require_text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CoverageLedgerError(
            "coverage row needs a non-empty %r: %r" % (key, row)
        )
    return value.strip()


def _clean_version(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _UNVERSIONED


def _normalise_slice_index(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise CoverageLedgerError("boolean is not a slice index")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (TypeError, ValueError) as exc:
            raise CoverageLedgerError("unparsable slice index %r" % (value,)) from exc
    raise CoverageLedgerError("unsupported slice index %r" % (value,))


def _sentinel_key(case_id, modality, issue_type):
    return (
        str(case_id).strip(),
        str(modality).strip().upper(),
        str(issue_type).strip(),
    )


def canonical_rows(rows):
    """Normalise raw coverage rows, raising on anything unusable."""
    canonical = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CoverageLedgerError(
                "coverage row must be a mapping, got %s" % type(row).__name__
            )
        status = row.get("status", OPEN)
        status = status.strip().upper() if isinstance(status, str) else OPEN
        if status not in _STATUSES:
            raise CoverageLedgerError("unknown coverage status %r" % (status,))
        canonical.append(
            {
                "case_id": _require_text(row, "case_id"),
                "modality": _require_text(row, "modality").upper(),
                "issue_type": _require_text(row, "issue_type"),
                "slice_index": _normalise_slice_index(row.get("slice_index")),
                "rule_version": _clean_version(row.get("rule_version")),
                "status": status,
            }
        )
    return canonical


def _row_identity(row):
    return (
        row["case_id"],
        row["modality"],
        row["issue_type"],
        row["slice_index"],
        row["rule_version"],
        row["status"],
    )


def reject_duplicate_rows(rows):
    """Drop exact duplicate rows; return (kept_rows, dropped_count)."""
    kept = []
    seen = set()
    dropped = 0
    for row in canonical_rows(rows):
        identity = _row_identity(row)
        if identity in seen:
            dropped += 1
            continue
        seen.add(identity)
        kept.append(row)
    return kept, dropped


def duplicate_row_count(rows):
    """Rows beyond the first occurrence of the same exact row identity."""
    canonical = canonical_rows(rows)
    return len(canonical) - len({_row_identity(row) for row in canonical})


def duplicate_issue_row_count(rows):
    """Rows beyond the first occurrence of the same (case, modality, issue)."""
    canonical = canonical_rows(rows)
    triples = [
        (row["case_id"], row["modality"], row["issue_type"]) for row in canonical
    ]
    return len(triples) - len(set(triples))


def dedup_stats(rows):
    """Report how many duplicate rows the ledger removes and the reduction."""
    canonical = canonical_rows(rows)
    kept, _dropped = reject_duplicate_rows(canonical)
    rows_before = duplicate_row_count(canonical)
    rows_after = duplicate_row_count(kept)
    issues_before = duplicate_issue_row_count(canonical)
    issues_after = duplicate_issue_row_count(kept)
    baseline = max(rows_before, issues_before)
    residual = max(rows_after, issues_after)
    reduction = 1.0 if baseline == 0 else (baseline - residual) / float(baseline)
    return {
        "input_rows": len(canonical),
        "unique_rows": len(kept),
        "duplicate_rows_before": rows_before,
        "duplicate_rows_after": rows_after,
        "duplicate_issue_rows_before": issues_before,
        "duplicate_issue_rows_after": issues_after,
        "duplicate_reduction": reduction,
    }


def content_hash(case_id, modality, issue_types, slice_indices):
    """Order- and duplicate-insensitive hash of an audit's content."""
    payload = {
        "case_id": str(case_id),
        "modality": str(modality).upper(),
        "issue_types": sorted({str(issue) for issue in issue_types}),
        "slice_indices": sorted(
            {int(index) for index in slice_indices if index is not None}
        ),
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def issue_content_hash(entry, issue_type):
    """Hash for a single issue of a merged entry, using that issue's slices."""
    slices = (entry.get("issue_slices") or {}).get(issue_type, ())
    return content_hash(entry["case_id"], entry["modality"], (issue_type,), slices)


def merge_by_case_modality(rows):
    """Merge rows into one entry per (case_id, modality).

    issue_types are unioned (each issue appears once), slice indices are kept
    per issue and as a row-level union, and the newest rule version seen for the
    pair wins (lexicographic comparison).  An entry is DONE only when every
    contributing row was DONE.
    """
    kept, _dropped = reject_duplicate_rows(rows)
    grouped = {}
    for row in kept:
        key = (row["case_id"], row["modality"])
        group = grouped.get(key)
        if group is None:
            group = grouped[key] = {
                "case_id": row["case_id"],
                "modality": row["modality"],
                "issue_types": set(),
                "issue_slices": {},
                "rule_versions": set(),
                "statuses": set(),
            }
        group["issue_types"].add(row["issue_type"])
        bucket = group["issue_slices"].setdefault(row["issue_type"], set())
        if row["slice_index"] is not None:
            bucket.add(row["slice_index"])
        group["rule_versions"].add(row["rule_version"])
        group["statuses"].add(row["status"])

    merged = []
    for key in sorted(grouped):
        group = grouped[key]
        issue_slices = {
            issue: tuple(sorted(group["issue_slices"][issue]))
            for issue in sorted(group["issue_slices"])
        }
        issue_types = tuple(sorted(group["issue_types"]))
        slice_indices = tuple(
            sorted({index for slices in issue_slices.values() for index in slices})
        )
        entry = {
            "case_id": group["case_id"],
            "modality": group["modality"],
            "issue_types": issue_types,
            "slice_indices": slice_indices,
            "issue_slices": issue_slices,
            "rule_version": sorted(group["rule_versions"])[-1],
            "rule_versions": tuple(sorted(group["rule_versions"])),
            "status": DONE if group["statuses"] == {DONE} else OPEN,
        }
        entry["entry_hash"] = content_hash(
            entry["case_id"], entry["modality"], issue_types, slice_indices
        )
        merged.append(entry)
    return merged


def max_issue_multiplicity(entries):
    """Largest number of merged rows carrying the same (case, modality, issue)."""
    counts = {}
    for entry in entries:
        for issue in entry.get("issue_types") or ():
            key = (entry.get("case_id"), entry.get("modality"), issue)
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def canonical_work_item(candidate):
    """Normalise one audit work item and (re)compute its content hash."""
    if not isinstance(candidate, Mapping):
        raise CoverageLedgerError(
            "work item must be a mapping, got %s" % type(candidate).__name__
        )
    case_id = _require_text(candidate, "case_id")
    modality = _require_text(candidate, "modality").upper()
    issue_type = _require_text(candidate, "issue_type")
    rule_version = _clean_version(candidate.get("rule_version"))

    raw = candidate.get("slice_indices")
    if raw is None:
        single = candidate.get("slice_index")
        raw = [] if single is None else [single]
    elif isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise CoverageLedgerError(
            "slice_indices must be a non-string iterable, got %r" % (raw,)
        )

    slices = set()
    for value in raw:
        index = _normalise_slice_index(value)
        if index is not None:
            slices.add(index)
    slice_indices = tuple(sorted(slices))

    return {
        "case_id": case_id,
        "modality": modality,
        "issue_type": issue_type,
        "slice_indices": slice_indices,
        "rule_version": rule_version,
        "content_hash": content_hash(
            case_id, modality, (issue_type,), slice_indices
        ),
    }


def work_item_from_entry(entry, issue_type, rule_version=None):
    """Build the work item for one issue of a merged entry."""
    slices = (entry.get("issue_slices") or {}).get(issue_type, ())
    candidate = {
        "case_id": entry["case_id"],
        "modality": entry["modality"],
        "issue_type": issue_type,
        "slice_indices": slices,
        "rule_version": (
            entry.get("rule_version") if rule_version is None else rule_version
        ),
    }
    return canonical_work_item(candidate)


def work_items_for_open_issues(entries):
    """Work items for every issue of every entry that is not fully DONE."""
    items = []
    for entry in entries:
        if entry.get("status") == DONE:
            continue
        for issue in entry.get("issue_types") or ():
            items.append(work_item_from_entry(entry, issue))
    return items


def closure_metrics(rows):
    """Acceptance-facing metrics for one batch of raw coverage rows."""
    stats = dedup_stats(rows)
    merged = merge_by_case_modality(rows)
    multiplicity = max_issue_multiplicity(merged)
    return {
        "merged_entries": len(merged),
        "issue_types_total": sum(len(entry["issue_types"]) for entry in merged),
        "max_issue_multiplicity": multiplicity,
        "unique_per_case_modality_issue": multiplicity <= 1,
        "duplicate_rows_after": stats["duplicate_rows_after"],
        "duplicate_reduction": stats["duplicate_reduction"],
    }


class CoverageLedger:
    """In-memory isolated-side ledger with DONE short-circuiting."""

    def __init__(self):
        self.entries = []
        self.done_sentinels = {}
        self.audit_events = []

    def load(self, rows):
        """Merge raw rows and rebuild DONE sentinels from fully DONE entries."""
        self.entries = merge_by_case_modality(rows)
        self.done_sentinels = {}
        for entry in self.entries:
            if entry["status"] != DONE:
                continue
            for issue in entry["issue_types"]:
                self.done_sentinels[
                    _sentinel_key(entry["case_id"], entry["modality"], issue)
                ] = {
                    "content_hash": issue_content_hash(entry, issue),
                    "rule_version": entry["rule_version"],
                }
        return self.entries

    def entry_for(self, case_id, modality):
        key = (str(case_id).strip(), str(modality).strip().upper())
        for entry in self.entries:
            if (entry["case_id"], entry["modality"]) == key:
                return entry
        return None

    def is_done(self, case_id, modality, issue_type):
        return _sentinel_key(case_id, modality, issue_type) in self.done_sentinels

    def should_skip(self, case_id, modality, issue_type, current_hash, rule_version):
        """True when a DONE marker still matches the hash and the rule version."""
        sentinel = self.done_sentinels.get(
            _sentinel_key(case_id, modality, issue_type)
        )
        if sentinel is None:
            return False
        return (
            sentinel["content_hash"] == current_hash
            and sentinel["rule_version"] == _clean_version(rule_version)
        )

    def propose(self, candidates):
        """Return only work items that are not already frozen as DONE."""
        proposals = []
        for candidate in candidates:
            item = canonical_work_item(candidate)
            skip = self.should_skip(
                item["case_id"],
                item["modality"],
                item["issue_type"],
                item["content_hash"],
                item["rule_version"],
            )
            self.audit_events.append(
                {"event": "skip_done" if skip else "propose", **item}
            )
            if not skip:
                proposals.append(item)
        return proposals

    def mark_done(
        self,
        case_id,
        modality,
        issue_type,
        content_hash,
        rule_version,
        validation_passed,
    ):
        """Freeze an issue as DONE, but only after validation has passed."""
        if not isinstance(content_hash, str) or not content_hash.strip():
            raise CoverageLedgerError("mark_done needs a non-empty content hash")
        key = _sentinel_key(case_id, modality, issue_type)
        version = _clean_version(rule_version)
        event = {
            "case_id": key[0],
            "modality": key[1],
            "issue_type": key[2],
            "content_hash": content_hash,
            "rule_version": version,
        }
        if not validation_passed:
            self.audit_events.append({"event": "mark_done_rejected", **event})
            return False
        self.done_sentinels[key] = {
            "content_hash": content_hash,
            "rule_version": version,
        }
        self.audit_events.append({"event": "mark_done", **event})
        return True
