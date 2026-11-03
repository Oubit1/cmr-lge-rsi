"""A1 write-side blocking hard gate (efficiency auditor).

Why this module exists
----------------------
The A1 output path used to append one row per downstream proposal for the same
``(case_id, modality)`` key, and to rebuild extractors even when a perfectly
valid cache digest was already present.  Duplicate rows then had to be spotted
by another downstream proposal, which is exactly the loop this gate removes.

This module implements three *blocking* rules on the write path:

1. :class:`A1WriteGate` refuses a second row for a ``(case_id, modality)`` key.
   The colliding record is merged into the stored row instead of appended.
2. The merge unions the issue-type sets and keeps every flagged slice / frame
   index addressable, while recording counts, ratios, the worst slice and a
   severity tier.
3. :class:`ExtractorCacheGuard` refuses a new extractor while a valid cache
   digest exists.  On a digest mismatch at most one extractor is (re)built per
   key and the mismatch is always reported.

Everything is pure Python, offline, deterministic and side-effect free: no
dataset, no patient image, no production label and no runtime database is read
or written.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "a1_write_blocking_gate/v1"
SEVERITY_ORDER: Tuple[str, ...] = ("info", "warning", "review", "critical")
DEFAULT_SEVERITY = "info"


class A1GateInputError(ValueError):
    """Raised when an A1 record cannot be keyed or normalised."""


def severity_rank(severity: Any) -> int:
    """Return the ordinal of a severity label (unknown labels map to 'info')."""
    text = str(severity or DEFAULT_SEVERITY).strip().lower()
    if text in SEVERITY_ORDER:
        return SEVERITY_ORDER.index(text)
    return SEVERITY_ORDER.index(DEFAULT_SEVERITY)


def severity_tier(severities: Iterable[Any]) -> str:
    """Return the worst severity tier of an iterable of severity labels."""
    worst = 0
    for severity in severities or ():
        worst = max(worst, severity_rank(severity))
    return SEVERITY_ORDER[worst]


def record_key(case_id: Any, modality: Any) -> str:
    """Canonical string key for one ``(case_id, modality)`` row."""
    return "{}::{}".format(str(case_id).strip(), str(modality).strip().upper())


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _pick(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _clean_indices(values: Any) -> List[int]:
    indices = set()
    for value in _as_list(values):
        if value is None:
            continue
        try:
            indices.add(int(value))
        except (TypeError, ValueError):
            raise A1GateInputError("slice/frame index is not an integer: {!r}".format(value))
    return sorted(indices)


def _coerce_occurrences(value: Any) -> int:
    try:
        occurrences = int(value)
    except (TypeError, ValueError):
        return 1
    return occurrences if occurrences >= 1 else 1


def normalize_issue(issue: Any) -> Dict[str, Any]:
    """Normalise one issue entry, preserving every slice / frame index."""
    if isinstance(issue, str):
        issue = {"issue_type": issue}
    if not isinstance(issue, Mapping):
        raise A1GateInputError("issue must be a mapping or a string")
    issue_type = _pick(issue, ("issue_type", "type", "problem_type"))
    if issue_type in (None, ""):
        raise A1GateInputError("issue requires a non-empty issue_type")
    severity = str(issue.get("severity") or DEFAULT_SEVERITY).strip().lower()
    if severity not in SEVERITY_ORDER:
        severity = DEFAULT_SEVERITY
    source = issue.get("source")
    return {
        "issue_type": str(issue_type).strip(),
        "slices": _clean_indices(_pick(issue, ("slices", "slice_indices", "slice_index"))),
        "frames": _clean_indices(_pick(issue, ("frames", "frame_indices", "frame_index"))),
        "severity": severity,
        "occurrences": _coerce_occurrences(issue.get("occurrences")),
        "source": str(source) if source else None,
    }


def normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise one A1 output record for the write gate."""
    if not isinstance(record, Mapping):
        raise A1GateInputError("record must be a mapping, got {}".format(type(record).__name__))
    case_id = record.get("case_id")
    modality = record.get("modality")
    if case_id in (None, "") or modality in (None, ""):
        raise A1GateInputError("record requires non-empty case_id and modality")
    case_id = str(case_id).strip()
    modality = str(modality).strip().upper()
    if not case_id or not modality:
        raise A1GateInputError("record requires non-empty case_id and modality")

    raw_issues = record.get("issues")
    if raw_issues is None:
        raw_issues = record.get("problems")
    if raw_issues is None:
        raw_issues = record.get("issue_types")
    issues = [normalize_issue(issue) for issue in _as_list(raw_issues)]

    total = record.get("total_slices")
    total_slices: Optional[int] = None
    if total not in (None, ""):
        try:
            total_slices = int(total)
        except (TypeError, ValueError):
            raise A1GateInputError("total_slices must be an integer")
        if total_slices <= 0:
            total_slices = None

    return {
        "case_id": case_id,
        "modality": modality,
        "key": (case_id, modality),
        "issues": issues,
        "total_slices": total_slices,
        "record_count": _coerce_occurrences(record.get("record_count")),
    }


def merge_issues(issue_groups: Iterable[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Union issue-type sets, keeping every slice / frame index addressable."""
    buckets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for group in issue_groups or ():
        for issue in group or ():
            issue_type = issue["issue_type"]
            bucket = buckets.get(issue_type)
            if bucket is None:
                bucket = {
                    "issue_type": issue_type,
                    "slices": set(),
                    "frames": set(),
                    "severity": issue.get("severity", DEFAULT_SEVERITY),
                    "occurrences": 0,
                    "sources": set(),
                }
                buckets[issue_type] = bucket
                order.append(issue_type)
            bucket["slices"].update(issue.get("slices") or ())
            bucket["frames"].update(issue.get("frames") or ())
            bucket["occurrences"] += int(issue.get("occurrences") or 1)
            if severity_rank(issue.get("severity")) > severity_rank(bucket["severity"]):
                bucket["severity"] = issue.get("severity")
            if issue.get("source"):
                bucket["sources"].add(issue["source"])

    merged: List[Dict[str, Any]] = []
    for issue_type in order:
        bucket = buckets[issue_type]
        merged.append(
            {
                "issue_type": issue_type,
                "slices": sorted(bucket["slices"]),
                "frames": sorted(bucket["frames"]),
                "severity": bucket["severity"],
                "occurrences": bucket["occurrences"],
                "sources": sorted(bucket["sources"]),
            }
        )
    merged.sort(key=lambda item: (-severity_rank(item["severity"]), -item["occurrences"], item["issue_type"]))
    return merged


def worst_slice_index(issues: Iterable[Mapping[str, Any]]) -> Optional[int]:
    """Slice carrying the worst severity (ties: more occurrences, then lower index)."""
    best: Optional[Tuple[Tuple[int, int, int, int], int]] = None
    for issue in issues or ():
        rank = severity_rank(issue.get("severity"))
        occurrences = int(issue.get("occurrences") or 1)
        for raw_index in issue.get("slices") or ():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            candidate = (rank, occurrences, -index, index)
            if best is None or candidate > best[0]:
                best = (candidate, index)
    return best[1] if best else None


def build_canonical_row(records: Iterable[Mapping[str, Any]], total_slices: Any = None) -> Dict[str, Any]:
    """Merge one or more records sharing a single ``(case_id, modality)`` key.

    The returned row is the only shape the gate ever stores: one row per key,
    issue-type sets unioned, all flagged indices addressable, plus counts,
    ratios, the worst slice and a severity tier.
    """
    normalised = [normalize_record(record) for record in records]
    if not normalised:
        raise A1GateInputError("cannot build a row from zero records")
    keys = {item["key"] for item in normalised}
    if len(keys) != 1:
        raise A1GateInputError("records do not share a single (case_id, modality) key")

    issues = merge_issues([item["issues"] for item in normalised])

    totals = [item["total_slices"] for item in normalised if item["total_slices"]]
    if total_slices not in (None, ""):
        try:
            explicit_total = int(total_slices)
        except (TypeError, ValueError):
            raise A1GateInputError("total_slices must be an integer")
        if explicit_total > 0:
            totals.append(explicit_total)
    total = max(totals) if totals else None

    flagged_slices = sorted({index for issue in issues for index in issue["slices"]})
    flagged_frames = sorted({index for issue in issues for index in issue["frames"]})
    occurrences = sum(int(issue["occurrences"]) for issue in issues)
    case_id = normalised[0]["case_id"]
    modality = normalised[0]["modality"]

    return {
        "case_id": case_id,
        "modality": modality,
        "key": [case_id, modality],
        "issue_types": [issue["issue_type"] for issue in issues],
        "issues": issues,
        "issue_count": len(issues),
        "issue_occurrences": occurrences,
        "flagged_slices": flagged_slices,
        "flagged_frames": flagged_frames,
        "flagged_slice_count": len(flagged_slices),
        "flagged_frame_count": len(flagged_frames),
        "worst_slice": worst_slice_index(issues),
        "severity_tier": severity_tier(issue["severity"] for issue in issues),
        "total_slices": total,
        "flagged_slice_ratio": (round(len(flagged_slices) / total, 6) if total else None),
        "issue_occurrence_ratio": (round(occurrences / total, 6) if total else None),
        "record_count": sum(int(item["record_count"]) for item in normalised),
    }


def index_addressability(row: Mapping[str, Any], sources: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Check that every slice / frame index of the sources survives the merge."""
    expected_slices = set()
    expected_frames = set()
    for source in sources or ():
        normalised = normalize_record(source)
        for issue in normalised["issues"]:
            expected_slices.update(issue["slices"])
            expected_frames.update(issue["frames"])
    present_slices = set(row.get("flagged_slices") or ())
    present_frames = set(row.get("flagged_frames") or ())
    missing_slices = sorted(expected_slices - present_slices)
    missing_frames = sorted(expected_frames - present_frames)
    return {
        "complete": not missing_slices and not missing_frames,
        "missing_slices": missing_slices,
        "missing_frames": missing_frames,
        "expected_slice_count": len(expected_slices),
        "expected_frame_count": len(expected_frames),
    }


class A1WriteGate:
    """Blocking write gate: at most one stored row per ``(case_id, modality)``."""

    def __init__(self, total_slices: Any = None) -> None:
        self._total_slices = total_slices
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._attempts = 0
        self._blocked: List[str] = []
        self._merges: List[Dict[str, Any]] = []
        self._addressability: List[Dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Attempt one write.  A colliding key is blocked and merged instead."""
        normalised = normalize_record(record)
        key = record_key(normalised["case_id"], normalised["modality"])
        self._attempts += 1

        if key in self._rows:
            previous = self._rows[key]
            merged = build_canonical_row([previous, normalised], total_slices=self._total_slices)
            check = index_addressability(merged, [previous, normalised])
            self._rows[key] = merged
            self._blocked.append(key)
            self._addressability.append(dict(check, key=key))
            merge_report = {
                "key": key,
                "blocked_write_number": self._attempts,
                "record_count": merged["record_count"],
                "issue_types": list(merged["issue_types"]),
                "issue_count": merged["issue_count"],
                "issue_occurrences": merged["issue_occurrences"],
                "flagged_slice_count": merged["flagged_slice_count"],
                "flagged_frame_count": merged["flagged_frame_count"],
                "worst_slice": merged["worst_slice"],
                "severity_tier": merged["severity_tier"],
                "indices_preserved": check["complete"],
            }
            self._merges.append(merge_report)
            return {
                "accepted": False,
                "blocked": True,
                "key": key,
                "row": merged,
                "merge": merge_report,
                "addressability": check,
            }

        row = build_canonical_row([normalised], total_slices=self._total_slices)
        self._rows[key] = row
        return {
            "accepted": True,
            "blocked": False,
            "key": key,
            "row": row,
            "merge": None,
            "addressability": None,
        }

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows.values())

    def snapshot(self) -> Dict[str, Any]:
        rows = list(self._rows.values())
        keys = [record_key(row["case_id"], row["modality"]) for row in rows]
        unique = len(set(keys))
        duplicates = len(keys) - unique
        return {
            "schema": SCHEMA,
            "rows": rows,
            "row_count": len(rows),
            "write_attempts": self._attempts,
            "blocked_writes": len(self._blocked),
            "blocked_keys": sorted(set(self._blocked)),
            "merges": list(self._merges),
            "unique_key_count": unique,
            "duplicate_row_count": duplicates,
            "duplicate_row_ratio": (round(duplicates / len(rows), 6) if rows else 0.0),
            "uniqueness_ratio": (round(unique / len(rows), 6) if rows else 1.0),
            "index_addressability": list(self._addressability),
            "all_indices_addressable": all(item["complete"] for item in self._addressability),
        }


class ExtractorCacheGuard:
    """Refuse a new extractor while a valid cache digest already exists."""

    def __init__(self, cache_state: Optional[Mapping[str, Any]] = None) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        for key, entry in (cache_state or {}).items():
            if isinstance(entry, Mapping):
                self._cache[str(key)] = {
                    "digest": entry.get("digest"),
                    "valid": bool(entry.get("valid", True)),
                }
            else:
                self._cache[str(key)] = {"digest": entry, "valid": True}
        self._plans: List[Dict[str, Any]] = []
        self._reports: List[Dict[str, Any]] = []
        self._created: set = set()
        self._key_creations: Dict[str, int] = {}
        self._creation_count = 0

    def _create(self, key: str, digest: Any) -> None:
        self._created.add(key)
        self._key_creations[key] = self._key_creations.get(key, 0) + 1
        self._creation_count += 1
        self._cache[key] = {"digest": digest, "valid": True}

    def plan(self, key: Any, candidate_digest: Any, cache_valid: Optional[bool] = None) -> Dict[str, Any]:
        """Decide whether an extractor may be built for ``key``."""
        key = str(key)
        entry = self._cache.get(key)
        cached_digest = entry.get("digest") if isinstance(entry, Mapping) else None
        valid = bool(entry.get("valid", True)) if isinstance(entry, Mapping) else False
        if cache_valid is not None and isinstance(entry, Mapping):
            valid = bool(cache_valid)

        if entry is not None and valid and cached_digest == candidate_digest:
            plan = {
                "key": key,
                "action": "reuse",
                "creations": 0,
                "report_required": False,
                "reason": "valid_cache_digest_match",
            }
        elif entry is not None and valid:
            self._reports.append(
                {
                    "key": key,
                    "reason": "hash_mismatch",
                    "cached_digest": cached_digest,
                    "candidate_digest": candidate_digest,
                    "cached_entry_valid": True,
                }
            )
            if key in self._created:
                plan = {
                    "key": key,
                    "action": "blocked_recreate",
                    "creations": 0,
                    "report_required": True,
                    "reason": "hash_mismatch_after_creation",
                }
            else:
                self._create(key, candidate_digest)
                plan = {
                    "key": key,
                    "action": "recreate",
                    "creations": 1,
                    "report_required": True,
                    "reason": "hash_mismatch",
                }
        elif key in self._created:
            plan = {
                "key": key,
                "action": "reuse_created",
                "creations": 0,
                "report_required": False,
                "reason": "extractor_already_created",
            }
        else:
            self._create(key, candidate_digest)
            plan = {
                "key": key,
                "action": "create",
                "creations": 1,
                "report_required": False,
                "reason": "no_valid_cache",
            }

        self._plans.append(dict(plan))
        return plan

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "plans": list(self._plans),
            "extractor_creations": self._creation_count,
            "key_creations": dict(self._key_creations),
            "max_creations_per_key": (max(self._key_creations.values()) if self._key_creations else 0),
            "reports": list(self._reports),
            "hash_mismatch_reports": len(self._reports),
            "report_required": bool(self._reports),
        }


def replay_evidence(
    records: Iterable[Mapping[str, Any]],
    cache_state: Optional[Mapping[str, Any]] = None,
    extractor_requests: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Offline replay of an A1 evidence bundle through gate + cache guard."""
    gate = A1WriteGate()
    writes = [gate.write(record) for record in records or ()]

    guard = ExtractorCacheGuard(cache_state)
    plans: List[Dict[str, Any]] = []
    for request in extractor_requests or ():
        key = request.get("key") or record_key(request.get("case_id"), request.get("modality"))
        plans.append(guard.plan(key, request.get("digest"), request.get("cache_valid")))

    return {
        "schema": SCHEMA,
        "gate": gate.snapshot(),
        "writes": writes,
        "extractor": guard.snapshot(),
        "extractor_plans": plans,
    }
