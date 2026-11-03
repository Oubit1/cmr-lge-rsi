"""A1 output-side hard gate: per-(case_id, modality) dedup, case rollup and
extractor cache guard.

Stdlib-only and offline: the gate consumes serialised A1 flag rows (plain
mappings) and returns plain mappings, so it can run inside the QC orchestrator
and in unit tests without touching datasets, patient images or runtime stores.

Hard rules implemented here
---------------------------
1. Exactly one delivered record per ``(case_id, modality)``; ``issue_type``
   sets are merged and every contributing slice / frame index stays
   addressable on the merged record (``issue_slices`` / ``issue_frames`` /
   ``index_map``).
2. High-frequency flags are rolled up per case as count, ratio, worst slice and
   severity tier.
3. When a valid cached summary exists, no new extractor may be created and
   every hash mismatch / missing hash is reported upward.

Duplicate rows are rejected outright and can never reach the delivery.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "DEFAULT_HIGH_FREQ_MIN_COUNT",
    "DEFAULT_HIGH_FREQ_RATIO",
    "DEFAULT_TIER_THRESHOLDS",
    "SEVERITY_TIERS",
    "dedup_rows",
    "guard_extractor_creation",
    "is_valid_cache_summary",
    "rollup_by_case",
    "run_a1_output_gate",
    "severity_tier",
]

SEVERITY_TIERS: Tuple[str, ...] = ("unknown", "none", "low", "moderate", "high", "critical")

DEFAULT_HIGH_FREQ_RATIO = 0.5
DEFAULT_HIGH_FREQ_MIN_COUNT = 3

DEFAULT_TIER_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.75, "critical"),
    (0.50, "high"),
    (0.25, "moderate"),
)

_SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITY_TIERS)}


def severity_tier(count: int, ratio: float, thresholds=None) -> str:
    """Map a flag ``count`` / ``ratio`` pair onto a severity tier label."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        return "none"
    if count <= 0:
        return "none"
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.0
    for min_ratio, tier in (thresholds or DEFAULT_TIER_THRESHOLDS):
        if ratio >= min_ratio:
            return tier
    return "low"


def _severity_max(labels) -> str:
    ranked = [label if label in _SEVERITY_RANK else "unknown" for label in labels]
    if not ranked:
        return "none"
    return max(ranked, key=lambda label: _SEVERITY_RANK[label])


def _coerce_index(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_indices(*groups) -> List[int]:
    """Union of slice / frame indices, returned sorted and duplicate-free."""
    merged: List[int] = []
    seen = set()
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (str, bytes)) or not hasattr(group, "__iter__"):
            single = _coerce_index(group)
            group = () if single is None else (single,)
        for value in group:
            index = _coerce_index(value)
            if index is None or index in seen:
                continue
            seen.add(index)
            merged.append(index)
    return sorted(merged)


def _worst_slice(slices):
    values = _merge_indices(slices)
    return values[-1] if values else None


def _normalise_row(row, position: int):
    """Return ``(record, errors)`` for one raw A1 flag row."""
    if not isinstance(row, Mapping):
        return None, ["row[%d] is not a mapping" % position]
    errors = []
    case_id = row.get("case_id")
    modality = row.get("modality")
    issue_type = row.get("issue_type")
    for name, value in (("case_id", case_id), ("modality", modality), ("issue_type", issue_type)):
        if value is None or str(value).strip() == "":
            errors.append("row[%d] missing %s" % (position, name))
    if errors:
        return None, errors
    return (
        {
            "case_id": str(case_id).strip(),
            "modality": str(modality).strip(),
            "issue_type": str(issue_type).strip(),
            "slices": _merge_indices(row.get("slices")),
            "frames": _merge_indices(row.get("frames")),
            "severity": str(row.get("severity") or "unknown"),
            "extractor_id": str(row.get("extractor_id") or ""),
        },
        [],
    )


def _row_signature(record) -> Tuple[Any, ...]:
    return (
        record["case_id"],
        record["modality"],
        record["issue_type"],
        tuple(record["slices"]),
        tuple(record["frames"]),
        record["severity"],
        record["extractor_id"],
    )


def _finalise_record(entry, high_freq_ratio, high_freq_min_count, tier_thresholds):
    issues = sorted(entry["issue_counts"])
    entry["issue_types"] = issues
    total = entry["source_rows"] or 1
    entry["index_map"] = {
        issue: {
            "slices": list(entry["issue_slices"].get(issue) or []),
            "frames": list(entry["issue_frames"].get(issue) or []),
        }
        for issue in issues
    }
    entry["issue_ratios"] = {
        issue: round(entry["issue_counts"][issue] / total, 6) for issue in issues
    }
    high_freq = []
    tiers = []
    for issue in issues:
        count = entry["issue_counts"][issue]
        ratio = count / total
        tier = severity_tier(count, ratio, tier_thresholds)
        tiers.append(tier)
        if count >= high_freq_min_count or ratio >= high_freq_ratio:
            high_freq.append(
                {
                    "issue_type": issue,
                    "count": count,
                    "ratio": round(ratio, 6),
                    "severity_tier": tier,
                    "worst_slice": _worst_slice(entry["issue_slices"].get(issue)),
                }
            )
    entry["severity_tier"] = _severity_max(tiers)
    entry["high_frequency_issues"] = high_freq
    entry["all_slices"] = _merge_indices(*[entry["issue_slices"].get(issue) for issue in issues])
    entry["all_frames"] = _merge_indices(*[entry["issue_frames"].get(issue) for issue in issues])
    return entry


def dedup_rows(
    rows,
    high_freq_ratio: float = DEFAULT_HIGH_FREQ_RATIO,
    high_freq_min_count: int = DEFAULT_HIGH_FREQ_MIN_COUNT,
    tier_thresholds=None,
) -> Dict[str, Any]:
    """Collapse raw A1 rows into one record per ``(case_id, modality)``.

    Identical rows are rejected outright (never delivered); rows that share a
    ``(case_id, modality, issue_type)`` but carry different slice / frame
    indices are merged, not duplicated.
    """
    records: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
    rejected: List[Dict[str, Any]] = []
    invalid: List[str] = []
    seen = set()
    accepted = 0

    for position, row in enumerate(rows or ()):
        record, errors = _normalise_row(row, position)
        if record is None:
            invalid.extend(errors)
            continue
        signature = _row_signature(record)
        if signature in seen:
            rejected.append(
                {
                    "index": position,
                    "reason": "duplicate_row",
                    "case_id": record["case_id"],
                    "modality": record["modality"],
                    "issue_type": record["issue_type"],
                    "slices": list(record["slices"]),
                    "frames": list(record["frames"]),
                }
            )
            continue
        seen.add(signature)
        accepted += 1

        key = (record["case_id"], record["modality"])
        entry = records.get(key)
        if entry is None:
            entry = {
                "case_id": record["case_id"],
                "modality": record["modality"],
                "issue_types": [],
                "issue_counts": {},
                "issue_ratios": {},
                "issue_slices": {},
                "issue_frames": {},
                "issue_severities": {},
                "index_map": {},
                "severity_tier": "none",
                "high_frequency_issues": [],
                "all_slices": [],
                "all_frames": [],
                "source_rows": 0,
            }
            records[key] = entry

        issue = record["issue_type"]
        entry["source_rows"] += 1
        entry["issue_counts"][issue] = entry["issue_counts"].get(issue, 0) + 1
        entry["issue_slices"][issue] = _merge_indices(
            entry["issue_slices"].get(issue), record["slices"]
        )
        entry["issue_frames"][issue] = _merge_indices(
            entry["issue_frames"].get(issue), record["frames"]
        )
        entry["issue_severities"][issue] = _severity_max(
            [entry["issue_severities"].get(issue), record["severity"]]
        )

    finalised = [
        _finalise_record(entry, high_freq_ratio, high_freq_min_count, tier_thresholds)
        for entry in records.values()
    ]
    return {
        "records": finalised,
        "input_rows": len(rows or ()),
        "accepted_rows": accepted,
        "duplicate_rows": len(rejected),
        "rejected": rejected,
        "invalid_rows": invalid,
    }


def _issue_precedes(candidate: str, current: Optional[str], issue_counts: Mapping[str, int]) -> bool:
    if current is None:
        return True
    if issue_counts.get(candidate, 0) != issue_counts.get(current, 0):
        return issue_counts.get(candidate, 0) > issue_counts.get(current, 0)
    return candidate < current


def _finalise_case(case, high_freq_ratio, high_freq_min_count, tier_thresholds):
    issues = sorted(case["issue_counts"])
    total = case["source_rows"] or 1
    tiers = []
    high_freq = []
    case["issue_ratios"] = {}
    for issue in issues:
        count = case["issue_counts"][issue]
        ratio = count / total
        case["issue_ratios"][issue] = round(ratio, 6)
        worst = _worst_slice(case["issue_slices"].get(issue))
        case["issue_worst_slices"][issue] = worst
        tier = severity_tier(count, ratio, tier_thresholds)
        tiers.append(tier)
        if count >= high_freq_min_count or ratio >= high_freq_ratio:
            high_freq.append(
                {
                    "issue_type": issue,
                    "count": count,
                    "ratio": round(ratio, 6),
                    "worst_slice": worst,
                    "severity_tier": tier,
                }
            )

    worst_slice = None
    worst_issue = None
    for issue in issues:
        candidate = case["issue_worst_slices"].get(issue)
        if candidate is None:
            continue
        if (
            worst_slice is None
            or candidate > worst_slice
            or (
                candidate == worst_slice
                and _issue_precedes(issue, worst_issue, case["issue_counts"])
            )
        ):
            worst_slice = candidate
            worst_issue = issue

    case["severity_tier"] = _severity_max(tiers)
    case["high_frequency_issues"] = high_freq
    case["high_frequency_issue_count"] = len(high_freq)
    case["worst_slice"] = worst_slice
    case["worst_slice_issue"] = worst_issue
    case["all_slices"] = _merge_indices(*[case["issue_slices"].get(issue) for issue in issues])
    case["all_frames"] = _merge_indices(*[case["issue_frames"].get(issue) for issue in issues])
    return case


def rollup_by_case(
    records,
    high_freq_ratio: float = DEFAULT_HIGH_FREQ_RATIO,
    high_freq_min_count: int = DEFAULT_HIGH_FREQ_MIN_COUNT,
    tier_thresholds=None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate merged records into a per-case summary keyed by ``case_id``."""
    cases: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for record in records or ():
        case_id = record.get("case_id")
        case = cases.get(case_id)
        if case is None:
            case = {
                "case_id": case_id,
                "record_count": 0,
                "modalities": [],
                "source_rows": 0,
                "issue_counts": {},
                "issue_ratios": {},
                "issue_slices": {},
                "issue_frames": {},
                "issue_worst_slices": {},
                "worst_slice": None,
                "worst_slice_issue": None,
                "severity_tier": "none",
                "high_frequency_issues": [],
                "high_frequency_issue_count": 0,
                "all_slices": [],
                "all_frames": [],
            }
            cases[case_id] = case

        case["record_count"] += 1
        modality = record.get("modality")
        if modality not in case["modalities"]:
            case["modalities"].append(modality)
        case["source_rows"] += int(record.get("source_rows") or 0)

        record_slices = record.get("issue_slices") or {}
        record_frames = record.get("issue_frames") or {}
        for issue, count in (record.get("issue_counts") or {}).items():
            case["issue_counts"][issue] = case["issue_counts"].get(issue, 0) + int(count)
            case["issue_slices"][issue] = _merge_indices(
                case["issue_slices"].get(issue), record_slices.get(issue)
            )
            case["issue_frames"][issue] = _merge_indices(
                case["issue_frames"].get(issue), record_frames.get(issue)
            )
        case["all_slices"] = _merge_indices(case["all_slices"], record.get("all_slices"))
        case["all_frames"] = _merge_indices(case["all_frames"], record.get("all_frames"))

    for case in cases.values():
        _finalise_case(case, high_freq_ratio, high_freq_min_count, tier_thresholds)
    return dict(cases)


def _cache_entry_count(summary: Mapping[str, Any]) -> int:
    for key in ("entries", "records", "cases", "items", "rows", "payload"):
        value = summary.get(key)
        if isinstance(value, Mapping):
            return len(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return len(value)
    count = summary.get("count")
    if isinstance(count, int) and count > 0:
        return count
    return 0


def is_valid_cache_summary(cache_summary) -> bool:
    """A cache summary is usable only with a non-empty hash and cached payload."""
    if not isinstance(cache_summary, Mapping):
        return False
    digest = cache_summary.get("hash") or cache_summary.get("digest")
    if not isinstance(digest, str) or not digest.strip():
        return False
    return _cache_entry_count(cache_summary) > 0


def _extractor_name(proposal, position: int) -> str:
    if isinstance(proposal, Mapping):
        name = proposal.get("name") or proposal.get("extractor_id")
        if name:
            return str(name)
    return "extractor[%d]" % position


def guard_extractor_creation(cache_summary, proposed_extractors=()):
    """Block extractor creation whenever a valid cached summary exists.

    Returns ``(allowed_extractors, report)``.  With a valid cache the allowed
    list is always empty (``report['created'] == 0``) and every hash mismatch /
    missing hash is recorded in ``report['hash_mismatches']`` and
    ``report['alerts']`` for upward reporting.
    """
    proposals = list(proposed_extractors or ())
    report: Dict[str, Any] = {
        "cache_valid": False,
        "cache_hash": None,
        "proposed": len(proposals),
        "created": 0,
        "blocked": [],
        "hash_mismatches": [],
        "hash_mismatch_count": 0,
        "missing_hash_count": 0,
        "alerts": [],
    }

    if not is_valid_cache_summary(cache_summary):
        report["created"] = len(proposals)
        report["allowed"] = [_extractor_name(p, i) for i, p in enumerate(proposals)]
        return proposals, report

    cache_hash = str(cache_summary.get("hash") or cache_summary.get("digest")).strip()
    report["cache_valid"] = True
    report["cache_hash"] = cache_hash

    for position, proposal in enumerate(proposals):
        name = _extractor_name(proposal, position)
        source_hash = proposal.get("source_hash") if isinstance(proposal, Mapping) else None
        source_hash = str(source_hash).strip() if source_hash is not None else ""
        if not source_hash:
            report["missing_hash_count"] += 1
            reason = "missing_source_hash"
        elif source_hash != cache_hash:
            report["hash_mismatch_count"] += 1
            reason = "hash_mismatch"
        else:
            reason = "cached_extractor_reused"

        report["blocked"].append(
            {
                "extractor": name,
                "reason": reason,
                "source_hash": source_hash or None,
            }
        )
        if reason != "cached_extractor_reused":
            mismatch = {
                "extractor": name,
                "reason": reason,
                "source_hash": source_hash or None,
                "cache_hash": cache_hash,
            }
            report["hash_mismatches"].append(mismatch)
            report["alerts"].append(
                "a1_output_gate: %s extractor=%s source_hash=%s cache_hash=%s"
                % (reason, name, source_hash or "<missing>", cache_hash)
            )

    report["created"] = 0
    report["allowed"] = []
    return [], report


def _slice_addressability_ok(records) -> bool:
    for record in records or ():
        issue_types = list(record.get("issue_types") or ())
        index_map = record.get("index_map") or {}
        if set(index_map.keys()) != set(issue_types):
            return False
        for issue in issue_types:
            slices = (record.get("issue_slices") or {}).get(issue)
            frames = (record.get("issue_frames") or {}).get(issue)
            if not isinstance(slices, list) or not isinstance(frames, list):
                return False
            if slices != sorted(set(slices)) or frames != sorted(set(frames)):
                return False
            mapped = index_map.get(issue)
            if not isinstance(mapped, Mapping):
                return False
            if list(mapped.get("slices") or []) != list(slices):
                return False
            if list(mapped.get("frames") or []) != list(frames):
                return False
    return True


def run_a1_output_gate(
    rows,
    cache_summary=None,
    proposed_extractors=(),
    high_freq_ratio: float = DEFAULT_HIGH_FREQ_RATIO,
    high_freq_min_count: int = DEFAULT_HIGH_FREQ_MIN_COUNT,
    tier_thresholds=None,
) -> Dict[str, Any]:
    """Run the full A1 output-side hard gate and return records + report."""
    dedup = dedup_rows(
        rows,
        high_freq_ratio=high_freq_ratio,
        high_freq_min_count=high_freq_min_count,
        tier_thresholds=tier_thresholds,
    )
    records = dedup["records"]
    case_summaries = rollup_by_case(
        records,
        high_freq_ratio=high_freq_ratio,
        high_freq_min_count=high_freq_min_count,
        tier_thresholds=tier_thresholds,
    )
    allowed_extractors, extractor_report = guard_extractor_creation(
        cache_summary, proposed_extractors
    )

    row_keys = {(record["case_id"], record["modality"]) for record in records}
    unique_row_compliance = len(row_keys) == len(records)
    slice_addressability_ok = _slice_addressability_ok(records)

    record_issue_total = sum(sum(record["issue_counts"].values()) for record in records)
    case_issue_total = sum(
        sum(case["issue_counts"].values()) for case in case_summaries.values()
    )
    reconciled = (
        dedup["accepted_rows"] == sum(record["source_rows"] for record in records)
        and record_issue_total == case_issue_total
    )

    duplicate_rows_delivered = 0  # hard gate: duplicates are dropped pre-delivery
    extractor_ok = (not extractor_report["cache_valid"]) or extractor_report["created"] == 0
    ok = bool(
        unique_row_compliance
        and slice_addressability_ok
        and reconciled
        and duplicate_rows_delivered == 0
        and extractor_ok
    )

    report = {
        "input_rows": dedup["input_rows"],
        "accepted_rows": dedup["accepted_rows"],
        "delivered_records": len(records),
        "rejected_duplicate_rows": dedup["duplicate_rows"],
        "duplicate_row_details": list(dedup["rejected"]),
        "invalid_rows": len(dedup["invalid_rows"]),
        "invalid_row_messages": list(dedup["invalid_rows"]),
        "case_count": len(case_summaries),
        "unique_row_compliance": unique_row_compliance,
        "slice_addressability_ok": slice_addressability_ok,
        "reconciled": reconciled,
        "duplicate_rows_delivered": duplicate_rows_delivered,
        "extractors_created": extractor_report["created"],
        "extractor_report": extractor_report,
    }
    return {
        "records": records,
        "case_summaries": case_summaries,
        "rejected": dedup["rejected"],
        "invalid_rows": dedup["invalid_rows"],
        "extractors": allowed_extractors,
        "report": report,
        "ok": ok,
    }
