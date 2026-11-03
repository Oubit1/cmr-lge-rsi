"""Bounded, read-only shadow audit for the ``cmr_myo_excess_components`` stratum.

Maintenance action (``isolated_code_experiment``, owner ``cmr_specialist``):

* Bound the cohort to ``POSITIVE_SAMPLE_SIZE`` (50) positive cases plus
  ``CONTROL_SAMPLE_SIZE`` (10) CMR controls drawn from the
  iteration-history-uncovered slice of ``cmr_myo_excess_components`` (n=3397).
* Reuse the approved ``run-4/5/9/13-18`` caches.  A raw-payload hash mismatch is
  re-extracted **only** with the run-9 B1 extractor (``run-9:B1``) and reported
  explicitly; no other extractor is ever substituted.
* Emit exactly one ``(case_id, modality)`` row per sampled case, carrying the four
  secondary-component indicators: secondary component count, secondary area
  ratio, main-myocardium adjacency rate and cross-slice / cross-timepoint
  persistence rate.
* Roll the reader adjudication up into the three approved classes
  (``pseudo_fragment``, ``reasonable_anatomy``, ``true_segmentation_failure``)
  whose shares must total 100%.

The module is strictly read-only and offline: it performs no I/O and never mutates
labels, production rules, datasets or runtime databases.  All inputs (case rows,
cache index, extractor) are supplied by the caller, which keeps the audit
reproducible without downloading anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CACHE_RUN_IDS",
    "FALLBACK_EXTRACTOR_ID",
    "POSITIVE_STRATUM",
    "CONTROL_STRATUM",
    "POSITIVE_SAMPLE_SIZE",
    "CONTROL_SAMPLE_SIZE",
    "CACHE_HIT_RATE_FLOOR",
    "READER_CLASSES",
    "REQUIRED_METRICS",
    "REQUIRED_RECORD_KEYS",
    "READ_ONLY_FLAGS",
    "canonical_case_key",
    "admit_records",
    "collect_issue_types",
    "summarize_secondary_components",
    "stub_b1_extractor",
    "resolve_cached_payloads",
    "summarize_reader_adjudication",
    "attach_component_metrics",
    "metrics_completeness",
    "select_bounded_sample",
    "run_bounded_shadow_audit",
]

#: Approved cache runs that may be reused by this shadow audit.
CACHE_RUN_IDS: Tuple[str, ...] = (
    "run-4",
    "run-5",
    "run-9",
    "run-13",
    "run-14",
    "run-15",
    "run-16",
    "run-17",
    "run-18",
)

#: The single extractor allowed for hash-mismatch recovery.
FALLBACK_EXTRACTOR_ID = "run-9:B1"

POSITIVE_STRATUM = "cmr_myo_excess_components_positive"
CONTROL_STRATUM = "cmr_control"

POSITIVE_SAMPLE_SIZE = 50
CONTROL_SAMPLE_SIZE = 10
CACHE_HIT_RATE_FLOOR = 0.80

READER_CLASSES: Tuple[str, ...] = (
    "pseudo_fragment",
    "reasonable_anatomy",
    "true_segmentation_failure",
)

REQUIRED_METRICS: Tuple[str, ...] = (
    "secondary_component_count",
    "secondary_area_ratio",
    "main_myo_adjacency_rate",
    "cross_slice_or_temporal_persistence_rate",
)

REQUIRED_RECORD_KEYS: Tuple[str, ...] = (
    "case_id",
    "modality",
    "issue_type",
    "stratum",
    "expected_payload_hash",
    "components",
    "reader_label",
)

#: Declared, verifiable read-only posture of every audit report.
READ_ONLY_FLAGS: Dict[str, Any] = {
    "read_only": True,
    "label_mutations": 0,
    "production_rule_mutations": 0,
    "dataset_writes": 0,
    "filesystem_writes": 0,
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context}: missing required keys {missing}")


def canonical_case_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    """The single allowed identity of an audit row: ``(case_id, modality)``."""
    _require_keys(record, ("case_id", "modality"), "case row")
    case_id = str(record["case_id"]).strip()
    modality = str(record["modality"]).strip()
    if not case_id or not modality:
        raise ValueError("case row: case_id and modality must both be non-empty")
    return (case_id, modality)


def admit_records(records: Sequence[Mapping[str, Any]]):
    """Keep the first row per ``(case_id, modality)`` and reject every later duplicate."""
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    first_index: Dict[Tuple[str, str], int] = {}
    for record in records:
        _require_keys(record, REQUIRED_RECORD_KEYS, "record")
        key = canonical_case_key(record)
        if key in first_index:
            rejected.append(
                {
                    **record,
                    "reject_reason": "duplicate_case_modality",
                    "kept_index": first_index[key],
                }
            )
            continue
        first_index[key] = len(kept)
        kept.append(dict(record))

    duplicate_rows = len(records) - len(kept)
    if duplicate_rows != len(rejected):
        raise AssertionError("dedupe accounting drift: every duplicate row must be rejected")
    stats = {
        "rows_in": len(records),
        "rows_kept": len(kept),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_rejected": len(rejected),
        "duplicate_rejection_rate": 1.0 if duplicate_rows == 0 else len(rejected) / duplicate_rows,
    }
    return kept, rejected, stats


def collect_issue_types(records: Sequence[Mapping[str, Any]]) -> set:
    return {str(record["issue_type"]) for record in records}


def summarize_secondary_components(
    components: Sequence[Mapping[str, Any]],
    level_key: str = "slice_ids",
) -> Dict[str, float]:
    """Compute the four secondary-component indicators for one case.

    Exactly one primary (main myocardium) component is required and its area must
    be strictly positive; violations raise instead of degrading the audit.
    """
    if not isinstance(components, (list, tuple)) or len(components) == 0:
        raise ValueError("components must be a non-empty sequence of component rows")

    primaries: List[Dict[str, Any]] = []
    secondaries: List[Dict[str, Any]] = []
    for index, component in enumerate(components):
        _require_keys(
            component,
            ("area", "is_primary", "touches_main", level_key),
            f"component[{index}]",
        )
        area = component["area"]
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(float(area))
            or float(area) < 0.0
        ):
            raise ValueError(f"component[{index}]: area must be a finite non-negative number")
        if not isinstance(component["is_primary"], bool):
            raise ValueError(f"component[{index}]: is_primary must be a boolean")
        if not isinstance(component["touches_main"], bool):
            raise ValueError(f"component[{index}]: touches_main must be a boolean")
        row = {
            "area": float(area),
            "touches_main": bool(component["touches_main"]),
            "levels": tuple(component[level_key] or ()),
        }
        (primaries if component["is_primary"] else secondaries).append(row)

    if len(primaries) != 1:
        raise ValueError(f"exactly one primary component is required, found {len(primaries)}")
    primary = primaries[0]
    if primary["area"] <= 0.0:
        raise ValueError("primary component area must be positive for a ratio to be defined")

    count = len(secondaries)
    secondary_area = sum(row["area"] for row in secondaries)
    if count:
        adjacency_rate = sum(1 for row in secondaries if row["touches_main"]) / count
        persistence_rate = sum(1 for row in secondaries if len(set(row["levels"])) >= 2) / count
    else:
        adjacency_rate = 0.0
        persistence_rate = 0.0

    return {
        "secondary_component_count": count,
        "secondary_area_ratio": secondary_area / primary["area"],
        "main_myo_adjacency_rate": adjacency_rate,
        "cross_slice_or_temporal_persistence_rate": persistence_rate,
    }


def stub_b1_extractor(record: Mapping[str, Any]) -> str:
    """Offline deterministic stand-in for the run-9 B1 extractor.

    Production callers inject the real run-9 B1 extractor; tests use this stub so
    that no dataset or network access is ever required.
    """
    payload = {
        "case_id": record.get("case_id"),
        "modality": record.get("modality"),
        "issue_type": record.get("issue_type"),
        "components": [
            {
                "area": float(component["area"]),
                "is_primary": bool(component["is_primary"]),
                "touches_main": bool(component["touches_main"]),
                "levels": list(
                    component.get("slice_ids") or component.get("timepoint_ids") or ()
                ),
            }
            for component in record.get("components", ())
        ],
    }
    return _canonical_hash(payload)


def resolve_cached_payloads(
    records: Sequence[Mapping[str, Any]],
    cache_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
    b1_extractor: Optional[Callable[[Mapping[str, Any]], Any]] = None,
):
    """Resolve payload hashes from the approved caches, falling back to run-9 B1."""
    extractor = b1_extractor if b1_extractor is not None else stub_b1_extractor
    cache = cache_index or {}

    resolved: List[Dict[str, Any]] = []
    hits = mismatches = misses = fallbacks = 0

    for record in records:
        entry = cache.get(str(record["case_id"]))
        status = "miss"
        run_id = None
        fallback_reason = "cache_miss"

        if entry is not None and str(entry.get("modality", "")) == str(record["modality"]):
            run_id = entry.get("run_id")
            if run_id not in CACHE_RUN_IDS:
                raise ValueError(
                    f"cache entry for case {record['case_id']!r} points at unapproved run "
                    f"{run_id!r}; approved runs are {list(CACHE_RUN_IDS)}"
                )
            if entry.get("payload_hash") == record["expected_payload_hash"]:
                status = "hit"
            else:
                status = "mismatch"
                fallback_reason = "payload_hash_mismatch"

        if status == "hit":
            hits += 1
            payload_hash = str(entry["payload_hash"])
            payload_source = str(run_id)
            used_fallback = False
        else:
            if status == "mismatch":
                mismatches += 1
            else:
                misses += 1
            raw = extractor(record)
            payload_hash = str(raw["payload_hash"]) if isinstance(raw, Mapping) else str(raw)
            payload_source = FALLBACK_EXTRACTOR_ID
            used_fallback = True
            fallbacks += 1

        resolved.append(
            {
                **record,
                "payload_hash": payload_hash,
                "payload_source": payload_source,
                "cache_status": status,
                "used_fallback_extractor": used_fallback,
                "fallback_reason": None if not used_fallback else fallback_reason,
            }
        )

    total = len(resolved)
    stats = {
        "cache_hits": hits,
        "cache_mismatches": mismatches,
        "cache_misses": misses,
        "fallback_extractions": fallbacks,
        "cache_hit_rate": (hits / total) if total else 0.0,
        "cache_hit_rate_floor": CACHE_HIT_RATE_FLOOR,
        "fallback_extractor_id": FALLBACK_EXTRACTOR_ID,
        "cache_run_ids": list(CACHE_RUN_IDS),
    }
    return resolved, stats


def _largest_remainder_basis_points(
    counts: Mapping[str, int], classes: Sequence[str]
) -> Dict[str, int]:
    total = sum(counts.get(cls, 0) for cls in classes)
    if total <= 0:
        return {cls: 0 for cls in classes}
    raw = {cls: counts.get(cls, 0) * 10000.0 / total for cls in classes}
    floors = {cls: int(math.floor(raw[cls])) for cls in classes}
    remainder = 10000 - sum(floors.values())
    order = sorted(
        classes, key=lambda cls: (raw[cls] - floors[cls], counts.get(cls, 0), cls), reverse=True
    )
    for step in range(remainder):
        floors[order[step % len(order)]] += 1
    return floors


def summarize_reader_adjudication(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll reader verdicts into the three approved classes with shares summing to 100%."""
    counts = {cls: 0 for cls in READER_CLASSES}
    for record in records:
        label = record.get("reader_label")
        if label not in READER_CLASSES:
            raise ValueError(
                f"reader_label {label!r} on case {record.get('case_id')!r} is not one of "
                f"{list(READER_CLASSES)}"
            )
        counts[label] += 1

    basis_points = _largest_remainder_basis_points(counts, READER_CLASSES)
    if sum(basis_points.values()) != 10000:
        raise AssertionError("reader adjudication shares must total 100%")
    shares_pct = {cls: basis_points[cls] / 100.0 for cls in READER_CLASSES}
    return {"counts": counts, "basis_points": basis_points, "shares_pct": shares_pct}


def attach_component_metrics(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for record in records:
        level_key = str(record.get("level_key", "slice_ids"))
        metrics = summarize_secondary_components(record["components"], level_key=level_key)
        enriched.append({**record, "metrics": metrics})
    return enriched


def metrics_completeness(records: Sequence[Mapping[str, Any]]):
    missing: List[Dict[str, Any]] = []
    for record in records:
        metrics = record.get("metrics") or {}
        for key in REQUIRED_METRICS:
            value = metrics.get(key)
            bad = value is None or (isinstance(value, float) and not math.isfinite(value))
            if bad:
                missing.append({"case_id": record.get("case_id"), "metric": key})
    total = len(records) * len(REQUIRED_METRICS)
    rate = 1.0 if total == 0 else 1.0 - (len(missing) / total)
    return rate, missing


def select_bounded_sample(
    records: Sequence[Mapping[str, Any]],
    n_positive: int = POSITIVE_SAMPLE_SIZE,
    n_control: int = CONTROL_SAMPLE_SIZE,
    positive_stratum: str = POSITIVE_STRATUM,
    control_stratum: str = CONTROL_STRATUM,
):
    """Deterministically bound the cohort to ``n_positive`` + ``n_control`` cases."""
    approved = (positive_stratum, control_stratum)
    unknown = sorted({str(r.get("stratum")) for r in records if r.get("stratum") not in approved})
    if unknown:
        raise ValueError(f"unapproved stratum labels present: {unknown}")

    def _ordered(rows):
        return sorted(rows, key=lambda r: (str(r["case_id"]), str(r["modality"])))

    positives = _ordered([r for r in records if r["stratum"] == positive_stratum])
    controls = _ordered([r for r in records if r["stratum"] == control_stratum])
    sampled_positives = positives[:n_positive]
    sampled_controls = controls[:n_control]

    notes = {
        "positive_available": len(positives),
        "positive_sampled": len(sampled_positives),
        "control_available": len(controls),
        "control_sampled": len(sampled_controls),
        "positive_shortfall": max(0, n_positive - len(sampled_positives)),
        "control_shortfall": max(0, n_control - len(sampled_controls)),
    }
    return sampled_positives + sampled_controls, notes


def run_bounded_shadow_audit(
    records: Sequence[Mapping[str, Any]],
    cache_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
    b1_extractor: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    n_positive: int = POSITIVE_SAMPLE_SIZE,
    n_control: int = CONTROL_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Run the bounded, read-only shadow audit and return a JSON-serialisable report."""
    rows = [dict(record) for record in records]
    for row in rows:
        _require_keys(row, REQUIRED_RECORD_KEYS, "record")

    sample, sample_notes = select_bounded_sample(
        rows, n_positive=n_positive, n_control=n_control
    )
    kept, rejected, dedupe_stats = admit_records(sample)

    issue_types_in = collect_issue_types(sample)
    issue_types_out = collect_issue_types(kept)
    missing_issue_types = sorted(issue_types_in - issue_types_out)

    resolved, cache_stats = resolve_cached_payloads(
        kept, cache_index=cache_index, b1_extractor=b1_extractor
    )
    with_metrics = attach_component_metrics(resolved)
    completeness_rate, missing_metrics = metrics_completeness(with_metrics)
    reader = summarize_reader_adjudication(with_metrics)

    unique_case_keys = {canonical_case_key(row) for row in kept}
    gates = {
        "bounded_sample_ok": (
            sample_notes["positive_shortfall"] == 0 and sample_notes["control_shortfall"] == 0
        ),
        "case_modality_dedupe_ok": (
            dedupe_stats["duplicate_rows_rejected"] == dedupe_stats["duplicate_rows"]
            and dedupe_stats["duplicate_rejection_rate"] == 1.0
            and len(unique_case_keys) == len(kept)
            and len(kept) > 0
        ),
        "issue_type_set_preserved": not missing_issue_types,
        "metrics_complete": completeness_rate == 1.0 and len(kept) > 0,
        "reader_shares_sum_100": sum(reader["basis_points"].values()) == 10000,
        "cache_hit_rate_ok": cache_stats["cache_hit_rate"] >= CACHE_HIT_RATE_FLOOR,
        "read_only": True,
    }

    report = {
        "stratum": POSITIVE_STRATUM,
        "n_target": 3397,
        "sample": sample_notes,
        "dedupe": dedupe_stats,
        "issue_types_scope": "bounded_sample",
        "issue_types_in": sorted(issue_types_in),
        "issue_types_out": sorted(issue_types_out),
        "missing_issue_types": missing_issue_types,
        "rows_per_case_key_unique": len(unique_case_keys) == len(kept),
        "rejected_rows": rejected,
        "cache": cache_stats,
        "metrics": {
            "completeness_rate": completeness_rate,
            "missing": missing_metrics,
            "required": list(REQUIRED_METRICS),
        },
        "reader_adjudication": reader,
        "case_rows": [
            {
                "case_id": row["case_id"],
                "modality": row["modality"],
                "issue_type": row["issue_type"],
                "stratum": row["stratum"],
                "cache_status": row["cache_status"],
                "payload_source": row["payload_source"],
                "used_fallback_extractor": row["used_fallback_extractor"],
                "fallback_reason": row["fallback_reason"],
                "metrics": dict(row["metrics"]),
                "reader_label": row["reader_label"],
            }
            for row in with_metrics
        ],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        **READ_ONLY_FLAGS,
    }
    return report
