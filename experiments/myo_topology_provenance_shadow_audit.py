"""Bounded, read-only, import-only provenance shadow audit for ``myo_topology``.

``myo_topology`` (n~578) has no dedicated main route in ``iteration_history``:
the rule is only visible through shared caches produced by runs 4/5/9/13-18.
This module therefore

* imports already-verified cache summaries instead of re-deriving provenance
  (no second tool set is created),
* verifies the canonical sha256 hash of every imported summary,
* on hash mismatch falls back to the single ``run-9 B1`` contract extractor
  (:data:`RUN9_B1_EXTRACTOR`) and reports the fallback,
* never writes state, never promotes routing, never weakens a QC gate.

Reported: per-case slice counts / positive rate / worst slice index and severity
strata, same-slice co-occurrence with ``mi_multiple_components`` and
``mi_fraction_high``, cross-slice continuous positive run lengths, and read-only
no-op / ring-continuity / max-connected-component arms scored on the frozen,
deduplicated slice set.

Everything below is pure computation over in-memory payloads: no file is
opened and no dataset, patient image, production label or runtime database is
touched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = [
    "MYO_TOPOLOGY_RULE",
    "MYO_TOPOLOGY_EXPECTED_CASES",
    "VERIFIED_CACHE_RUNS",
    "RUN9_B1_EXTRACTOR",
    "CO_OCCURRENCE_FLAGS",
    "SEVERITY_STRATA",
    "DEFAULT_ARM_AGREEMENT_BAR",
    "DEFAULT_MIN_COMPONENT_SHARE",
    "DEFAULT_ARMS",
    "EXTRACTORS",
    "canonical_payload_hash",
    "severity_stratum",
    "is_verified_run",
    "slice_key",
    "assert_single_extractor",
    "extract_b1_contract",
    "freeze_slices",
    "per_case_topology",
    "same_slice_cooccurrence",
    "continuous_positive_runs",
    "noop_arm",
    "ring_continuity_arm",
    "max_component_arm",
    "arm_agreement",
    "verify_cache_hashes",
    "run_shadow_audit",
]

MYO_TOPOLOGY_RULE = "myo_topology"
MYO_TOPOLOGY_EXPECTED_CASES = 578
VERIFIED_CACHE_RUNS = (4, 5, 9, 13, 14, 15, 16, 17, 18)
RUN9_B1_EXTRACTOR = "run9_b1_contract"
CO_OCCURRENCE_FLAGS = ("mi_multiple_components", "mi_fraction_high")
SEVERITY_STRATA = ("none", "low", "moderate", "high")
SEVERITY_HIGH = 0.66
SEVERITY_MODERATE = 0.33
DEFAULT_ARM_AGREEMENT_BAR = 0.90
DEFAULT_MIN_COMPONENT_SHARE = 0.5
AUDIT_MODE = "read_only_import_only_shadow"
PROMOTES_ROUTING = False
WRITES_STATE = False

_VERIFIED_RUN_TOKENS = frozenset(str(run) for run in VERIFIED_CACHE_RUNS)

CASE_ID_KEYS = ("case_id", "case", "study_id", "patient_key")
SLICE_INDEX_KEYS = ("slice_index", "slice", "z")
POSITIVE_KEYS = ("myo_topology", "positive", "flag")
SEVERITY_KEYS = ("severity", "severity_score", "score")
RING_KEYS = ("ring_continuous", "ring_contiguous", "ring_closed")
COMPONENT_KEYS = ("n_components", "components", "connected_components")
MAX_SHARE_KEYS = ("max_component_share", "largest_component_share", "max_cc_share")


def _first_present(record, keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def slice_key(record):
    """Frozen slice identity: exactly one record per (case_id, slice_index)."""
    return (str(record["case_id"]), int(record["slice_index"]))


def is_verified_run(run_id):
    """Only the already-verified run ids may be imported for myo_topology."""
    return str(run_id) in _VERIFIED_RUN_TOKENS


def canonical_payload_hash(payload):
    """Deterministic sha256 over a JSON-canonical form of the payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def severity_stratum(score):
    """Severity stratification used for the worst-slice report."""
    if score is None:
        return "none"
    value = _safe_float(score, 0.0)
    if value <= 0.0:
        return "none"
    if value >= SEVERITY_HIGH:
        return "high"
    if value >= SEVERITY_MODERATE:
        return "moderate"
    return "low"


def _run9_b1_contract(payload):
    """The one and only extractor for myo_topology provenance records."""
    if isinstance(payload, Mapping):
        records = payload.get("records")
        if records is None:
            records = payload.get("slices")
        if records is None:
            records = payload.get("slice_records")
        if records is None:
            records = [payload] if _first_present(payload, CASE_ID_KEYS) is not None else []
    else:
        records = list(payload or [])
    return [_normalize_record(raw) for raw in records]


EXTRACTORS = {RUN9_B1_EXTRACTOR: _run9_b1_contract}


def assert_single_extractor(registry=None):
    """Guard the single-tool contract: no second extraction tool set."""
    reg = EXTRACTORS if registry is None else registry
    if len(reg) != 1:
        raise RuntimeError(
            "myo_topology shadow audit is single-extractor by contract: found %d extractors"
            % (len(reg),)
        )
    return next(iter(reg))


def _normalize_record(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("run-9 B1 contract violation: record is not a mapping")
    case_raw = _first_present(raw, CASE_ID_KEYS)
    if case_raw is None:
        raise ValueError("run-9 B1 contract violation: record has no case identifier")
    slice_raw = _first_present(raw, SLICE_INDEX_KEYS)
    if slice_raw is None:
        raise ValueError("run-9 B1 contract violation: record has no slice index")

    flags = {}
    raw_flags = raw.get("flags")
    if isinstance(raw_flags, Mapping):
        flags.update({str(key): bool(value) for key, value in raw_flags.items()})
    for flag in CO_OCCURRENCE_FLAGS:
        if flag in raw:
            flags[flag] = bool(raw[flag])
        else:
            flags.setdefault(flag, False)

    return {
        "case_id": str(case_raw),
        "slice_index": _safe_int(slice_raw, 0),
        "positive": bool(_first_present(raw, POSITIVE_KEYS, False)),
        "severity": _safe_float(_first_present(raw, SEVERITY_KEYS, 0.0), 0.0),
        "ring_continuous": bool(_first_present(raw, RING_KEYS, False)),
        "components": _safe_int(_first_present(raw, COMPONENT_KEYS, 0), 0),
        "max_component_share": _safe_float(_first_present(raw, MAX_SHARE_KEYS, 1.0), 1.0),
        "flags": flags,
    }


def extract_b1_contract(payload):
    """Import-only ingestion path; always routed through the single extractor."""
    extractor_name = assert_single_extractor()
    return EXTRACTORS[extractor_name](payload)


def _records_of(summary):
    if isinstance(summary, Mapping):
        for key in ("records", "slices", "slice_records"):
            value = summary.get(key)
            if value is not None:
                return list(value)
        if _first_present(summary, CASE_ID_KEYS) is not None:
            return [summary]
        return []
    return list(summary or [])


def freeze_slices(records):
    """Order the slice set deterministically and enforce single-record dedup."""
    seen = {}
    duplicates = []
    for position, record in enumerate(records):
        key = slice_key(record)
        if key in seen:
            duplicates.append(
                {
                    "key": key,
                    "first_position": seen[key]["position"],
                    "duplicate_position": position,
                }
            )
            continue
        seen[key] = {"position": position, "record": record}
    frozen = tuple(seen[key]["record"] for key in sorted(seen))
    return frozen, tuple(duplicates)


def per_case_topology(slices):
    """Per-case slice count, positive rate, worst slice index and stratum."""
    by_case = {}
    for record in slices:
        by_case.setdefault(record["case_id"], []).append(record)

    report = {}
    for case_id in sorted(by_case):
        ordered = sorted(by_case[case_id], key=lambda record: record["slice_index"])
        positives = [record for record in ordered if record["positive"]]
        count = len(ordered)
        positive_count = len(positives)
        worst = None
        if positives:
            worst_record = min(
                positives, key=lambda record: (-record["severity"], record["slice_index"])
            )
            worst = {
                "slice_index": worst_record["slice_index"],
                "severity": worst_record["severity"],
                "stratum": severity_stratum(worst_record["severity"]),
            }
        report[case_id] = {
            "case_id": case_id,
            "slice_count": count,
            "positive_slices": positive_count,
            "positive_rate": round(positive_count / count, 6) if count else 0.0,
            "worst_slice": worst,
            "severity_stratum": severity_stratum(
                max((record["severity"] for record in positives), default=0.0)
            ),
            "positive_slice_indices": tuple(record["slice_index"] for record in positives),
        }
    return report


def same_slice_cooccurrence(slices, flags=CO_OCCURRENCE_FLAGS):
    """Same-slice co-occurrence of a positive myo_topology with the MI flags."""
    positives = [record for record in slices if record["positive"]]
    result = {}
    for flag in flags:
        co = [record for record in positives if bool(record["flags"].get(flag, False))]
        result[flag] = {
            "flag": flag,
            "same_slice_positive_and_flag": len(co),
            "positive_slices": len(positives),
            "cooccurrence_rate": round(len(co) / len(positives), 6) if positives else 0.0,
            "cases": tuple(sorted({record["case_id"] for record in co})),
            "slice_keys": tuple(slice_key(record) for record in co),
        }
    return result


def continuous_positive_runs(slices, case_id=None):
    """Cross-slice continuous positive run lengths per case."""
    by_case = {}
    for record in slices:
        if not record["positive"]:
            continue
        by_case.setdefault(record["case_id"], []).append(record["slice_index"])

    result = {}
    for current in sorted(by_case):
        indices = sorted(by_case[current])
        runs = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            runs.append((start, previous))
            start = previous = index
        runs.append((start, previous))
        result[current] = {
            "runs": tuple(runs),
            "run_count": len(runs),
            "max_run_length": max(end - begin + 1 for begin, end in runs),
        }

    if case_id is not None:
        return result.get(case_id, {"runs": (), "run_count": 0, "max_run_length": 0})
    return result


def noop_arm(slices, **_kwargs):
    """Read-only baseline arm: the imported verdict is never modified."""
    return frozenset(slice_key(record) for record in slices if record["positive"])


def ring_continuity_arm(slices, **_kwargs):
    """Read-only arm: keep a positive slice only when the ring is continuous."""
    return frozenset(
        slice_key(record)
        for record in slices
        if record["positive"] and record["ring_continuous"]
    )


def max_component_arm(slices, min_component_share=DEFAULT_MIN_COMPONENT_SHARE, **_kwargs):
    """Read-only arm: keep a positive slice unless its largest connected
    component covers less than ``min_component_share`` of the slice (i.e. the
    positive is fragment-dominated).  Records without component metadata abstain
    and keep the no-op verdict; nothing is invented."""
    supported = set()
    for record in slices:
        if not record["positive"]:
            continue
        has_metadata = record["components"] > 0
        if not has_metadata or record["max_component_share"] >= min_component_share:
            supported.add(slice_key(record))
    return frozenset(supported)


DEFAULT_ARMS = {
    "noop": noop_arm,
    "ring_continuity": ring_continuity_arm,
    "max_component": max_component_arm,
}


def arm_agreement(slices, arms=None, baseline="noop", **arm_kwargs):
    """Exact per-slice agreement of every read-only arm with the no-op arm."""
    arm_map = dict(DEFAULT_ARMS if arms is None else arms)
    if baseline not in arm_map:
        raise ValueError("baseline arm %r is not registered" % (baseline,))

    frozen = tuple(slices)
    keys = tuple(slice_key(record) for record in frozen)
    baseline_set = frozenset(arm_map[baseline](frozen, **arm_kwargs))

    results = {}
    for name in sorted(arm_map):
        if name == baseline:
            continue
        decision = frozenset(arm_map[name](frozen, **arm_kwargs))
        disagreements = tuple(
            key for key in keys if (key in baseline_set) != (key in decision)
        )
        agreement = 1.0 if not keys else round((len(keys) - len(disagreements)) / len(keys), 6)
        results[name] = {
            "agreement": agreement,
            "n_slices": len(keys),
            "disagreements": disagreements,
        }
    return results


def verify_cache_hashes(caches, expected_hashes=None):
    """Canonical sha256 verification of each imported cache summary."""
    details = {}
    for run_id in sorted(caches, key=str):
        payload = caches[run_id]
        digest = canonical_payload_hash(payload)
        expectation = None
        if expected_hashes is not None:
            expectation = expected_hashes.get(run_id, expected_hashes.get(str(run_id)))
        if expectation is None and isinstance(payload, Mapping):
            expectation = payload.get("payload_hash")
        if expectation is None and isinstance(payload, Mapping):
            expectation = payload.get("hash")
        present = expectation is not None
        details[run_id] = {
            "run_id": run_id,
            "hash": digest,
            "expected": expectation,
            "expectation_present": present,
            "verified": bool(present and str(expectation) == digest),
        }
    return details


def _per_case_complete(per_case, n_slices):
    if n_slices <= 0 or not per_case:
        return False
    for case in per_case.values():
        if case["slice_count"] < 1:
            return False
        if not 0.0 <= case["positive_rate"] <= 1.0:
            return False
        if case["positive_slices"] > 0 and case["worst_slice"] is None:
            return False
        if case["severity_stratum"] not in SEVERITY_STRATA:
            return False
    return True


def run_shadow_audit(
    caches=None,
    expected_hashes=None,
    records=None,
    arms=None,
    arm_bar=DEFAULT_ARM_AGREEMENT_BAR,
    min_component_share=DEFAULT_MIN_COMPONENT_SHARE,
):
    """Run the bounded, read-only, import-only shadow audit.

    ``caches`` maps run id -> already-verified cache summary.  Only run ids in
    :data:`VERIFIED_CACHE_RUNS` are imported; unknown runs are reported and
    skipped, never re-derived.  A hash mismatch routes that summary through the
    single run-9 B1 contract extractor and is reported as a fallback.
    """
    extractor_name = assert_single_extractor()

    cache_details = verify_cache_hashes(caches, expected_hashes) if caches else {}
    verified_run_ids = []
    mismatched_run_ids = []
    unknown_run_ids = []
    fallback_run_ids = []
    imported_raw = []

    for run_id in sorted(cache_details, key=str):
        entry = cache_details[run_id]
        if not is_verified_run(entry["run_id"]):
            unknown_run_ids.append(entry["run_id"])
            continue
        if entry["verified"]:
            verified_run_ids.append(entry["run_id"])
        else:
            mismatched_run_ids.append(entry["run_id"])
            fallback_run_ids.append(entry["run_id"])
        imported_raw.extend(_records_of(caches[entry["run_id"]]))

    fallback = {
        "used": bool(fallback_run_ids),
        "extractor": extractor_name if fallback_run_ids else None,
        "run_ids": tuple(fallback_run_ids),
    }

    if records is not None:
        source = "explicit_records"
        batch = _records_of(records) if isinstance(records, Mapping) else list(records)
    else:
        source = "imported_cache_summaries"
        batch = imported_raw

    frozen, duplicates = freeze_slices(extract_b1_contract(batch))
    before_hash = canonical_payload_hash(list(frozen))

    per_case = per_case_topology(frozen)
    co_occurrence = same_slice_cooccurrence(frozen)
    positive_runs = continuous_positive_runs(frozen)
    agreement = arm_agreement(
        frozen, arms=arms, min_component_share=min_component_share
    )

    after_hash = canonical_payload_hash(list(frozen))
    mutation_free = before_hash == after_hash

    per_case_complete = _per_case_complete(per_case, len(frozen))
    cooccurrence_computable = all(
        "cooccurrence_rate" in entry for entry in co_occurrence.values()
    )
    arms_meet_bar = bool(frozen) and all(
        entry["agreement"] >= arm_bar for entry in agreement.values()
    )
    cache_hash_ok = not mismatched_run_ids
    dedup_compliant = not duplicates
    meets_bar = bool(
        per_case_complete
        and cooccurrence_computable
        and arms_meet_bar
        and cache_hash_ok
        and dedup_compliant
        and mutation_free
    )

    return {
        "rule": MYO_TOPOLOGY_RULE,
        "mode": AUDIT_MODE,
        "source": source,
        "promotes_routing": PROMOTES_ROUTING,
        "writes_state": WRITES_STATE,
        "n_cases": len(per_case),
        "n_slices": len(frozen),
        "per_case": per_case,
        "co_occurrence": co_occurrence,
        "positive_runs": positive_runs,
        "arms": {
            "bar": arm_bar,
            "min_component_share": min_component_share,
            "agreement": agreement,
            "meets_bar": arms_meet_bar,
        },
        "cache": {
            "checked": bool(cache_details),
            "verified_run_ids": tuple(verified_run_ids),
            "mismatched_run_ids": tuple(mismatched_run_ids),
            "unknown_run_ids": tuple(unknown_run_ids),
            "details": cache_details,
        },
        "fallback": fallback,
        "dedup": {
            "compliant": dedup_compliant,
            "duplicate_count": len(duplicates),
            "duplicates": duplicates,
        },
        "mutation_free": mutation_free,
        "acceptance": {
            "per_case_complete": per_case_complete,
            "cooccurrence_computable": cooccurrence_computable,
            "arms_meet_bar": arms_meet_bar,
            "cache_hash_ok": cache_hash_ok,
            "dedup_compliant": dedup_compliant,
            "meets_bar": meets_bar,
        },
        "notes": (
            "myo_topology has no dedicated main route in iteration_history; audit is shadow/report-only.",
            "imports reuse verified run-4/5/9/13-18 cache summaries only; unknown runs are skipped, not re-derived.",
            "single extractor contract: %s" % (extractor_name,),
            "hash mismatch falls back to the same single extractor and is reported.",
            "no writes, no routing promotion, no QC gate change.",
        ),
    }
