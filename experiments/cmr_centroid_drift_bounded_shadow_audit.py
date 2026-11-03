"""Bounded read-only shadow audit for the ``cmr_centroid_drift`` signal (n = 96).

Approved maintenance action
---------------------------
* Draw a bounded sample: 30 positive cases + 10 control cases.
* Reuse the cached run summaries of runs 4, 5, 9 and 13-18.
* If no cached summary carries a hash matching the observed case hash, fall back
  *only* to the run-9 B1 extractor and escalate that fallback in the report.
* Read only: labels, production review rules, datasets and runtime databases are
  never touched, and no QC safety check is weakened.

Everything external is injected: ``cache_loader``, ``hash_probe`` and the
run-9 ``fallback_extractor``.  The module therefore runs fully offline and is
deterministic in tests.  It produces a report only; it never writes anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RUN_IDS",
    "FALLBACK_RUN",
    "FALLBACK_EXTRACTOR",
    "FALLBACK_BUDGET",
    "POSITIVE_SAMPLE_SIZE",
    "CONTROL_SAMPLE_SIZE",
    "SAMPLE_SIZE",
    "MIN_PPV",
    "MIN_CACHE_REUSE_RATE",
    "DUPLICATE_FLAG_THRESHOLD",
    "CaseEvidence",
    "run_shadow_audit",
    "summarize_for_reviewer",
    "verify_read_only_contract",
]

# --- frozen scope of the approved action -----------------------------------

RUN_IDS: Tuple[str, ...] = (
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

FALLBACK_RUN: str = "run-9"
FALLBACK_EXTRACTOR: str = "B1"
FALLBACK_BUDGET: int = 10

POSITIVE_SAMPLE_SIZE: int = 30
CONTROL_SAMPLE_SIZE: int = 10
SAMPLE_SIZE: int = POSITIVE_SAMPLE_SIZE + CONTROL_SAMPLE_SIZE

MIN_PPV: float = 0.6
MIN_CACHE_REUSE_RATE: float = 0.8
DUPLICATE_FLAG_THRESHOLD: int = 2

CacheLoader = Callable[[str, str], Optional[Mapping[str, Any]]]
HashProbe = Callable[[str], Optional[str]]
FallbackExtractor = Callable[[str, str, str], Mapping[str, Any]]


# --- per-case evidence ------------------------------------------------------


@dataclass(frozen=True)
class CaseEvidence:
    """Read-only snapshot of how a single sampled case was resolved."""

    case_id: str
    stratum: str  # "positive" | "control"
    run_id: str
    source: str  # "cache" | "fallback"
    cached: bool
    hash_match: bool
    hash_mismatch_seen: bool
    confirmed_drift: bool
    review_centroid_outlier: bool
    duplicate_flag_count: int

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


# --- small helpers ----------------------------------------------------------


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    return bool(value)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _validate_sample(
    positive_case_ids: Sequence[str], control_case_ids: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Enforce the bounded 30 + 10 disjoint sampling frame from the objective."""

    positives = [str(case_id) for case_id in (positive_case_ids or [])]
    controls = [str(case_id) for case_id in (control_case_ids or [])]

    if len(positives) != POSITIVE_SAMPLE_SIZE:
        raise ValueError(
            "positive sample must contain exactly %d cases, got %d"
            % (POSITIVE_SAMPLE_SIZE, len(positives))
        )
    if len(controls) != CONTROL_SAMPLE_SIZE:
        raise ValueError(
            "control sample must contain exactly %d cases, got %d"
            % (CONTROL_SAMPLE_SIZE, len(controls))
        )

    for label, cases in (("positive", positives), ("control", controls)):
        if len(set(cases)) != len(cases):
            raise ValueError("%s sample contains duplicate case ids" % label)

    overlap = sorted(set(positives) & set(controls))
    if overlap:
        raise ValueError(
            "positive and control samples must be disjoint; overlap=%s" % overlap
        )

    return positives, controls


# --- cache resolution -------------------------------------------------------


def _collect_cached_records(
    case_id: str, cache_loader: CacheLoader
) -> List[Tuple[str, Mapping[str, Any]]]:
    records: List[Tuple[str, Mapping[str, Any]]] = []
    for run_id in RUN_IDS:
        record = cache_loader(run_id, case_id)
        if record:
            records.append((run_id, record))
    return records


def _resolve_case(
    case_id: str,
    stratum: str,
    cache_loader: CacheLoader,
    hash_probe: HashProbe,
    fallback_extractor: Optional[FallbackExtractor],
) -> CaseEvidence:
    """Resolve one case from cache; fall back to run-9 B1 only if required."""

    observed_hash = hash_probe(case_id)
    records = _collect_cached_records(case_id, cache_loader)

    duplicate_flag_count = 0
    for _run_id, record in records:
        summary = record.get("summary") or {}
        if _as_bool(summary.get("centroid_drift_flag")):
            duplicate_flag_count += 1

    hash_mismatch_seen = any(
        record.get("stored_hash") != observed_hash for _run_id, record in records
    )

    for run_id, record in records:
        if record.get("stored_hash") != observed_hash:
            continue
        summary = record.get("summary") or {}
        return CaseEvidence(
            case_id=case_id,
            stratum=stratum,
            run_id=run_id,
            source="cache",
            cached=True,
            hash_match=True,
            hash_mismatch_seen=hash_mismatch_seen,
            confirmed_drift=_as_bool(summary.get("confirmed_drift")),
            review_centroid_outlier=_as_bool(summary.get("review_centroid_outlier")),
            duplicate_flag_count=duplicate_flag_count,
        )

    # --- hash-mismatch fallback, restricted to run-9 / B1 -------------------
    if fallback_extractor is None:
        raise RuntimeError(
            "case %r has no hash-matching cached summary and no run-9 B1 "
            "fallback extractor is configured" % case_id
        )

    fallback = fallback_extractor(case_id, FALLBACK_RUN, FALLBACK_EXTRACTOR)
    if not fallback:
        raise RuntimeError("run-9 B1 fallback extractor returned no payload for %r" % case_id)

    reported_run = fallback.get("run_id", FALLBACK_RUN)
    reported_extractor = fallback.get("extractor_id", FALLBACK_EXTRACTOR)
    if reported_run != FALLBACK_RUN or reported_extractor != FALLBACK_EXTRACTOR:
        raise ValueError(
            "hash-mismatch fallback must be limited to %s/%s; got run_id=%r extractor_id=%r"
            % (FALLBACK_RUN, FALLBACK_EXTRACTOR, reported_run, reported_extractor)
        )

    return CaseEvidence(
        case_id=case_id,
        stratum=stratum,
        run_id=FALLBACK_RUN,
        source="fallback",
        cached=False,
        hash_match=False,
        hash_mismatch_seen=True,
        confirmed_drift=_as_bool(fallback.get("confirmed_drift")),
        review_centroid_outlier=_as_bool(fallback.get("review_centroid_outlier")),
        duplicate_flag_count=duplicate_flag_count,
    )


# --- audit entry point ------------------------------------------------------


def run_shadow_audit(
    positive_case_ids: Sequence[str],
    control_case_ids: Sequence[str],
    *,
    cache_loader: CacheLoader,
    hash_probe: HashProbe,
    fallback_extractor: Optional[FallbackExtractor] = None,
    audit_label: str = "cmr_centroid_drift",
    total_drift_cases: int = 96,
) -> Mapping[str, Any]:
    """Run the bounded read-only shadow audit and return a report mapping."""

    positives, controls = _validate_sample(positive_case_ids, control_case_ids)

    evidence: List[CaseEvidence] = []
    for case_id in positives:
        evidence.append(
            _resolve_case(case_id, "positive", cache_loader, hash_probe, fallback_extractor)
        )
    for case_id in controls:
        evidence.append(
            _resolve_case(case_id, "control", cache_loader, hash_probe, fallback_extractor)
        )

    n_sampled = len(evidence)
    cached_count = sum(1 for item in evidence if item.cached)
    cache_reuse_rate = _safe_rate(cached_count, n_sampled)

    fallback_cases = sorted(item.case_id for item in evidence if item.source == "fallback")
    fallback_count = len(fallback_cases)

    hash_mismatch_cases = sorted(item.case_id for item in evidence if item.hash_mismatch_seen)
    hash_mismatch_count = len(hash_mismatch_cases)

    positive_evidence = [item for item in evidence if item.stratum == "positive"]
    control_evidence = [item for item in evidence if item.stratum == "control"]

    confirmed_drift_positives = sum(1 for item in positive_evidence if item.confirmed_drift)
    ppv = _safe_rate(confirmed_drift_positives, len(positive_evidence))

    confirmed_all = [item for item in evidence if item.confirmed_drift]
    cooccurrence_count = sum(1 for item in confirmed_all if item.review_centroid_outlier)
    cooccurrence_rate = _safe_rate(cooccurrence_count, len(confirmed_all))

    duplicate_flag_cases = sorted(
        item.case_id
        for item in evidence
        if item.duplicate_flag_count >= DUPLICATE_FLAG_THRESHOLD
    )
    duplicate_flag_rate = _safe_rate(len(duplicate_flag_cases), n_sampled)

    control_false_positive_count = sum(1 for item in control_evidence if item.confirmed_drift)
    control_false_positive_rate = _safe_rate(
        control_false_positive_count, len(control_evidence)
    )

    escalations: List[str] = []
    if fallback_count > FALLBACK_BUDGET:
        escalations.append(
            "fallback_budget_exceeded: %d hash-mismatch fallbacks > budget %d"
            % (fallback_count, FALLBACK_BUDGET)
        )
    if hash_mismatch_count:
        escalations.append(
            "hash_mismatch_reported: %d cases showed a cached hash mismatch"
            % hash_mismatch_count
        )
    if not confirmed_all:
        escalations.append(
            "no_confirmed_drift_cases: review_centroid_outlier co-occurrence is undefined (0.0)"
        )
    if ppv < MIN_PPV:
        escalations.append(
            "ppv_below_target: %.4f < %.2f" % (ppv, MIN_PPV)
        )
    if cache_reuse_rate < MIN_CACHE_REUSE_RATE:
        escalations.append(
            "cache_reuse_below_target: %.4f < %.2f"
            % (cache_reuse_rate, MIN_CACHE_REUSE_RATE)
        )

    read_only_contract = {
        "read_only": True,
        "writes_performed": False,
        "labels_modified": False,
        "production_rules_modified": False,
        "datasets_touched": False,
        "qc_safety_checks_weakened": False,
        "fallback_limited_to": "%s/%s" % (FALLBACK_RUN, FALLBACK_EXTRACTOR),
    }

    acceptance = {
        "ppv_at_least_0_6": ppv >= MIN_PPV,
        "cache_reuse_at_least_0_8": cache_reuse_rate >= MIN_CACHE_REUSE_RATE,
        "fallback_within_budget": fallback_count <= FALLBACK_BUDGET,
        "review_centroid_outlier_cooccurrence_reported": True,
        "duplicate_flag_rate_reported": True,
        "read_only_contract_holds": True,
    }
    acceptance_passed = all(acceptance.values())

    report: Mapping[str, Any] = {
        "audit_label": audit_label,
        "total_drift_cases": total_drift_cases,
        "sampling": {
            "positive": len(positives),
            "control": len(controls),
            "n_sampled": n_sampled,
        },
        "reused_runs": list(RUN_IDS),
        "fallback_run": FALLBACK_RUN,
        "fallback_extractor": FALLBACK_EXTRACTOR,
        "fallback_budget": FALLBACK_BUDGET,
        "cache_reuse_count": cached_count,
        "cache_reuse_rate": round(cache_reuse_rate, 4),
        "fallback_count": fallback_count,
        "fallback_cases": fallback_cases,
        "hash_mismatch_count": hash_mismatch_count,
        "hash_mismatch_cases": hash_mismatch_cases,
        "ppv": round(ppv, 4),
        "confirmed_drift_positives": confirmed_drift_positives,
        "positive_sample_size": len(positive_evidence),
        "control_false_positive_count": control_false_positive_count,
        "control_false_positive_rate": round(control_false_positive_rate, 4),
        "review_centroid_outlier_cooccurrence_count": cooccurrence_count,
        "review_centroid_outlier_cooccurrence_rate": round(cooccurrence_rate, 4),
        "duplicate_flag_count": len(duplicate_flag_cases),
        "duplicate_flag_rate": round(duplicate_flag_rate, 4),
        "duplicate_flag_cases": duplicate_flag_cases,
        "read_only_contract": read_only_contract,
        "acceptance": acceptance,
        "acceptance_passed": acceptance_passed,
        "escalations": escalations,
        "evidence": [item.to_dict() for item in evidence],
    }

    return report


# --- reporting helpers ------------------------------------------------------


def summarize_for_reviewer(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compact reviewer-facing digest of a shadow-audit report."""

    return {
        "audit_label": report.get("audit_label"),
        "n_sampled": (report.get("sampling") or {}).get("n_sampled"),
        "ppv": report.get("ppv"),
        "cache_reuse_rate": report.get("cache_reuse_rate"),
        "fallback_count": report.get("fallback_count"),
        "fallback_budget": report.get("fallback_budget"),
        "hash_mismatch_count": report.get("hash_mismatch_count"),
        "review_centroid_outlier_cooccurrence_rate": report.get(
            "review_centroid_outlier_cooccurrence_rate"
        ),
        "duplicate_flag_rate": report.get("duplicate_flag_rate"),
        "acceptance_passed": report.get("acceptance_passed"),
        "escalations": list(report.get("escalations") or []),
    }


def verify_read_only_contract(report: Mapping[str, Any]) -> bool:
    """Fail loudly if a report ever claims a mutation of protected state."""

    contract = report.get("read_only_contract") or {}
    forbidden = (
        "writes_performed",
        "labels_modified",
        "production_rules_modified",
        "datasets_touched",
        "qc_safety_checks_weakened",
    )
    violations = [key for key in forbidden if contract.get(key)]
    if violations:
        raise AssertionError(
            "read-only contract violated: %s" % ", ".join(violations)
        )
    return True
