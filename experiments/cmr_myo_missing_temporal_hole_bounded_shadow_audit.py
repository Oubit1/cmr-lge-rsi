"""Bounded read-only shadow audit for cmr_myo_missing_temporal_hole.

The module is deliberately offline: it consumes already-collected shadow-audit
cases and only validates cache reuse, hash-mismatch routing, deduplication,
slice-level evidence traceability, and classification metrics with Wilson
confidence intervals. It never mutates labels, production rules, datasets,
credentials, or runtime databases.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

TARGET_LABEL = "cmr_myo_missing_temporal_hole"
TARGET_N = 242
UNCOVERED_RUNS: Tuple[int, ...] = tuple(range(28, 37))
CACHE_RUNS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)
B1_EXTRACTOR_RUN = 9
B1_EXTRACTOR = "run-9 B1"
POSITIVE_N = 40
CONTROL_N = 10
MIN_CACHE_REUSE = 0.95
Z_95 = 1.959963984540054


class ShadowAuditViolation(ValueError):
    """Raised when a bounded shadow-audit acceptance guard fails."""


@dataclass(frozen=True)
class ShadowAuditCase:
    """One pre-collected shadow-audit observation.

    The module treats ``label_positive`` as ground truth supplied by the audit
    harness and never changes it. Positives must be label-positive and controls
    must not be label-positive; anything else is treated as a label mutation.
    """

    case_id: str
    stratum: str  # "positive" or "control"
    covered_run: int
    slice_evidence_refs: Tuple[str, ...]
    hash_ok: bool
    extractor_route: str
    predicted_positive: bool
    label_positive: bool
    audit_record_id: str


@dataclass(frozen=True)
class MutationGuard:
    """Caller-declared guard proving labels/production rules were not touched."""

    labels_modified: bool = False
    production_rules_modified: bool = False

    def violations(self) -> Tuple[str, ...]:
        out = []
        if self.labels_modified:
            out.append("labels_modified")
        if self.production_rules_modified:
            out.append("production_rules_modified")
        return tuple(out)


@dataclass(frozen=True)
class ProportionCI:
    numerator: int
    denominator: int
    value: float
    low: float
    high: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "low": self.low,
            "high": self.high,
        }


def wilson_interval(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Boundary cases are pinned to their exact mathematical values: k == 0 has a
    lower bound of exactly 0.0 and k == n has an upper bound of exactly 1.0.
    This avoids returning values such as 0.9999999999999999 that arise purely
    from floating-point cancellation in the closed-form expression.
    """

    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ShadowAuditViolation(f"invalid proportion numerator={k} denominator={n}")
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))) / denom
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    if k == 0:
        low = 0.0
    if k == n:
        high = 1.0
    return (low, high)


def proportion_ci(k: int, n: int) -> ProportionCI:
    low, high = wilson_interval(k, n)
    value = 0.0 if n <= 0 else k / n
    return ProportionCI(numerator=k, denominator=n, value=value, low=low, high=high)


def ensure_no_mutation(guard: MutationGuard | None) -> None:
    if guard is None:
        return
    violations = guard.violations()
    if violations:
        raise ShadowAuditViolation(
            "label/production-rule mutation guard failed: " + ", ".join(violations)
        )


def _validate_case(case: ShadowAuditCase) -> None:
    if case.stratum not in ("positive", "control"):
        raise ShadowAuditViolation(f"unknown stratum {case.stratum!r} for {case.case_id}")
    if not case.case_id:
        raise ShadowAuditViolation("case_id is required")
    if not case.audit_record_id:
        raise ShadowAuditViolation(f"audit_record_id is required for {case.case_id}")
    if not case.slice_evidence_refs:
        raise ShadowAuditViolation(f"slice-level evidence missing for {case.case_id}")
    if not case.hash_ok and case.extractor_route != B1_EXTRACTOR:
        raise ShadowAuditViolation(
            f"hash mismatch for {case.case_id} routed to {case.extractor_route!r}; "
            f"only {B1_EXTRACTOR!r} is permitted"
        )
    if case.stratum == "control" and case.label_positive:
        raise ShadowAuditViolation(
            f"control {case.case_id} has positive label; refusing label mutation"
        )
    if case.stratum == "positive" and not case.label_positive:
        raise ShadowAuditViolation(
            f"positive {case.case_id} lacks positive label; refusing label mutation"
        )


def compute_cache_reuse_rate(cases: Sequence[ShadowAuditCase]) -> float:
    if not cases:
        return 0.0
    cache_hits = sum(1 for c in cases if c.covered_run in CACHE_RUNS)
    return cache_hits / len(cases)


def detect_duplicate_audit_records(cases: Sequence[ShadowAuditCase]) -> Tuple[int, float]:
    counts = Counter(c.audit_record_id for c in cases)
    duplicate_record_count = sum(n - 1 for n in counts.values() if n > 1)
    total = len(cases)
    duplicate_rate = 0.0 if total == 0 else duplicate_record_count / total
    return duplicate_record_count, duplicate_rate


def compute_classification_metrics(cases: Sequence[ShadowAuditCase]) -> Dict[str, object]:
    positives = [c for c in cases if c.stratum == "positive"]
    controls = [c for c in cases if c.stratum == "control"]

    tp = sum(1 for c in positives if c.predicted_positive)
    fn = sum(1 for c in positives if not c.predicted_positive)
    fp = sum(1 for c in controls if c.predicted_positive)
    tn = sum(1 for c in controls if not c.predicted_positive)

    support = proportion_ci(tp, len(positives))
    precision = proportion_ci(tp, tp + fp)
    fpr = proportion_ci(fp, len(controls))

    return {
        "positive_n": len(positives),
        "control_n": len(controls),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "support_rate": support.as_dict(),
        "precision": precision.as_dict(),
        "false_positive_rate": fpr.as_dict(),
    }


def compute_audit_report(
    cases: Sequence[ShadowAuditCase],
    guard: MutationGuard | None = None,
) -> Dict[str, object]:
    cases = tuple(cases)
    for case in cases:
        _validate_case(case)

    cache_reuse_rate = compute_cache_reuse_rate(cases)
    duplicate_record_count, duplicate_record_rate = detect_duplicate_audit_records(cases)

    hash_mismatch_count = sum(1 for c in cases if not c.hash_ok)
    hash_mismatch_routed_to_b1 = sum(
        1 for c in cases if (not c.hash_ok) and c.extractor_route == B1_EXTRACTOR
    )
    non_b1_mismatch_routes = hash_mismatch_count - hash_mismatch_routed_to_b1

    evidence_traceable = sum(1 for c in cases if c.slice_evidence_refs)
    evidence_traceability_rate = (
        0.0 if not cases else evidence_traceable / len(cases)
    )

    mutation_violations = 0 if guard is None else len(guard.violations())

    return {
        "target_label": TARGET_LABEL,
        "target_n": TARGET_N,
        "uncovered_runs": list(UNCOVERED_RUNS),
        "cache_runs": list(CACHE_RUNS),
        "b1_extractor": B1_EXTRACTOR,
        "b1_extractor_run": B1_EXTRACTOR_RUN,
        "case_count": len(cases),
        "cache_reuse_rate": cache_reuse_rate,
        "duplicate_record_count": duplicate_record_count,
        "duplicate_record_rate": duplicate_record_rate,
        "hash_mismatch_count": hash_mismatch_count,
        "hash_mismatch_routed_to_b1": hash_mismatch_routed_to_b1,
        "non_b1_mismatch_routes": non_b1_mismatch_routes,
        "evidence_traceable_count": evidence_traceable,
        "evidence_traceability_rate": evidence_traceability_rate,
        "label_and_production_rule_changes": mutation_violations,
        "metrics": compute_classification_metrics(cases),
    }


def run_bounded_shadow_audit(
    cases: Sequence[ShadowAuditCase],
    guard: MutationGuard | None = None,
    expected_positive_n: int = POSITIVE_N,
    expected_control_n: int = CONTROL_N,
) -> Dict[str, object]:
    """Validate and report a bounded shadow audit.

    Raises ShadowAuditViolation if any acceptance guard fails. The function is
    read-only and does not persist or mutate any audit state.
    """

    report = compute_audit_report(cases, guard=guard)
    errors = []

    if report["cache_reuse_rate"] < MIN_CACHE_REUSE:
        errors.append(
            f"cache reuse {report['cache_reuse_rate']:.3f} < {MIN_CACHE_REUSE}"
        )
    if report["duplicate_record_count"] != 0:
        errors.append(f"duplicate audit records: {report['duplicate_record_count']}")
    if report["non_b1_mismatch_routes"] != 0:
        errors.append(
            f"hash mismatches not routed to {B1_EXTRACTOR}: "
            f"{report['non_b1_mismatch_routes']}"
        )
    if report["evidence_traceability_rate"] != 1.0:
        errors.append(
            f"slice evidence traceability {report['evidence_traceability_rate']:.3f} != 1.0"
        )
    if report["label_and_production_rule_changes"] != 0:
        errors.append(
            f"label/production rule changes: "
            f"{report['label_and_production_rule_changes']}"
        )

    metrics = report["metrics"]
    if metrics["positive_n"] != expected_positive_n:
        errors.append(
            f"positive count {metrics['positive_n']} != {expected_positive_n}"
        )
    if metrics["control_n"] != expected_control_n:
        errors.append(
            f"control count {metrics['control_n']} != {expected_control_n}"
        )

    if errors:
        raise ShadowAuditViolation("; ".join(errors))
    return report
