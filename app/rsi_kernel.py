"""Mechanism-level recursive self-improvement primitives.

This module deliberately does not know how to judge a medical label.  It
compares two versions only when an external evaluator has measured both on the
same immutable benchmark.  That separation prevents model conversation,
token spend, code churn, or a passing unit test from being mistaken for
domain improvement.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal


Decision = Literal["PROMOTE", "RETAIN_PARENT", "NEEDS_EVIDENCE"]
Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    direction: Direction
    tolerance: float = 0.0
    minimum_delta: float = 0.0
    required: bool = True
    # Cost metrics can break a quality tie, but cannot by themselves establish
    # that the system became better at its assigned task.
    quality_metric: bool = True

    def __post_init__(self):
        if self.direction not in {"higher", "lower"}:
            raise ValueError("invalid metric direction")
        if (not math.isfinite(self.tolerance) or self.tolerance < 0
                or not math.isfinite(self.minimum_delta) or self.minimum_delta < 0):
            raise ValueError("metric tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class EvolutionDecision:
    decision: Decision
    reasons: tuple[str, ...]
    deltas: dict[str, float]
    improved_metrics: tuple[str, ...]
    regressed_metrics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "deltas": self.deltas,
            "improved_metrics": list(self.improved_metrics),
            "regressed_metrics": list(self.regressed_metrics),
        }


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def variant_id(
    *, parent_variant_id: str | None, source_run_id: int | None,
    action_id: str | None, code_commit: str | None,
    changed_files: list[str], artifact_digest: str | None = None,
) -> str:
    """Return a deterministic identity for a candidate and its ancestry."""
    identity = {
        "parent_variant_id": parent_variant_id,
        "source_run_id": source_run_id,
        "action_id": action_id,
        "code_commit": code_commit,
        "changed_files": sorted(set(changed_files)),
        "artifact_digest": artifact_digest,
    }
    return "v_" + canonical_hash(identity)[:20]


def compare_fitness(
    *, baseline: dict[str, Any], candidate: dict[str, Any],
    baseline_benchmark_id: str, candidate_benchmark_id: str,
    baseline_benchmark_hash: str, candidate_benchmark_hash: str,
    metric_specs: dict[str, MetricSpec],
    constraints: dict[str, bool] | None = None,
) -> EvolutionDecision:
    """Compare a child with its parent using Pareto-style selection.

    Promotion requires all hard constraints, the exact same benchmark identity,
    no material regression, and an improvement in at least one task-quality
    metric.  Missing or incomparable evidence is not converted into a zero.
    """
    reasons: list[str] = []
    if not all((baseline_benchmark_id, candidate_benchmark_id,
                baseline_benchmark_hash, candidate_benchmark_hash)):
        reasons.append("benchmark_identity_missing")
    if not metric_specs or not any(s.quality_metric and s.required for s in metric_specs.values()):
        reasons.append("required_quality_metric_missing")
    if baseline_benchmark_id != candidate_benchmark_id:
        reasons.append("benchmark_id_mismatch")
    if baseline_benchmark_hash != candidate_benchmark_hash:
        reasons.append("benchmark_hash_mismatch")
    failed_constraints = sorted(
        name for name, passed in (constraints or {}).items() if passed is not True
    )
    if failed_constraints:
        reasons.extend(f"constraint_failed:{name}" for name in failed_constraints)
    if reasons:
        return EvolutionDecision(
            decision="NEEDS_EVIDENCE",
            reasons=tuple(reasons), deltas={}, improved_metrics=(),
            regressed_metrics=(),
        )

    missing = sorted(
        name for name, spec in metric_specs.items()
        if spec.required and (name not in baseline or name not in candidate)
    )
    if missing:
        return EvolutionDecision(
            decision="NEEDS_EVIDENCE",
            reasons=tuple(f"metric_missing:{name}" for name in missing),
            deltas={}, improved_metrics=(), regressed_metrics=(),
        )

    deltas: dict[str, float] = {}
    improved: list[str] = []
    regressed: list[str] = []
    quality_improved = False
    for name, spec in metric_specs.items():
        if name not in baseline or name not in candidate:
            continue
        try:
            if isinstance(baseline[name], bool) or isinstance(candidate[name], bool):
                raise ValueError("boolean is not a measurement")
            before = float(baseline[name])
            after = float(candidate[name])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("nonfinite measurement")
        except (TypeError, ValueError):
            return EvolutionDecision(
                decision="NEEDS_EVIDENCE",
                reasons=(f"metric_not_numeric:{name}",), deltas={},
                improved_metrics=(), regressed_metrics=(),
            )
        signed = after - before if spec.direction == "higher" else before - after
        deltas[name] = round(signed, 12)
        if signed < -abs(spec.tolerance):
            regressed.append(name)
        elif signed > max(spec.tolerance, spec.minimum_delta):
            improved.append(name)
            quality_improved |= spec.quality_metric

    if regressed:
        return EvolutionDecision(
            decision="RETAIN_PARENT",
            reasons=tuple(f"metric_regressed:{name}" for name in sorted(regressed)),
            deltas=deltas, improved_metrics=tuple(sorted(improved)),
            regressed_metrics=tuple(sorted(regressed)),
        )
    if not quality_improved:
        return EvolutionDecision(
            decision="RETAIN_PARENT", reasons=("no_quality_improvement",),
            deltas=deltas, improved_metrics=tuple(sorted(improved)),
            regressed_metrics=(),
        )
    return EvolutionDecision(
        decision="PROMOTE", reasons=("pareto_quality_improvement",),
        deltas=deltas, improved_metrics=tuple(sorted(improved)),
        regressed_metrics=(),
    )


def default_metric_specs() -> dict[str, MetricSpec]:
    """Provider-neutral fitness dimensions used by the orchestration layer.

    Domain evaluators may add stricter metrics.  They must not silently remove
    these dimensions when the values are available.
    """
    return {
        "resolved_items": MetricSpec("higher", minimum_delta=0.0),
        "holdout_score": MetricSpec("higher", tolerance=0.0),
        "false_routes": MetricSpec("lower", tolerance=0.0),
        "regressions": MetricSpec("lower", tolerance=0.0),
        "human_review_items": MetricSpec("lower", tolerance=0.0),
        "cost_tokens": MetricSpec(
            "lower", tolerance=0.0, required=False, quality_metric=False
        ),
    }


def selection_payload(
    *, parent_variant_id: str | None, child_variant_id: str,
    benchmark_id: str, benchmark_hash: str,
    baseline: dict[str, Any], candidate: dict[str, Any],
    constraints: dict[str, bool], decision: EvolutionDecision,
) -> dict[str, Any]:
    """Create the durable, model-independent selection record."""
    return {
        "schema_version": "rsi-selection-v1",
        "parent_variant_id": parent_variant_id,
        "child_variant_id": child_variant_id,
        "benchmark_id": benchmark_id,
        "benchmark_hash": benchmark_hash,
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "constraints": constraints,
        **decision.as_dict(),
    }
