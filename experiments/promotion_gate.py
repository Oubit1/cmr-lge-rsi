from __future__ import annotations

from typing import Any

from contracts.models import PromotionDecision


def evaluate_metrics(baseline: dict[str, float], candidate: dict[str, float],
                     config: dict[str, Any]) -> PromotionDecision:
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    deltas: dict[str, float] = {}
    for metric, rule in config.get("metrics", {}).items():
        if metric not in baseline or metric not in candidate:
            missing.append(metric)
            continue
        delta = float(candidate[metric]) - float(baseline[metric])
        deltas[metric] = delta
        direction = rule.get("direction", "higher")
        if "min_delta" in rule:
            ok = delta >= float(rule["min_delta"]) if direction == "higher" else delta <= -float(rule["min_delta"])
        elif "max_drop" in rule:
            ok = delta >= -float(rule["max_drop"]) if direction == "higher" else delta <= float(rule["max_drop"])
        elif "max_value" in rule:
            ok = float(candidate[metric]) <= float(rule["max_value"])
        elif "min_value" in rule:
            ok = float(candidate[metric]) >= float(rule["min_value"])
        else:
            missing.append(metric)
            continue
        (passed if ok else failed).append(metric)
    if missing:
        return PromotionDecision(decision="needs_human_review", passed_checks=passed,
                                 failed_checks=failed + [f"missing:{name}" for name in missing],
                                 metric_deltas=deltas)
    return PromotionDecision(decision="promote" if not failed else "reject",
                             passed_checks=passed, failed_checks=failed, metric_deltas=deltas)
