"""Read-only rule draft: consecutive small-area myocardium review candidacy.

Rule (draft, read-only):
  - Candidate: >=2 consecutive slices with myo_area < 0.15 * cavity_area.
  - Suppress (route to stack-level review) when any of:
      * myocardium empty on any slice in the run
      * myocardium touches image border
      * temporal hole present in the run
      * centroid drift exceeds threshold

This module performs no writes and does not modify production labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

AREA_RATIO_THRESHOLD = 0.15
MIN_CONSECUTIVE = 2
DEFAULT_CENTROID_DRIFT_THRESHOLD = 5.0


@dataclass
class SliceObservation:
    slice_index: int
    myo_area: float
    cavity_area: float
    myo_empty: bool = False
    touches_border: bool = False
    temporal_hole: bool = False
    centroid: Optional[tuple] = None


@dataclass
class RuleDecision:
    candidate: bool
    suppressed: bool
    route: str
    reasons: List[str] = field(default_factory=list)
    consecutive_run: int = 0


def _centroid_drift(observations: Sequence[SliceObservation]) -> float:
    pts = [o.centroid for o in observations if o.centroid is not None]
    if len(pts) < 2:
        return 0.0
    max_d = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        d = (dx * dx + dy * dy) ** 0.5
        if d > max_d:
            max_d = d
    return max_d


def _is_small(o: SliceObservation) -> bool:
    if o.myo_empty or o.cavity_area <= 0:
        return False
    return o.myo_area < AREA_RATIO_THRESHOLD * o.cavity_area


def _longest_small_run(observations: Sequence[SliceObservation]) -> int:
    best = 0
    cur = 0
    for o in observations:
        if _is_small(o):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def evaluate(
    observations: Iterable[SliceObservation],
    centroid_drift_threshold: float = DEFAULT_CENTROID_DRIFT_THRESHOLD,
) -> RuleDecision:
    obs = sorted(observations, key=lambda o: o.slice_index)
    reasons: List[str] = []

    run = _longest_small_run(obs)
    candidate = run >= MIN_CONSECUTIVE
    if candidate:
        reasons.append(f"consecutive_small_area_run={run}")

    suppressed = False
    if any(o.myo_empty for o in obs):
        suppressed = True
        reasons.append("myo_empty")
    if any(o.touches_border for o in obs):
        suppressed = True
        reasons.append("myo_touches_border")
    if any(o.temporal_hole for o in obs):
        suppressed = True
        reasons.append("temporal_hole")
    drift = _centroid_drift(obs)
    if drift > centroid_drift_threshold:
        suppressed = True
        reasons.append(f"centroid_drift={drift:.3f}>{centroid_drift_threshold}")

    if suppressed:
        route = "stack_level_review"
    elif candidate:
        route = "small_area_review_candidate"
    else:
        route = "no_action"

    return RuleDecision(
        candidate=candidate,
        suppressed=suppressed,
        route=route,
        reasons=reasons,
        consecutive_run=run,
    )


def suppression_coverage(decisions: Sequence[RuleDecision], expected_suppressed: Sequence[bool]) -> float:
    if not expected_suppressed:
        return 1.0
    hits = sum(1 for d, e in zip(decisions, expected_suppressed) if d.suppressed == e)
    return hits / len(expected_suppressed)
