"""In-memory, report-only experiments for myocardial topology QC.

This module deliberately has no filesystem or image-I/O surface.  It is intended
for comparing topology hypotheses on an already cached, de-identified subset;
callers own loading the arrays and can persist the returned report if desired.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np


Candidate = str
CANDIDATES: tuple[Candidate, ...] = ("ring_close_holes", "largest_merge", "neighbor_propagation", "noop")


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[set[tuple[int, int]]] = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component: set[tuple[int, int]] = set()
        while queue:
            point = queue.popleft()
            component.add(point)
            r, c = point
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(component)
    return result


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1)
        result = np.zeros_like(result)
        for dr in range(3):
            for dc in range(3):
                result |= padded[dr:dr + result.shape[0], dc:dc + result.shape[1]]
    return result


def _close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _erode(_dilate(mask, radius), radius)


def _erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1)
        result = np.ones_like(result)
        for dr in range(3):
            for dc in range(3):
                result &= padded[dr:dr + result.shape[0], dc:dc + result.shape[1]]
    return result


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    outside = np.zeros(mask.shape, dtype=bool)
    queue = deque()
    for r, c in [(0, c) for c in range(mask.shape[1])] + [(mask.shape[0] - 1, c) for c in range(mask.shape[1])]:
        queue.append((r, c))
    for r, c in [(r, 0) for r in range(mask.shape[0])] + [(r, mask.shape[1] - 1) for r in range(mask.shape[0])]:
        queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]) or outside[r, c] or mask[r, c]:
            continue
        outside[r, c] = True
        queue.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
    return mask | (~mask & ~outside)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    points = np.argwhere(mask)
    return points.mean(axis=0) if len(points) else None


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    total = int(left.sum() + right.sum())
    return 1.0 if total == 0 else 2.0 * float(np.logical_and(left, right).sum()) / total


def _max_internal_gap(mask: np.ndarray) -> int:
    """Approximate the widest break along a row/column ring traversal."""
    widest = 0
    for line in list(mask) + list(mask.T):
        points = np.flatnonzero(line)
        if len(points) > 1:
            gaps = np.diff(points) - 1
            widest = max(widest, int(gaps.max(initial=0)))
    return widest


def _topology(mask: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask, bool)
    components = _components(mask)
    filled = _fill_holes(mask)
    holes = int(np.logical_and(filled, ~mask).sum())
    border_touch = int(bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
    return {
        "component_count": len(components),
        "hole_count": int(len(_components(filled & ~mask))),
        "hole_area": holes,
        "euler_characteristic": len(components) - holes,
        "ring_break_count": int(sum(
            1 for line in list(mask) + list(mask.T)
            if len(np.flatnonzero(line)) > 1 and np.any(np.diff(np.flatnonzero(line)) > 1)
        )),
        "ring_gap_width": _max_internal_gap(mask),
        "border_touch": border_touch,
    }


def _repair(mask: np.ndarray, candidate: Candidate, band: np.ndarray,
            previous: np.ndarray | None, next_mask: np.ndarray | None,
            continuity_limit: float) -> np.ndarray:
    mask = np.asarray(mask, bool)
    band = np.asarray(band, bool)
    if candidate == "noop":
        return mask.copy()
    if candidate == "ring_close_holes":
        # Both operations are clipped to the supplied myocardial band.
        return (_fill_holes(_close(mask, 1)) & band) | (mask & ~band)
    if candidate == "largest_merge":
        components = _components(mask)
        if not components:
            return mask.copy()
        largest = max(components, key=len)
        merged = np.zeros_like(mask)
        merged[tuple(zip(*largest))] = True
        for component in components:
            if component is largest:
                continue
            distance = min(abs(r - rr) + abs(c - cc) for r, c in largest for rr, cc in component)
            if distance <= 2:
                merged[tuple(zip(*component))] = True
        return (merged & band) | (mask & ~band)
    if candidate == "neighbor_propagation":
        result = mask.copy()
        for neighbour in (previous, next_mask):
            if neighbour is None:
                continue
            current_centre, neighbour_centre = _centroid(mask), _centroid(neighbour)
            if current_centre is None or neighbour_centre is None:
                continue
            if float(np.linalg.norm(current_centre - neighbour_centre)) <= continuity_limit:
                result |= np.asarray(neighbour, bool) & band
        return result
    raise ValueError(f"unknown topology candidate: {candidate}")


def evaluate_topology_shadow(
    myo_slices: Iterable[np.ndarray],
    mi_slices: Iterable[np.ndarray],
    *,
    band_slices: Iterable[np.ndarray] | None = None,
    myo_references: Iterable[np.ndarray] | None = None,
    mi_references: Iterable[np.ndarray] | None = None,
    slice_positions: Iterable[float] | None = None,
    through_plane_motion: Iterable[float] | None = None,
    continuity_limit: float = 6.0,
) -> dict[str, Any]:
    """Return candidate metrics without mutating inputs or writing any masks."""
    myo = [np.asarray(item, bool).copy() for item in myo_slices]
    mi = [np.asarray(item, bool).copy() for item in mi_slices]
    if len(myo) != len(mi) or not myo:
        raise ValueError("myo_slices and mi_slices must be non-empty and aligned")
    bands = [np.asarray(item, bool) for item in (band_slices or myo)]
    if len(bands) != len(myo) or any(a.shape != b.shape for a, b in zip(myo, bands)):
        raise ValueError("band_slices must align with myo_slices")
    myo_ref = list(myo_references) if myo_references is not None else None
    mi_ref = list(mi_references) if mi_references is not None else None
    positions = list(slice_positions) if slice_positions is not None else list(range(len(myo)))
    motion = list(through_plane_motion) if through_plane_motion is not None else [0.0] * len(myo)
    if len(positions) != len(myo) or len(motion) != len(myo):
        raise ValueError("slice_positions and through_plane_motion must align with slices")

    baseline_topology = [_topology(item) for item in myo]
    baseline_mi_components = [len(_components(item)) for item in mi]
    report: dict[str, Any] = {"shadow_only": True, "candidates": {}, "contraindications": {
        "band_restricted": True, "explicit_continuity_check": True,
        "mi_fragmentation_does_not_suppress_myo_flags": True, "mask_writes": False,
    }}
    for candidate in CANDIDATES:
        repaired = [_repair(item, candidate, bands[index], myo[index - 1] if index else None,
                            myo[index + 1] if index + 1 < len(myo) else None, continuity_limit)
                    for index, item in enumerate(myo)]
        after_topology = [_topology(item) for item in repaired]
        displacements = []
        for before, after, expected_motion in zip(myo, repaired, motion):
            before_centre, after_centre = _centroid(before), _centroid(after)
            displacements.append(None if before_centre is None or after_centre is None
                                else max(0.0, float(np.linalg.norm(after_centre - before_centre)) - float(expected_motion)))
        def inter_slice_metrics(stack: list[np.ndarray]) -> tuple[list[float | None], list[bool]]:
            distances: list[float | None] = []
            violations: list[bool] = []
            for index in range(len(stack) - 1):
                left, right = _centroid(stack[index]), _centroid(stack[index + 1])
                if left is None or right is None:
                    distances.append(None)
                    violations.append(False)
                    continue
                residual = max(0.0, float(np.linalg.norm(right - left)) - abs(float(motion[index + 1] - motion[index])))
                distances.append(residual)
                violations.append(residual > continuity_limit)
            return distances, violations
        before_inter_slice, before_violations = inter_slice_metrics(myo)
        after_inter_slice, after_violations = inter_slice_metrics(repaired)
        myo_dice_before = [_dice(a, b) for a, b in zip(myo, myo_ref)] if myo_ref is not None else None
        myo_dice_after = [_dice(a, b) for a, b in zip(repaired, myo_ref)] if myo_ref is not None else None
        mi_dice = [_dice(a, b) for a, b in zip(mi, mi_ref)] if mi_ref is not None else None
        report["candidates"][candidate] = {
            "myo_topology_before": baseline_topology,
            "myo_topology_after": after_topology,
            "myo_topology_flag_rate_before": sum(t["component_count"] > 1 or t["hole_count"] > 0 for t in baseline_topology) / len(myo),
            "myo_topology_flag_rate_after": sum(t["component_count"] > 1 or t["hole_count"] > 0 for t in after_topology) / len(myo),
            "mi_component_count_before": baseline_mi_components,
            "mi_component_count_after": baseline_mi_components,
            "mi_multiple_components_flag_rate_before": sum(c > 1 for c in baseline_mi_components) / len(mi),
            "mi_multiple_components_flag_rate_after": sum(c > 1 for c in baseline_mi_components) / len(mi),
            "mi_fragmentation_outcome": "persist-independently",
            "centroid_displacement": displacements,
            "inter_slice_centroid_displacement_before": before_inter_slice,
            "inter_slice_centroid_displacement_after": after_inter_slice,
            "continuity_violations_before": before_violations,
            "continuity_violations_after": after_violations,
            "slice_positions": positions,
            "through_plane_motion_accounted": motion,
            "myo_dice_before": myo_dice_before,
            "myo_dice_after": myo_dice_after,
            "mi_dice_before": mi_dice,
            "mi_dice_after": mi_dice,
        }
    return report
