"""Read-only LGE/CMR feature extraction and shadow candidate simulation.

This module intentionally accepts already-cached numpy masks.  It never loads a
dataset and never writes labels; candidates are represented by pseudo-flags and
in-memory masks only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ShadowParameters:
    """Frozen before validation; pilot and validation IDs must be disjoint."""

    a_min: int = 20
    d_max: float = 1.0
    connectivity: int = 8


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("slice masks must be 2-D")
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)]
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(int(y), int(x))]
        pixels: list[tuple[int, int]] = []
        while q:
            cy, cx = q.pop()
            pixels.append((cy, cx))
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        result.append(np.asarray(pixels, dtype=np.int32))
    return result


def _holes(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    outside = ~np.asarray(mask, dtype=bool)
    padded = np.pad(outside, 1, constant_values=True)
    reached = np.zeros(padded.shape, dtype=bool)
    q = deque([(0, 0)])
    reached[0, 0] = True
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)] if connectivity == 4 else [
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while q:
        y, x = q.popleft()
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1] and padded[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True
                q.append((ny, nx))
    inner = outside & ~reached[1:-1, 1:-1]
    return _components(inner, connectivity)


def _centroid(pixels: np.ndarray) -> list[float] | None:
    return [float(v) for v in pixels.mean(axis=0)[::-1]] if len(pixels) else None


def extract_slice_features(case_id: str, slice_index: int, mi_mask: np.ndarray,
                           myo_mask: np.ndarray, *, frame_index: int | None = None,
                           connectivity: int = 8) -> dict[str, Any]:
    """Extract one slice/frame's shared metrics from binary cached masks."""
    mi = np.asarray(mi_mask, dtype=bool)
    myo = np.asarray(myo_mask, dtype=bool)
    if mi.shape != myo.shape or mi.ndim != 2:
        raise ValueError("MI and myocardium masks must be matching 2-D arrays")
    mi_components = _components(mi, connectivity)
    myo_components = _components(myo, connectivity)
    holes = _holes(myo)
    myo_pixels = np.argwhere(myo)
    mi_pixels = np.argwhere(mi)
    distances = []
    if len(mi_pixels) and len(myo_pixels):
        # Chebyshev distance matches the one-voxel neighbourhood used by the
        # shadow merge and is deterministic without a scipy dependency.
        distances = np.max(np.abs(mi_pixels[:, None, :] - myo_pixels[None, :, :]), axis=2).min(axis=1).tolist()
    radial_breaks = 0
    gap_width = 0
    if len(myo_pixels):
        cy, cx = myo_pixels.mean(axis=0)
        angles = (np.arctan2(myo_pixels[:, 0] - cy, myo_pixels[:, 1] - cx) + np.pi) / (2 * np.pi)
        occupied = np.bincount(np.floor(angles * 36).astype(int).clip(0, 35), minlength=36) > 0
        runs = np.diff(np.r_[occupied, occupied]).nonzero()[0]
        gaps = []
        for start, end in zip(runs[::2], runs[1::2]):
            if not occupied[start % 36]:
                gaps.append((end - start) % 36 or 36)
        radial_breaks = sum(g > 1 for g in gaps)
        gap_width = max(gaps, default=0)
    return {
        "case_id": case_id, "slice_index": int(slice_index), "frame_index": frame_index,
        "mi_component_count": len(mi_components),
        "mi_components": [{"area": int(len(c)), "centroid": _centroid(c)} for c in mi_components],
        "mi_min_distance_to_myo": float(min(distances)) if distances else None,
        "mi_overlap_with_myo": int(np.count_nonzero(mi & myo)),
        "mi_fraction": float(np.count_nonzero(mi) / np.count_nonzero(myo)) if np.count_nonzero(myo) else None,
        "myo_component_count": len(myo_components),
        "myo_area": int(np.count_nonzero(myo)),
        "myo_hole_count": len(holes), "myo_hole_area": int(sum(len(h) for h in holes)),
        "myo_euler_characteristic": len(myo_components) - len(holes),
        "myo_radial_ring_break_count": int(radial_breaks), "myo_gap_width": int(gap_width),
        "myo_border_touch": bool(np.any(myo[0]) or np.any(myo[-1]) or np.any(myo[:, 0]) or np.any(myo[:, -1])),
        "myo_centroid": _centroid(myo_pixels),
    }


def extract_case_features(case_id: str, mi_mask: np.ndarray, myo_mask: np.ndarray,
                          *, connectivity: int = 8) -> list[dict[str, Any]]:
    """Extract per-slice/frame rows from [slice,y,x] or [slice,frame,y,x] masks."""
    mi, myo = np.asarray(mi_mask, bool), np.asarray(myo_mask, bool)
    if mi.shape != myo.shape or mi.ndim not in (2, 3, 4):
        raise ValueError("case masks must match and be 2-D, 3-D or 4-D")
    rows = []
    if mi.ndim == 2:
        rows.append(extract_slice_features(case_id, 0, mi, myo, connectivity=connectivity))
    elif mi.ndim == 3:
        for z in range(mi.shape[0]):
            rows.append(extract_slice_features(case_id, z, mi[z], myo[z], connectivity=connectivity))
    else:
        for z in range(mi.shape[0]):
            for t in range(mi.shape[1]):
                rows.append(extract_slice_features(case_id, z, mi[z, t], myo[z, t], frame_index=t, connectivity=connectivity))
    by_frame: dict[int | None, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(row["frame_index"], []).append(row)
    for frame_rows in by_frame.values():
        frame_rows.sort(key=lambda row: row["slice_index"])
        for previous, current in zip(frame_rows, frame_rows[1:]):
            a, b = previous.get("myo_centroid"), current.get("myo_centroid")
            previous["slice_to_slice_area_jump"] = abs(previous["myo_area"] - current["myo_area"])
            previous["slice_to_slice_centroid_jump"] = float(np.linalg.norm(np.asarray(a) - b)) if a and b else None
            z0, z1 = previous["slice_index"], current["slice_index"]
            if frame_rows[0]["frame_index"] is None:
                m0, m1 = myo[z0], myo[z1]
            else:
                t = int(frame_rows[0]["frame_index"])
                m0, m1 = myo[z0, t], myo[z1, t]
            intersection = int(np.count_nonzero(m0 & m1))
            union = int(np.count_nonzero(m0 | m1))
            previous["adjacent_slice_dice"] = float(2 * intersection / (np.count_nonzero(m0) + np.count_nonzero(m1))) if np.count_nonzero(m0) + np.count_nonzero(m1) else 1.0
            previous["adjacent_slice_iou"] = float(intersection / union) if union else 1.0
    return rows


def _dilate(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result |= np.roll(np.roll(mask, dy, 0), dx, 1)
    result[[0, -1], :] = False; result[:, [0, -1]] = False
    return result


def simulate_candidates(mi_mask: np.ndarray, myo_mask: np.ndarray, *, parameters: ShadowParameters = ShadowParameters()) -> dict[str, Any]:
    """Return in-memory candidate masks and pseudo-flags; never edits inputs."""
    mi, myo = np.asarray(mi_mask, bool), np.asarray(myo_mask, bool)
    if mi.shape != myo.shape or mi.ndim != 2:
        raise ValueError("candidate masks must be matching 2-D arrays")
    comps = _components(mi, parameters.connectivity)
    largest = np.zeros_like(mi)
    if comps:
        largest[tuple(max(comps, key=len).T)] = True
    small = np.zeros_like(mi)
    for comp in comps:
        if len(comp) <= parameters.a_min:
            small[tuple(comp.T)] = True
    # Closing is restricted to myocardium and only accepted where a component
    # overlaps the myocardium or is within d_max (the frozen one-voxel rule).
    closed = _dilate(_dilate(mi) & myo) & myo
    mergeable = myo & (_dilate(mi) if parameters.d_max <= 1 else myo)
    merged = (closed | (mi & mergeable)) & myo
    return {"largest_component_only": {"mask": largest, "pseudo_flag": bool(len(comps) > 1)},
            "absolute_small_area_removal": {"mask": mi & ~small, "pseudo_flag": bool(np.any(small))},
            "restricted_morphological_merge": {"mask": merged, "pseudo_flag": bool(np.any(merged != mi))},
            "no_op": {"mask": mi.copy(), "pseudo_flag": False}}


def validate_split(pilot_ids: Iterable[str], validation_ids: Iterable[str], parameters: ShadowParameters) -> dict[str, Any]:
    pilot, validation = set(pilot_ids), set(validation_ids)
    if pilot & validation:
        raise ValueError("pilot and validation case IDs must be disjoint")
    return {"pilot_case_count": len(pilot), "validation_case_count": len(validation), "parameters_frozen": asdict(parameters)}


def run_shadow_simulation(cases: Iterable[dict[str, Any]], *, pilot_ids: Iterable[str] = (),
                          validation_ids: Iterable[str] = (), parameters: ShadowParameters = ShadowParameters(),
                          blind_salt: str = "") -> dict[str, Any]:
    """Run the bounded offline experiment over cached masks.

    Each case contains ``case_id``, ``mi_mask``, ``myo_mask`` and optional
    ``issue_types``.  The returned queue contains one row per case/stratum and
    blinded IDs; no input mask is modified and no production artifact is read.
    """
    cases = list(cases)
    split = validate_split(pilot_ids, validation_ids, parameters)
    feature_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    distributions: dict[str, list[dict[str, float]]] = {name: [] for name in (
        "original", "largest_component_only", "absolute_small_area_removal",
        "restricted_morphological_merge", "no_op")}
    for case in cases:
        case_id = str(case["case_id"])
        rows = extract_case_features(case_id, case["mi_mask"], case["myo_mask"], connectivity=parameters.connectivity)
        feature_rows.extend(rows)
        candidates = simulate_candidates(case["mi_mask"], case["myo_mask"], parameters=parameters)
        issue_types = list(case.get("issue_types", [])) or ["unclassified"]
        blinded = hashlib.sha256((blind_salt + case_id).encode("utf-8")).hexdigest()[:16]
        for issue_type in issue_types:
            queue.append({"blinded_case_id": blinded, "stratum": issue_type, "case_id": None,
                          "candidate_flags": {name: bool(value["pseudo_flag"]) for name, value in candidates.items()}})
        def distribution(mask: np.ndarray) -> dict[str, float]:
            candidate_features = extract_case_features(case_id, mask, case["myo_mask"], connectivity=parameters.connectivity)
            return {"mi_component_count": float(np.mean([r["mi_component_count"] for r in candidate_features])),
                    "mi_fraction": float(np.mean([r["mi_fraction"] or 0.0 for r in candidate_features])),
                    "mi_multiple_components_rate": float(np.mean([r["mi_component_count"] > 1 for r in candidate_features]))}
        original = distribution(np.asarray(case["mi_mask"], bool))
        distributions["original"].append(original)
        for name, value in candidates.items():
            distributions[name].append(distribution(value["mask"]))
        candidate_rows.append({"case_id": case_id, "original": original,
                               "pseudo_flags": {name: bool(value["pseudo_flag"]) for name, value in candidates.items()}})
    aggregate_distributions = {name: {metric: float(np.mean([row[metric] for row in values])) if values else 0.0
                                       for metric in ("mi_component_count", "mi_fraction", "mi_multiple_components_rate")}
                               for name, values in distributions.items()}
    return {"production_writes": False, "label_mutation": False, "parameters": split["parameters_frozen"],
            "features": feature_rows, "candidate_results": candidate_rows,
            "distributions": aggregate_distributions, "review_queue": queue,
            "pilot_validation_split": split}


def export_review_queue(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write only an experiment review queue (JSON or CSV), never masks/labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        fields = sorted({key for row in rows for key in row})
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    else:
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
