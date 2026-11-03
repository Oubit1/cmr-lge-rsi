"""Draft-only LGE advisory rules for cached shadow outputs.

This module is deliberately independent of production QC entry points.  It
turns A1/A2-style per-slice feature rows into advisory pseudo-flags and
diagnostic sub-signals; it never loads, writes, or repairs a mask.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class AdvisoryParameters:
    """Frozen pilot parameters; validation must use disjoint slice IDs."""

    a_min_mm2: float = 20.0
    d_max_voxels: float = 1.0
    connectivity: int = 8

    def __post_init__(self) -> None:
        if self.a_min_mm2 <= 0 or self.d_max_voxels < 0:
            raise ValueError("advisory thresholds must be positive/non-negative")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


RULE_ID = "R-LGE-MI-FRAG-01"
TOPOLOGY_ID = "R-LGE-MYO-TOPO-02"
CONTRAINDICATIONS = (
    "paired CMR/LGE adjudication is required; a pseudo-flag is not a false-scar conclusion",
    "mi_fraction is not scar-burden ground truth",
    "no closing or hole filling outside the myocardial band",
    "no cross-slice merge without an explicit continuity check",
)
NON_RULES = (
    "mi_fraction_high_alone_is_insufficient",
    "mi_component_count_at_least_two_alone_is_insufficient",
)


def area_floor_pixels(voxel_area_mm2: float, parameters: AdvisoryParameters = AdvisoryParameters()) -> int:
    """Convert the fixed physical floor to an absolute voxel count."""
    if voxel_area_mm2 <= 0:
        raise ValueError("voxel_area_mm2 must be positive")
    return max(1, math.ceil(parameters.a_min_mm2 / voxel_area_mm2))


def _component_centroid_distance(row: dict[str, Any], component: dict[str, Any]) -> float | None:
    """Use the A1 boundary-distance field when present, otherwise remain unknown."""
    value = component.get("centroid_boundary_distance", row.get("mi_centroid_boundary_distance"))
    return None if value is None else float(value)


def evaluate_mi_fragment_rule(
    row: dict[str, Any],
    *,
    voxel_area_mm2: float,
    parameters: AdvisoryParameters = AdvisoryParameters(),
) -> dict[str, Any]:
    """Evaluate the four-condition advisory conjunction on one cached slice."""
    components = list(row.get("mi_components") or [])
    count = int(row.get("mi_component_count", len(components)))
    floor = area_floor_pixels(voxel_area_mm2, parameters)
    ordered = sorted(components, key=lambda item: float(item.get("area", 0)), reverse=True)
    non_dominant = ordered[1:] if len(ordered) > 1 else []
    small_near_boundary = [
        component for component in non_dominant
        if float(component.get("area", 0)) <= floor
        and (_component_centroid_distance(row, component) is not None)
        and _component_centroid_distance(row, component) <= parameters.d_max_voxels
    ]
    conditions = {
        "component_count_at_least_two": count >= 2,
        "non_dominant_small_component": bool(small_near_boundary),
        "small_component_near_myo_boundary": bool(small_near_boundary),
        "myo_ring_break_present": int(row.get("myo_radial_ring_break_count", 0)) > 0,
    }
    return {
        "rule_id": RULE_ID,
        "pseudo_flag": all(conditions.values()),
        "conditions": conditions,
        "a_min_pixels": floor,
        "connectivity": parameters.connectivity,
        "contraindications": list(CONTRAINDICATIONS),
        "non_rules": list(NON_RULES),
        "status": "advisory_only",
    }


def decompose_myo_topology(
    row: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    following: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose collapsed topology signals and classify local versus through-plane loss."""
    area = float(row.get("myo_area", 0) or 0)
    neighbours = [item for item in (previous, following) if item is not None]
    neighbour_areas = [float(item.get("myo_area", 0) or 0) for item in neighbours]
    mean_area = sum(neighbour_areas) / len(neighbour_areas) if neighbour_areas else None
    area_jump_ratio = (area / mean_area if mean_area else None)
    centroid_displacement = row.get("slice_to_slice_centroid_jump")
    local_outlier = bool(
        (int(row.get("myo_radial_ring_break_count", 0)) > 0 or int(row.get("myo_component_count", 0)) != 1)
        and all(int(item.get("myo_radial_ring_break_count", 0)) == 0 and int(item.get("myo_component_count", 0)) == 1
                for item in neighbours)
    ) if neighbours else False
    continuity_loss = bool(
        len(neighbours) >= 1 and (
            (centroid_displacement is not None and float(centroid_displacement) > 20.0)
            or (area_jump_ratio is not None and (area_jump_ratio > 3.0 or area_jump_ratio < 1 / 3))
            or any(int(item.get("myo_radial_ring_break_count", 0)) > 0 for item in neighbours)
        )
    )
    return {
        "rule_id": TOPOLOGY_ID,
        "slice_index": row.get("slice_index"),
        "myo_topology": bool(int(row.get("myo_component_count", 0)) != 1
                              or int(row.get("myo_hole_count", 0)) != 1
                              or int(row.get("myo_radial_ring_break_count", 0)) > 0
                              or bool(row.get("myo_border_touch", False))),
        "sub_signals": {
            "component_count": int(row.get("myo_component_count", 0)),
            "hole_count": int(row.get("myo_hole_count", 0)),
            "hole_area": int(row.get("myo_hole_area", 0)),
            "euler_characteristic": row.get("myo_euler_characteristic"),
            "ring_break_count": int(row.get("myo_radial_ring_break_count", 0)),
            "ring_gap_width": int(row.get("myo_gap_width", 0)),
            "border_touch": bool(row.get("myo_border_touch", False)),
            "adjacent_slice_centroid_displacement": centroid_displacement,
            "area_jump_ratio": area_jump_ratio,
        },
        "classification": "single_slice_outlier" if local_outlier else (
            "case_level_continuity_loss" if continuity_loss else "not_separated"
        ),
        "status": "advisory_only",
    }


def validate_advisory_rows(
    rows: Iterable[dict[str, Any]],
    *,
    pilot_ids: Iterable[str],
    validation_ids: Iterable[str],
    voxel_area_mm2: float,
    parameters: AdvisoryParameters = AdvisoryParameters(),
) -> dict[str, Any]:
    """Summarize flags before/after simulated rows without promoting a rule."""
    pilot, validation = set(map(str, pilot_ids)), set(map(str, validation_ids))
    if pilot & validation:
        raise ValueError("pilot and validation slice IDs must be disjoint")
    results = []
    for row in rows:
        slice_id = str(row.get("slice_id", row.get("case_id", row.get("slice_index"))))
        baseline = evaluate_mi_fragment_rule(row, voxel_area_mm2=voxel_area_mm2, parameters=parameters)
        candidate_rates = {}
        for candidate, candidate_row in (row.get("simulated_candidates") or {}).items():
            candidate_rates[candidate] = {
                "before": bool(baseline["pseudo_flag"]),
                "after": bool(evaluate_mi_fragment_rule(
                    candidate_row, voxel_area_mm2=voxel_area_mm2, parameters=parameters
                )["pseudo_flag"]),
            }
        results.append({
            "slice_id": slice_id,
            "split": "pilot" if slice_id in pilot else "validation" if slice_id in validation else "unassigned",
            "fragment": baseline,
            "candidate_before_after": candidate_rates,
            "topology": decompose_myo_topology(row, previous=row.get("previous"), following=row.get("following")),
        })
    counts = {
        split: {
            "fragment_pseudo_flag_rate": (sum(item["fragment"]["pseudo_flag"] for item in results if item["split"] == split)
                                           / max(1, sum(item["split"] == split for item in results))),
            "topology_pseudo_flag_rate": (sum(item["topology"]["myo_topology"] for item in results if item["split"] == split)
                                           / max(1, sum(item["split"] == split for item in results))),
        }
        for split in ("pilot", "validation", "unassigned")
    }
    return {
        "status": "draft",
        "promotion_blocked": True,
        "production_writes": False,
        "label_mutation": False,
        "parameters": asdict(parameters),
        "counts": counts,
        "rows": results,
        "contraindications": list(CONTRAINDICATIONS),
        "non_rules": list(NON_RULES),
    }
