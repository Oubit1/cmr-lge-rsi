"""
LGE MI Fragmentation Topology Review

Skill: Shadow-only guidance for LGE scar fragmentation assessment.
Applicability: LGE-only (A3 rule semantics).

Routing:
- Route mi_multiple_components co-occurring with myo_topology and/or
  mi_fraction_high to a bounded LGE MI fragmentation check.
- Do NOT route directly to label QC escalation.

Invariant:
- Single-component enforcement is prohibited without per-slice review.

Tool Guidance (shadow-only, pre-declared parameters):
- closing_radius: 2.0 mm (morphological closing to bridge thin gaps)
- merge_distance: 5.0 mm (maximum distance to consider components as mergeable)
- minimum_size: 10 voxels (minimum component size to retain)
- No threshold sweep. No ad hoc component merging.

Escalation:
- Escalate to human review only when:
  1. Multiple components persist after the bounded variant, OR
  2. True multifocal scar cannot be excluded.

Paired CMR/LGE:
- When paired CMR/LGE data exist, perform per-slice topology and overlap check.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Pre-declared shadow parameters (guidance-only, no runtime tuning)
# ---------------------------------------------------------------------------
CLOSING_RADIUS_MM: float = 2.0
MERGE_DISTANCE_MM: float = 5.0
MINIMUM_COMPONENT_SIZE_VOXELS: int = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Component:
    """Minimal representation of a scar component for shadow review."""
    label: int
    centroid_xyz: tuple[float, float, float]
    volume_voxels: int
    slice_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SliceTopology:
    """Per-slice topology summary for a single slice."""
    slice_index: int
    num_components: int
    component_labels: tuple[int, ...]
    overlap_with_cmr: Optional[float] = None  # fraction 0..1, None if no paired CMR


@dataclasses.dataclass(frozen=True)
class FragmentationReview:
    """Output of the bounded LGE MI fragmentation check."""
    routed: bool
    invariant_held: bool
    tool_guidance_applied: bool
    components_after_bounded: Sequence[Component]
    slice_topologies: Sequence[SliceTopology]
    paired_cmr_checked: bool
    escalation_required: bool
    escalation_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Routing predicates (A3 rule semantics)
# ---------------------------------------------------------------------------

def _flag_predicate(flags: Sequence[str], predicate: str) -> bool:
    return predicate in flags


def should_route_to_fragmentation_check(flags: Sequence[str]) -> bool:
    """
    Route to bounded LGE MI fragmentation check if mi_multiple_components
    co-occurs with myo_topology and/or mi_fraction_high.
    """
    has_multiple = _flag_predicate(flags, "mi_multiple_components")
    has_topology = _flag_predicate(flags, "myo_topology")
    has_fraction = _flag_predicate(flags, "mi_fraction_high")
    return has_multiple and (has_topology or has_fraction)


# ---------------------------------------------------------------------------
# Invariant enforcement
# ---------------------------------------------------------------------------

def invariant_single_component_requires_per_slice_review(
    components: Sequence[Component],
    per_slice_reviewed: bool,
) -> bool:
    """
    Invariant: single-component enforcement is prohibited without
    per-slice review.
    """
    if len(components) == 1 and not per_slice_reviewed:
        return False
    return True


# ---------------------------------------------------------------------------
# Bounded variant (shadow tool guidance)
# ---------------------------------------------------------------------------

def apply_bounded_variant(
    components: Sequence[Component],
    closing_radius_mm: float = CLOSING_RADIUS_MM,
    merge_distance_mm: float = MERGE_DISTANCE_MM,
    minimum_size: int = MINIMUM_COMPONENT_SIZE_VOXELS,
) -> Sequence[Component]:
    """
    Shadow-only bounded variant.
    Returns components that persist after applying pre-declared parameters.
    No actual image processing is performed; this is guidance scaffolding.
    """
    # Filter by minimum size
    size_filtered = [c for c in components if c.volume_voxels >= minimum_size]
    # Merge distance and closing radius are declared for guidance;
    # actual merging/closing would be performed by the downstream tool.
    # We return the filtered set as the "persisting" components.
    return size_filtered


# ---------------------------------------------------------------------------
# Escalation criteria
# ---------------------------------------------------------------------------

def check_escalation_required(
    components_after_bounded: Sequence[Component],
    true_multifocal_cannot_be_excluded: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Escalate to human review only when:
    1. Multiple components persist after the bounded variant, OR
    2. True multifocal scar cannot be excluded.
    """
    if len(components_after_bounded) > 1:
        return True, "multiple_components_persist_after_bounded_variant"
    if true_multifocal_cannot_be_excluded:
        return True, "true_multifocal_scar_cannot_be_excluded"
    return False, None


# ---------------------------------------------------------------------------
# Paired CMR/LGE per-slice topology and overlap check
# ---------------------------------------------------------------------------

def check_paired_cmr_lge_topology(
    lge_slice_topologies: Sequence[SliceTopology],
    cmr_slice_topologies: Optional[Sequence[SliceTopology]] = None,
) -> Sequence[SliceTopology]:
    """
    When paired CMR/LGE data exist, perform per-slice topology and overlap check.
    Returns enriched slice topologies with overlap fractions.
    """
    if cmr_slice_topologies is None:
        # No paired data; return LGE topologies unchanged
        return list(lge_slice_topologies)

    # Build lookup by slice index
    cmr_by_slice = {st.slice_index: st for st in cmr_slice_topologies}

    enriched: list[SliceTopology] = []
    for lge_st in lge_slice_topologies:
        cmr_st = cmr_by_slice.get(lge_st.slice_index)
        if cmr_st is not None:
            # Compute overlap as simple Jaccard-like proxy using component labels
            lge_labels = set(lge_st.component_labels)
            cmr_labels = set(cmr_st.component_labels)
            union = lge_labels | cmr_labels
            intersection = lge_labels & cmr_labels
            overlap = len(intersection) / len(union) if union else 0.0
            enriched.append(
                SliceTopology(
                    slice_index=lge_st.slice_index,
                    num_components=lge_st.num_components,
                    component_labels=lge_st.component_labels,
                    overlap_with_cmr=overlap,
                )
            )
        else:
            enriched.append(lge_st)
    return enriched


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_fragmentation_review(
    flags: Sequence[str],
    components: Sequence[Component] = (),
    per_slice_reviewed: bool = False,
    cmr_slice_topologies: Optional[Sequence[SliceTopology]] = None,
    true_multifocal_cannot_be_excluded: bool = False,
) -> FragmentationReview:
    """
    Run the bounded LGE MI fragmentation check.
    """
    routed = should_route_to_fragmentation_check(flags)
    if not routed:
        return FragmentationReview(
            routed=False,
            invariant_held=True,
            tool_guidance_applied=False,
            components_after_bounded=[],
            slice_topologies=[],
            paired_cmr_checked=False,
            escalation_required=False,
        )

    # Invariant check
    invariant_held = invariant_single_component_requires_per_slice_review(
        components, per_slice_reviewed
    )

    # Apply bounded variant (shadow tool guidance)
    components_after = apply_bounded_variant(components)

    # Paired CMR/LGE check
    lge_slice_topologies = [
        SliceTopology(
            slice_index=min(c.slice_indices) if c.slice_indices else 0,
            num_components=1,
            component_labels=(c.label,),
        )
        for c in components
    ]
    enriched_topologies = check_paired_cmr_lge_topology(
        lge_slice_topologies, cmr_slice_topologies
    )

    # Escalation check
    escalation_required, escalation_reason = check_escalation_required(
        components_after, true_multifocal_cannot_be_excluded
    )

    return FragmentationReview(
        routed=True,
        invariant_held=invariant_held,
        tool_guidance_applied=True,
        components_after_bounded=components_after,
        slice_topologies=enriched_topologies,
        paired_cmr_checked=cmr_slice_topologies is not None,
        escalation_required=escalation_required,
        escalation_reason=escalation_reason,
    )
