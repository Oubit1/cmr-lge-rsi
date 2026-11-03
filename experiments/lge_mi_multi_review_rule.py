"""Draft review-only QC rule R-LGE-MI-MULTI-REVIEW-01.

Disabled by default. No activation, no thresholds, no label edits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

RULE_ID = "R-LGE-MI-MULTI-REVIEW-01"
ESCALATION_RULE_ID = "R-LGE-MI-MULTI-REVIEW-01-E"
RULE_ENABLED = False
REVIEW_ONLY = True
PROMOTION_BLOCKED = True
PROMOTION_REQUIRES = ("A1", "A2", "human_review_acceptance")

MODALITY_LGE = "lge"
SUPPORT_FLAGS = ("mi_fraction_high", "myo_topology", "mi_tiny_unsupported")

ISSUE_MULTI_COMPONENT = "lge_mi_multiple_components_review"
ISSUE_FRACTION_HIGH = "lge_mi_fraction_high_review"
ISSUE_MYO_TOPOLOGY = "lge_myo_topology_review"
ISSUE_TINY_UNSUPPORTED = "lge_mi_tiny_unsupported_review"
ISSUE_ESCALATION = "lge_mi_multiple_components_casebook_escalation"

CONTRAINDICATIONS = (
    "isolated mi_multiple_components without a support flag is not failure",
    "CMR ring-contiguity is stratification only and never a gate",
    "no production label edits",
    "no rule activation or threshold introduction",
    "no second extractor when a valid cached digest exists",
    "no patient-identity inference or absolute paths",
)

RULE_TEXT = """R-LGE-MI-MULTI-REVIEW-01 (review-only, disabled by default)

Status: disabled_review_only. RULE_ENABLED=False. Promotion is blocked pending A1/A2 shadow metrics and human review acceptance as review-only.

Slice/frame positive predicate:
  modality == "lge"
  AND mi_multiple_components == true
  AND at least one same-slice support flag true among {mi_fraction_high, myo_topology, mi_tiny_unsupported}

Casebook escalation R-LGE-MI-MULTI-REVIEW-01-E:
  at least one slice-level positive
  AND either (mi_fraction_high == true AND myo_topology == true on the same slice)
  OR (mi_tiny_unsupported == true AND mi_fraction_high == true on the same slice)

Output:
  one review-only record per (case_id, modality)
  with merged issue_type set and addressable slice indices.

Contraindications:
  - isolated mi_multiple_components without a support flag is not failure
  - CMR ring-contiguity is stratification only and never a gate
  - no production label edits
  - no rule activation or threshold introduction
  - no second extractor when a valid cached digest exists
  - no patient-identity inference or absolute paths

This draft contains no activation path. It is not wired into the orchestrator.
"""


def _is_true(value: Any) -> bool:
    return value is True


def _modality(slice_payload: Mapping[str, Any]) -> str:
    value = slice_payload.get("modality")
    return str(value).lower() if value is not None else ""


def is_lge_slice(slice_payload: Mapping[str, Any]) -> bool:
    return _modality(slice_payload) == MODALITY_LGE


def has_support_flag(slice_payload: Mapping[str, Any]) -> bool:
    return any(_is_true(slice_payload.get(flag)) for flag in SUPPORT_FLAGS)


def is_slice_positive(slice_payload: Mapping[str, Any]) -> bool:
    return (
        is_lge_slice(slice_payload)
        and _is_true(slice_payload.get("mi_multiple_components"))
        and has_support_flag(slice_payload)
    )


def is_escalation_pair_slice(slice_payload: Mapping[str, Any]) -> bool:
    if not is_lge_slice(slice_payload):
        return False
    fraction_high = _is_true(slice_payload.get("mi_fraction_high"))
    myo_topology = _is_true(slice_payload.get("myo_topology"))
    tiny_unsupported = _is_true(slice_payload.get("mi_tiny_unsupported"))
    return (fraction_high and myo_topology) or (tiny_unsupported and fraction_high)


def _slice_index(slice_payload: Mapping[str, Any]) -> Optional[int]:
    value = slice_payload.get("slice_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def evaluate_case(
    case_id: str,
    modality: str,
    slices: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return zero or one review-only casebook record for (case_id, modality).

    The rule is disabled by default and this function does not contain an
    activation path. Records carry the current disabled/review-only status.
    """
    if str(modality).lower() != MODALITY_LGE:
        return []

    lge_slices = [s for s in slices if is_lge_slice(s)]
    positive_slices = [s for s in lge_slices if is_slice_positive(s)]
    if not positive_slices:
        return []

    escalation_pair_slices = [s for s in lge_slices if is_escalation_pair_slice(s)]
    escalated = bool(escalation_pair_slices)

    issue_types: Set[str] = set()
    positive_indices: Set[int] = set()
    escalation_indices: Set[int] = set()

    for slice_payload in positive_slices:
        issue_types.add(ISSUE_MULTI_COMPONENT)
        if _is_true(slice_payload.get("mi_fraction_high")):
            issue_types.add(ISSUE_FRACTION_HIGH)
        if _is_true(slice_payload.get("myo_topology")):
            issue_types.add(ISSUE_MYO_TOPOLOGY)
        if _is_true(slice_payload.get("mi_tiny_unsupported")):
            issue_types.add(ISSUE_TINY_UNSUPPORTED)
        idx = _slice_index(slice_payload)
        if idx is not None:
            positive_indices.add(idx)

    if escalated:
        issue_types.add(ISSUE_ESCALATION)
        for slice_payload in escalation_pair_slices:
            idx = _slice_index(slice_payload)
            if idx is not None:
                escalation_indices.add(idx)

    addressable_indices = set(positive_indices) | set(escalation_indices)

    return [
        {
            "case_id": case_id,
            "modality": MODALITY_LGE,
            "rule_id": RULE_ID,
            "escalation_rule_id": ESCALATION_RULE_ID if escalated else None,
            "review_only": REVIEW_ONLY,
            "rule_enabled": RULE_ENABLED,
            "promotion_blocked": PROMOTION_BLOCKED,
            "promotion_requires": list(PROMOTION_REQUIRES),
            "issue_types": sorted(issue_types),
            "slice_indices": sorted(positive_indices),
            "escalation_slice_indices": sorted(escalation_indices),
            "addressable_slice_indices": sorted(addressable_indices),
        }
    ]
