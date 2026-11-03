'''CMR myocardial undersegmentation routing precedence and invariants.

This module encodes the approved QC skill guidance for CMR myocardial
undersegmentation. It is intentionally offline, read-only and side-effect free.
The read-only skill copy under skill/ is not modified here.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

CMR_SMALL_AREA_ISSUE = 'cmr_myo_small_area'
CMR_EXCESS_COMPONENTS_ISSUE = 'cmr_myo_excess_components'
CMR_EMPTY_ISSUE = 'cmr_myo_empty'
CMR_MISSING_TEMPORAL_HOLE_ISSUE = 'cmr_myo_missing_temporal_hole'
CMR_TOUCHES_IMAGE_BORDER_ISSUE = 'cmr_myo_touches_image_border'
CMR_CENTROID_DRIFT_ISSUE = 'cmr_centroid_drift'
CMR_LAYER_AREA_JUMP_ISSUE = 'cmr_layer_area_jump'
CMR_STRATIFICATION_ISSUES = frozenset({
    CMR_EMPTY_ISSUE,
    CMR_MISSING_TEMPORAL_HOLE_ISSUE,
    CMR_TOUCHES_IMAGE_BORDER_ISSUE,
    CMR_CENTROID_DRIFT_ISSUE,
    CMR_LAYER_AREA_JUMP_ISSUE,
})
CMR_TRIGGER_ISSUES = frozenset({CMR_SMALL_AREA_ISSUE, CMR_EXCESS_COMPONENTS_ISSUE})

LGE_MULTIPLE_COMPONENTS_ISSUE = 'mi_multiple_components'
LGE_FRACTION_HIGH_ISSUE = 'mi_fraction_high'
LGE_FRAGMENTATION_ISSUES = frozenset({LGE_MULTIPLE_COMPONENTS_ISSUE, LGE_FRACTION_HIGH_ISSUE})

ROUTE_NONE = 'none'
ROUTE_CMR_SMALL_AREA_AUDIT = 'cmr_small_area_provenance_audit'
ROUTE_LGE_FRAGMENTATION = 'lge_fragmentation_run3_run4'
LGE_FRAGMENTATION_RULE_SOURCE = 'run-3/4 promoted LGE fragmentation rules'
BOUNDED_READ_ONLY = True
DEFAULT_ESCALATION_BUDGET = 8

TOOL_GUIDANCE = (
    'Reuse one cached digest or one consolidated extractor; never build a second harness.'
)

ROUTING_PRECEDENCE_TEXT = (
    'CMR myocardial undersegmentation precedence: for modality including cmr, when the issue set '
    'includes cmr_myo_small_area and/or cmr_myo_excess_components, route first to the bounded '
    'read-only CMR small-area provenance check. run-9 issue_counts order: cmr_myo_small_area '
    '(11,105) and cmr_myo_excess_components (3,397) ahead of cmr_myo_empty (106) and '
    'cmr_centroid_drift (81). Treat cmr_myo_empty, cmr_myo_missing_temporal_hole, '
    'cmr_myo_touches_image_border, cmr_centroid_drift and cmr_layer_area_jump as stratification '
    'signals inside that route, not separate routes. Keep LGE mi_multiple_components and '
    'mi_fraction_high routed to run-3/4 promoted LGE fragmentation rules.'
)

RUN9_ISSUE_COUNTS = {
    CMR_SMALL_AREA_ISSUE: 11105,
    CMR_EXCESS_COMPONENTS_ISSUE: 3397,
    CMR_EMPTY_ISSUE: 106,
    CMR_CENTROID_DRIFT_ISSUE: 81,
}

INVARIANTS = (
    'cmr_myo_empty is a stratification signal, not an independent etiology.',
    'Paired-LGE presence or absence is report-only and must not gate firing, routing or adjudication inclusion.',
    'No single-component enforcement, ring-closing or hole-filling label edits may be derived.',
    'No cross-modality gating may alter CMR routing or adjudication inclusion.',
    'No label edit may be derived from this route.',
)

FORBIDDEN_LABEL_EDITS = frozenset({
    'single_component_enforcement',
    'ring_closing',
    'hole_filling',
})

NEVER_REEXECUTE = (
    'run-5 A1-A4',
    'run-6 proposals',
)

DUPLICATION_GUARDS = (
    'run-3/4 LGE fragmentation rules are referenced, not duplicated, by the CMR route.',
    'run-5 A1-A4 and run-6 proposals are not re-executed by this module.',
)

LABEL_EDIT_ALLOWED = False
CROSS_MODALITY_GATING_ALLOWED = False
PAIRED_LGE_REPORT_ONLY = True

IssueInput = Optional[Union[str, Iterable[str]]]
ModalityInput = Optional[Union[str, Iterable[str]]]


@dataclass(frozen=True)
class RoutingDecision:
    case_id: str
    route: str
    read_only: bool
    stratifiers: Tuple[str, ...] = ()
    trigger_issues: Tuple[str, ...] = ()
    lge_route: str = ROUTE_NONE
    label_edits: Tuple[str, ...] = ()
    cross_modality_gating: bool = False
    paired_lge_report_only: bool = True
    route_priority: int = 99


def _as_issue_set(issues: IssueInput) -> frozenset:
    if issues is None:
        return frozenset()
    if isinstance(issues, str):
        return frozenset({issues})
    return frozenset(str(item) for item in issues)


def _modality_tokens(modality: ModalityInput) -> Tuple[str, ...]:
    if modality is None:
        return ()
    if isinstance(modality, str):
        raw = modality.replace(',', ' ').split()
    else:
        raw = []
        for item in modality:
            raw.extend(str(item).replace(',', ' ').split())
    return tuple(token.lower() for token in raw if token)


def modality_includes_cmr(modality: ModalityInput) -> bool:
    return 'cmr' in _modality_tokens(modality)


def modality_includes_lge(modality: ModalityInput) -> bool:
    return 'lge' in _modality_tokens(modality)


def run9_precedence_order() -> Tuple[str, ...]:
    return tuple(sorted(RUN9_ISSUE_COUNTS, key=lambda key: RUN9_ISSUE_COUNTS[key], reverse=True))


def route(case_id: str, modality: ModalityInput, issues: IssueInput) -> RoutingDecision:
    issue_set = _as_issue_set(issues)
    cmr_modality = modality_includes_cmr(modality)
    lge_modality = modality_includes_lge(modality)
    trigger_issues = tuple(sorted(issue_set & CMR_TRIGGER_ISSUES))
    stratifiers = tuple(sorted(issue_set & CMR_STRATIFICATION_ISSUES))
    lge_route = ROUTE_NONE
    if issue_set & LGE_FRAGMENTATION_ISSUES:
        lge_route = ROUTE_LGE_FRAGMENTATION

    if cmr_modality and trigger_issues:
        return RoutingDecision(
            case_id=case_id,
            route=ROUTE_CMR_SMALL_AREA_AUDIT,
            read_only=BOUNDED_READ_ONLY,
            stratifiers=stratifiers,
            trigger_issues=trigger_issues,
            lge_route=lge_route,
            label_edits=(),
            cross_modality_gating=False,
            paired_lge_report_only=PAIRED_LGE_REPORT_ONLY,
            route_priority=1,
        )

    if lge_modality and (issue_set & LGE_FRAGMENTATION_ISSUES):
        return RoutingDecision(
            case_id=case_id,
            route=ROUTE_LGE_FRAGMENTATION,
            read_only=False,
            stratifiers=(),
            trigger_issues=(),
            lge_route=ROUTE_LGE_FRAGMENTATION,
            label_edits=(),
            cross_modality_gating=False,
            paired_lge_report_only=PAIRED_LGE_REPORT_ONLY,
            route_priority=2,
        )

    if cmr_modality and stratifiers:
        return RoutingDecision(
            case_id=case_id,
            route=ROUTE_CMR_SMALL_AREA_AUDIT,
            read_only=BOUNDED_READ_ONLY,
            stratifiers=stratifiers,
            trigger_issues=(),
            lge_route=lge_route,
            label_edits=(),
            cross_modality_gating=False,
            paired_lge_report_only=PAIRED_LGE_REPORT_ONLY,
            route_priority=3,
        )

    return RoutingDecision(
        case_id=case_id,
        route=ROUTE_NONE,
        read_only=False,
        stratifiers=stratifiers,
        trigger_issues=trigger_issues,
        lge_route=lge_route,
        label_edits=(),
        cross_modality_gating=False,
        paired_lge_report_only=PAIRED_LGE_REPORT_ONLY,
        route_priority=99,
    )


def is_label_edit_allowed(operation: str) -> bool:
    return not (operation in FORBIDDEN_LABEL_EDITS)


def escalation_text(budget: int = DEFAULT_ESCALATION_BUDGET) -> str:
    if budget < 0:
        raise ValueError('budget must be non-negative')
    return (
        'Escalation: adjudicate only discordant component-count or topology slices and every '
        'merge or drop slice, capped at {budget} slices by the pre-declared budget. '
        'Shadow-only with pre-declared connectivity, closing radius, merge distance and absolute '
        'size floors. No new threshold tuning.'
    ).format(budget=budget)


def is_escalation_slice(candidate: Mapping[str, object]) -> bool:
    kind = str(candidate.get('kind', '')).lower()
    if kind in {'merge', 'drop'}:
        return True
    if kind in {'discordant_component_count', 'discordant_topology'}:
        return True
    if candidate.get('discordant_component_count') is True:
        return True
    if candidate.get('discordant_topology') is True:
        return True
    if candidate.get('merge') is True or candidate.get('drop') is True:
        return True
    return False


def select_escalation_slices(
    candidates: Sequence[Mapping[str, object]],
    budget: int = DEFAULT_ESCALATION_BUDGET,
) -> Tuple[Mapping[str, object], ...]:
    if budget < 0:
        raise ValueError('budget must be non-negative')
    selected = []
    for candidate in candidates:
        if is_escalation_slice(candidate):
            selected.append(candidate)
            if len(selected) >= budget:
                break
    return tuple(selected)
