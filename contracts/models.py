from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceCase(BaseModel):
    case_id: str
    modality: str
    issue_type: str


class FailureAnalysis(BaseModel):
    failure_type: str
    affected_modalities: List[str]
    evidence_cases: List[str]
    hypothesis: str
    suggested_experiment: str
    expected_metrics: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool = True


class RepairProposal(BaseModel):
    case_id: str
    modality: Literal["cmr", "lge"]
    target: Literal["myo", "mi", "pairing", "workflow"]
    issue_type: str
    proposed_action: str
    evidence: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_review: bool = True


class SpecialistAssessment(BaseModel):
    role: Literal["cmr_specialist", "lge_myo_specialist", "lge_mi_specialist"]
    applicable: bool
    finding: str
    supporting_issue_types: List[str]
    suggested_experiment: str
    contraindications: List[str]
    expected_metrics: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewDecision(BaseModel):
    decision: Literal["accept_experiment", "reject", "needs_human_review"]
    reasons: List[str]
    required_validation_metrics: List[str]
    risk_flags: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class MaintenanceProposal(BaseModel):
    role: Literal["qc_code_optimizer", "rule_curator", "skill_maintainer", "efficiency_auditor", "workflow_maintainer"]
    action_class: Literal[
        "no_change", "known_tool", "isolated_code_experiment",
        "rule_draft", "skill_draft", "human_review", "framework_repair",
    ]
    problem: str
    proposed_changes: List[str]
    target_components: List[str]
    evidence_issue_types: List[str]
    validation_plan: List[str]
    expected_benefits: List[str]
    risks: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_effort: Literal["small", "medium", "large"]


class SupervisorAction(BaseModel):
    action_id: str
    owner_role: str
    action_class: Literal[
        "known_tool", "isolated_code_experiment", "rule_draft",
        "skill_draft", "human_review", "stop", "framework_repair",
    ]
    objective: str
    acceptance_metrics: List[str]
    priority: int = Field(ge=1, le=5)
    max_attempts: int = Field(ge=1, le=3)
    auto_execute: bool = False
    # A near-duplicate may be a continuation only when it names the prior
    # action and one exact unmet metric from that action's reviewer feedback.
    continuation_of: Optional[str] = None
    addresses_metric: Optional[str] = None
    continuation_delta: Optional[str] = None


class SupervisorDecision(BaseModel):
    decision: Literal["execute", "experiment", "needs_human_review", "stop_no_gain"]
    rationale: List[str]
    actions: List[SupervisorAction]
    rejected_proposals: List[str]
    global_risk_flags: List[str]
    max_parallel_actions: int = Field(ge=1, le=4)
    confidence: float = Field(ge=0.0, le=1.0)


class AgentScorecard(BaseModel):
    role: str
    output_valid: bool
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    actionability: float = Field(ge=0.0, le=1.0)
    risk_awareness: float = Field(ge=0.0, le=1.0)
    confidence_calibration: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    notes: List[str]


class PromotionDecision(BaseModel):
    decision: Literal["promote", "reject", "needs_human_review"]
    passed_checks: List[str]
    failed_checks: List[str]
    metric_deltas: Dict[str, float]


class ManualEditAnalysis(BaseModel):
    operation_summary: str
    myocardial_coordinate_summary: str
    likely_edit_intent: str
    reusable_casebook_pattern: str
    supporting_features: List[str]
    prompt_improvements: List[str]
    qc_rule_candidate: str
    contraindications: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_additional_human_review: bool = True


class PromptMaskOperation(BaseModel):
    operation: Literal[
        "keep_largest_component",
        "remove_components_below_area",
        "fill_internal_holes_below_area",
        "binary_close",
        "binary_dilate",
        "binary_erode",
        "clip_mi_to_myo",
        "remove_components_in_region",
        "keep_components_in_region",
        "erase_region_pixels",
        "erase_box",
        "fill_box",
        "erase_circle",
        "fill_circle",
        "erase_polygon",
        "fill_polygon",
        "fill_from_adjacent_consensus",
    ]
    max_area: Optional[int] = Field(default=None, ge=1, le=500)
    radius: Optional[int] = Field(default=None, ge=1, le=6)
    region: Optional[Literal["left_upper", "right_upper", "left_lower",
                             "right_lower", "center"]] = None
    # 归一化屏幕坐标 0..1000（x=列，y=行，原点左上）
    x0: Optional[int] = Field(default=None, ge=0, le=1000)
    y0: Optional[int] = Field(default=None, ge=0, le=1000)
    x1: Optional[int] = Field(default=None, ge=0, le=1000)
    y1: Optional[int] = Field(default=None, ge=0, le=1000)
    cx: Optional[int] = Field(default=None, ge=0, le=1000)
    cy: Optional[int] = Field(default=None, ge=0, le=1000)
    r: Optional[int] = Field(default=None, ge=1, le=1000)
    # Polygon vertices in normalized screen coordinates [[x,y], ...].
    # The local executor enforces 3..24 vertices and 0..1000 bounds.
    points: Optional[List[List[int]]] = None


class PromptRepairPlan(BaseModel):
    decision: Literal["apply_operations", "no_change", "manual_required"]
    target: Literal["myo", "mi"]
    operations: List[PromptMaskOperation] = Field(default_factory=list, max_length=3)
    rationale: str
    expected_effect: str
    risk_flags: List[str]
    evidence_used: List[str] = Field(default_factory=list, max_length=4)
    constraints_checked: List[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
