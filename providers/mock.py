from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

from contracts.compat import model_validate
from providers.base import ModelProvider


T = TypeVar("T", bound=BaseModel)


class MockProvider(ModelProvider):
    name = "mock"

    async def generate_structured(self, *, system_prompt: str, input_data: Dict[str, Any],
                                  output_schema: Type[T], model: str,
                                  timeout_seconds: int = 180) -> T:
        if output_schema.__name__ == "FailureAnalysis":
            top = input_data.get("issue_counts", {})
            failure = max(top, key=top.get) if top else "no_issue"
            cases = [item["case_id"] for item in input_data.get("representative_cases", [])[:3]]
            return model_validate(output_schema, {
                "failure_type": failure,
                "affected_modalities": sorted(input_data.get("modalities", [])),
                "evidence_cases": cases,
                "hypothesis": "Mock analysis: deterministic QC category requires focused review.",
                "suggested_experiment": "Run a read-only matcher on the fixed regression subset.",
                "expected_metrics": ["issue_recall", "false_repair_rate"],
                "confidence": 0.5,
                "requires_human_review": True,
            })
        if output_schema.__name__ == "ReviewDecision":
            return model_validate(output_schema, {
                "decision": "needs_human_review",
                "reasons": ["Mock provider never authorizes mutation."],
                "required_validation_metrics": ["issue_recall", "false_repair_rate"],
                "risk_flags": ["mock_analysis"],
                "confidence": 1.0,
            })
        if output_schema.__name__ == "SpecialistAssessment":
            role = input_data.get("role", "cmr_specialist")
            issue_counts = input_data.get("evidence_summary", {}).get("issue_counts", {})
            return model_validate(output_schema, {
                "role": role,
                "applicable": True,
                "finding": "Mock specialist assessment; no production mutation authorized.",
                "supporting_issue_types": list(issue_counts)[:3],
                "suggested_experiment": "Evaluate a read-only matcher on the fixed regression set.",
                "contraindications": ["no image-level evidence in first-version shadow package"],
                "expected_metrics": ["issue_recall", "false_repair_rate"],
                "confidence": 0.5,
            })
        if output_schema.__name__ == "MaintenanceProposal":
            role = input_data.get("role", "qc_code_optimizer")
            return model_validate(output_schema, {
                "role": role,
                "action_class": "human_review",
                "problem": "Mock provider cannot authorize a maintenance experiment.",
                "proposed_changes": [],
                "target_components": [],
                "evidence_issue_types": list(input_data.get("evidence_summary", {}).get("issue_counts", {}))[:3],
                "validation_plan": ["Run deterministic regression tests."],
                "expected_benefits": [],
                "risks": ["mock_analysis"],
                "confidence": 0.5,
                "estimated_effort": "small",
            })
        if output_schema.__name__ == "SupervisorDecision":
            return model_validate(output_schema, {
                "decision": "needs_human_review",
                "rationale": ["Mock supervisor does not execute maintenance actions."],
                "actions": [{
                    "action_id": "mock-review",
                    "owner_role": "independent_reviewer",
                    "action_class": "human_review",
                    "objective": "Review unresolved QC evidence.",
                    "acceptance_metrics": ["false_repair_rate"],
                    "priority": 3,
                    "max_attempts": 1,
                    "auto_execute": False,
                }],
                "rejected_proposals": [],
                "global_risk_flags": ["mock_analysis"],
                "max_parallel_actions": 1,
                "confidence": 1.0,
            })
        if output_schema.__name__ == "ManualEditAnalysis":
            delta = input_data.get("manual_edit_delta", {})
            return model_validate(output_schema, {
                "operation_summary": f"Mock {delta.get('operation', 'unknown')} edit.",
                "myocardial_coordinate_summary": "Image-coordinate location only.",
                "likely_edit_intent": delta.get("suggested_pattern", "manual refinement"),
                "reusable_casebook_pattern": "Retain as a reviewed example; do not auto-promote.",
                "supporting_features": ["de-identified mask delta"],
                "prompt_improvements": ["Require topology and adjacent-layer evidence."],
                "qc_rule_candidate": "shadow_only_manual_delta_pattern",
                "contraindications": ["no intensity or adjacent-layer evidence supplied"],
                "confidence": 0.5,
                "requires_additional_human_review": True,
            })
        if output_schema.__name__ == "PromptRepairPlan":
            request = input_data.get("prompt_repair", {})
            return model_validate(output_schema, {
                "decision": "apply_operations",
                "target": request.get("target", "myo"),
                "operations": [{"operation": "fill_box", "x0": 450, "y0": 50,
                                "x1": 550, "y1": 180}],
                "rationale": "Mock plan for a staged visual candidate.",
                "expected_effect": "Fill a bounded internal gap without changing production labels.",
                "risk_flags": ["mock_analysis", "visual_review_required"],
                "evidence_used": ["operator_instruction", "current_mask_geometry"],
                "constraints_checked": ["small_local_delta"],
                "confidence": 0.5,
            })
        raise ValueError(f"mock provider has no fixture for {output_schema.__name__}")
