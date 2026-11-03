from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from agents.shadow_analysis import CONCISE_OUTPUT, REVIEW_PROMPT, expand_env, stable_hash
from agents.invocation import invoke_traced
from app.database import Database
from contracts.compat import model_dump
from contracts.models import ReviewDecision
from providers.registry import ProviderRegistry


PRODUCTION_REVIEW_POLICY = """For production_path changes, apply the repository's named
deterministic gate as the acceptance contract; do not invent stricter evidence after the gate ran.
For scripts/shadow_spatial_boundary.py, spatial_boundary_checked_controls is sufficient when its
suite passed and all four checks are true: zero_label_mutations, checked_false_routes_eq_0,
control_gate_pass and nonempty_pending_coverage. Here pending_spatial_boundary_candidates>0 means
any positive count, not equality to the pending denominator. Suppressing checked false routes is
the intended control behavior, not pending over-suppression. Independent-label concordance,
precision and CI are unavailable monitoring debt, not a promotion blocker under the current
operator policy. This call itself is the independent implementation review. Promotion creates the
recoverable backup after acceptance, so do not require a pre-promotion backup rehearsal. Still
reject failing tests, a failed named gate, unverifiable changed files, label writes, or claims that
contradict the deterministic report."""
PRODUCTION_REVIEW_POLICY += """\nFor a quality_scorer_all_targets RSI selection, review the actual paired aggregate
metrics on the registered frozen gold/control benchmark, its all-target nonregression constraints,
the existing image_label_quality gate and full regression. PROMOTE means measured scoring improvement
within that benchmark scope, not clinical acceptance of repaired labels. Do not demand new per-case
doctor signoff as a prerequisite for a code experiment when those reports pass; retain that as
clinical monitoring debt. Reject changed reference/evaluator contracts, degraded metrics or unproven claims."""
PRODUCTION_REVIEW_POLICY += """\nFor scripts/lge_candidate_generator.py, the lge_candidate_generator_contract is an
interface/snapshot-isolation gate, not clinical accuracy proof. A passing contract and full regression
allow interface compatibility, while producer changes also require the frozen doctor-reference challenge to show parent-relative nonregression and measured improvement; the trusted controller independently grades each actual
candidate, checks paired masks/holdout/source hashes and alone applies labels. The producer must never
write source labels, report invented validation, hardcode case IDs or bypass the controller. Do not
demand production label writes as a prerequisite for installing this output-only producer."""


PRODUCTION_REVIEW_POLICY += "\nFor framework_repair, only the protocol alias JSON and its named test are permitted. A passing lge_protocol_contract and full regression justify transport configuration installation, not medical improvement. Reject edits to algorithms, source labels, writers, evaluators, budgets or permissions."


def review_implementation(*, db: Database, run_id: int, action: dict[str, Any],
                          implementation: dict[str, Any], validation: dict[str, Any],
                          providers_path: Path, agents_path: Path) -> ReviewDecision:
    """Ask the independent reviewer to approve the implemented result, not only its plan."""
    agents = expand_env(yaml.safe_load(agents_path.read_text(encoding="utf-8")))["agents"]
    cfg = agents["independent_reviewer"]
    action_id = action["action_id"]
    owner = action["owner_role"]
    policy = PRODUCTION_REVIEW_POLICY
    if action.get("action_class") == "segmentation_training":
        policy = ("Review the exact typed segmentation-cycle-v1 plan and its deterministic preflight. "
                  "This authorizes only bounded training and evaluation through the trusted host; "
                  "it is neither a framework repair nor code/label promotion. Require unchanged "
                  "artifact hashes, existing independent policy, frozen cohort and target ownership. "
                  "Do not invent references, exposure provenance, metric results or filesystem approvals. "
                  "Independent evaluation evidence and exact assessment activation approval remain "
                  "separate later gates. Reject missing or unverifiable inputs; list any unresolved "
                  "pre-training requirements in required_validation_metrics.")
    thread = f"run-{run_id}:{action_id}"
    result_message = db.post_agent_message(
        run_id, thread_id=thread, sender_role=owner,
        recipient_role="independent_reviewer", message_type="implementation_result",
        payload={"implementation": implementation, "validation": validation},
    )
    review_input = {
        "role": "independent_reviewer",
        "_max_output_tokens": 1500,
        "_schema_retries": 0,
        "_fallback_on_error": False,
        "mode": "post_implementation_review_no_production_write",
        "action": action,
        "implementation": implementation,
        "deterministic_validation": validation,
        "production_review_policy": policy,
        "handoff_messages": db.agent_inbox(run_id, "independent_reviewer", thread_id=thread),
    }
    provider = ProviderRegistry(providers_path).get(cfg["provider"])
    review = asyncio.run(invoke_traced(
        db=db, run_id=run_id, thread_id=thread,
        role="independent_reviewer:implementation", provider_name=cfg["provider"],
        provider=provider,
        system_prompt=(REVIEW_PROMPT + "\nReview the completed isolated implementation and "
                       "its deterministic test evidence. Reject unsupported result claims. "
                       "For promotion_scope=shadow_only, accept_experiment may approve archiving "
                       "code confined to experiments/ and tests/ when tests pass, while listing "
                       "clinical/held-out metrics still owed; this does not activate a production "
                       "QC route or validate those metrics. For production_path, follow the "
                       "explicit named-gate policy below.\n" + policy + "\n"
                       + CONCISE_OUTPUT),
        input_data=review_input, output_schema=ReviewDecision, model=cfg["model"],
    ))
    payload = model_dump(review)
    payload["implementation_sha256"] = stable_hash(implementation)
    db.save_agent_output(
        run_id, "independent_reviewer:implementation", cfg["provider"], cfg["model"],
        "SUCCEEDED", stable_hash(review_input), payload,
    )
    db.post_agent_message(
        run_id, thread_id=thread, sender_role="independent_reviewer",
        recipient_role=owner, message_type="implementation_review",
        payload=payload, reply_to=result_message,
    )
    return review


def review_segmentation_stage(*, db: Database, plan: dict, request: dict, request_id: int,
                             providers_path: Path, agents_path: Path) -> ReviewDecision:
    """Produce a real DB review from the configured independent role, not a flag.

    Source-message IDs have already been independently verified by the host-side
    dispatcher. This call cannot invent exposure/reference facts or measurements.
    The read-only host revalidates all referenced messages before accepting them.
    """
    phase = request["phase"]
    if phase not in {"evaluation", "activation"}:
        raise ValueError("unknown_segmentation_review_phase")
    agents = expand_env(yaml.safe_load(agents_path.read_text(encoding="utf-8")))["agents"]
    cfg = agents["independent_reviewer"]
    thread = f"run-{plan['run_id']}:{plan['action_id']}"
    data = {"role": "independent_reviewer", "_max_output_tokens": 1500,
            "_schema_retries": 0, "_fallback_on_error": False,
            "mode": "segmentation_" + phase + "_review", "request": request}
    review = asyncio.run(invoke_traced(db=db, run_id=plan["run_id"], thread_id=thread,
        role="independent_reviewer:segmentation_" + phase, provider_name=cfg["provider"],
        provider=ProviderRegistry(providers_path).get(cfg["provider"]), model=cfg["model"],
        system_prompt=("Review this exact immutable segmentation plan/report/candidate binding. "
            "The metrics are independently computed; provenance comes only from supplied verified DB message IDs. "
            "Never infer unknown baseline exposure, create references, invent facts or relax the approved policy. "
            "For evaluation, accept only sufficient existing independent provenance and honest measurements. "
            "For activation, additionally require the exact passed deterministic assessment and parent-relative "
            "nonregression. Any unresolved requirement must remain in required_validation_metrics. "
            "This cannot authorize source label writes. Return the requested structured decision."),
        input_data=data, output_schema=ReviewDecision))
    accepted = review.decision == "accept_experiment" and not review.required_validation_metrics
    keys = ("plan_sha256", "report_sha256", "candidate_id", "parent_id", "policy_id", "target")
    payload = {key: request[key] for key in keys}
    if phase == "evaluation":
        payload.update(schema="segmentation-evaluation-evidence-v1",
                       status="VERIFIED" if accepted else "REJECTED" if review.decision == "reject" else "UNKNOWN")
        payload.update(request["source_message_ids"])
        kind = "segmentation_evaluation_evidence"
    else:
        payload.update(assessment_id=request["assessment_id"],
                       decision="accept_experiment" if accepted else "reject" if review.decision == "reject" else "needs_human_review")
        kind = "segmentation_activation_review"
    payload["review"] = model_dump(review)
    db.save_agent_output(plan["run_id"], "independent_reviewer:segmentation_" + phase,
                         cfg["provider"], cfg["model"], "SUCCEEDED", stable_hash(data), payload)
    db.post_agent_message(plan["run_id"], thread_id=thread, sender_role="independent_reviewer",
        recipient_role=plan["owner_role"], message_type=kind, payload=payload, reply_to=request_id)
    return review
