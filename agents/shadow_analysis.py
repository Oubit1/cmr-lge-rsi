from __future__ import annotations

import difflib
import hashlib
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from app.database import Database
from app.state_machine import RunState
from agents.evidence import (action_builds_stage_consumer,
                             action_uses_stale_lge_shadow_contract,
                             action_matches_workflow_stage,
                             build_evidence_package,
                             semantic_action_workflow_stage)
from agents.governance import enforce_action_ownership, enforce_autonomous_executor_contract
from agents.governance import ACTION_OWNERS
from agents.invocation import invoke_traced
from contracts.compat import model_dump
from contracts.models import (
    AgentScorecard,
    FailureAnalysis,
    MaintenanceProposal,
    ReviewDecision,
    SpecialistAssessment,
    SupervisorDecision,
)
from providers.registry import ProviderRegistry


TRIAGE_PROMPT = """You are a read-only cardiac MRI QC failure-analysis specialist.
Use only the anonymized aggregate evidence. Identify one testable failure class and
one bounded experiment. Respect iteration_history: do not repeat an attempted experiment unless
new evidence directly addresses its prior failure. Use progress_ledger to distinguish
unfinished validation from a genuinely new experiment. Prefer the largest unresolved issue family.
Analyze only workflow_dependency.stage. Later stages are blocked until the persisted prior-stage
report passes; parallelism is only across samples inside the active stage.
Do not prescribe direct label edits and do not infer patient identity.
Return only the requested structured object."""

REVIEW_PROMPT = """You are an independent safety reviewer for a medical-image QC workflow.
Review the supervisor plan and its source proposals, identify unsupported assumptions, and decide
whether bounded isolated experiments are justified. You cannot authorize direct production label
or code writes. You may approve a patch to a named production-path file only inside a disposable
snapshot when prior measured shadow evidence identifies the blocker; final promotion still requires
deterministic production gates, independent implementation review and rollback evidence.
Return only the requested structured object."""

PLAN_REVIEW_POLICY = """This call reviews permission to START an isolated experiment, not
permission to promote its results. auto_execute means the assigned owner can run the
bounded experiment in a disposable snapshot; it does not authorize production writes.
Missing outcome measurements or a failing production control gate are not by themselves
reasons to block a read-only experiment that collects those measurements. Require a
feasible plan, explicit inputs and outputs, and honest handling of unavailable truth:
without independent reference labels, concordance/precision/CI must be reported as
unavailable, never fabricated. Coverage counts and zero-write audit may still be collected.
Reject unsafe or infeasible plans and state the specific dependency. If the experiment is
feasible, accept_experiment with the metrics still owed; the later implementation review
and production gates decide promotion. A production-path snapshot patch must name its target,
cite the measured predecessor and define a zero-write replay gate; otherwise reject it.
An empty actions list cannot be approved."""

SPECIALIST_PROMPTS = {
    "cmr_specialist": """You are a read-only CMR myocardium QC specialist. Assess only CMR
topology, temporal continuity, spatial stack continuity and paired-LGE support. Propose a bounded
experiment, never a production label edit. Return only the requested structured object.""",
    "lge_myo_specialist": """You are a read-only LGE myocardium QC specialist. Assess LV
location, myocardial topology and cross-slice continuity. You may test already-validated CMR
myocardium layer/topology priors, but never transfer CMR temporal-frame adjacency constraints;
measure LGE-specific validity before promotion. Propose a bounded experiment, never a production
label edit. Return only the requested structured object.""",
    "lge_mi_specialist": """You are a read-only LGE infarct QC specialist. Assess MI containment,
contrast support, component continuity and termination across slices. Propose a bounded experiment,
never a production label edit. Return only the requested structured object.""",
}

MAINTENANCE_PROMPTS = {
    "workflow_maintainer": "Repair only configs/lge_protocol_aliases.json to map observed transport field aliases to the existing snapshot paths/candidate filename. Propose framework_repair owned by workflow_maintainer. Do not change medical decisions, masks, QC thresholds, evaluators, budgets, permissions, writers or other source. Use protocol_repair evidence and the existing generator source; preserve canonical fields. Unsupported faults must be reported rather than pretending the config can fix them.",
    "qc_code_optimizer": """You optimize deterministic CMR/LGE QC implementation. Propose a
bounded code experiment that targets the observed residual failure class, including tests and a
fixed-validation comparison. After a measured shadow result, you may propose a named production-path
patch, but it must be implemented only in the disposable owner snapshot and pass the production gate.
When evidence_summary.check_iteration.production_blocker is present, propose that exact named
production-path patch before any new measurement-only experiment. Choose the discriminator yourself
from real_evaluation_input.spatial_control; the control plane does not prescribe medical thresholds.
Use registered RSI benchmark weaknesses when supplied: name the component and exact metric to
improve, preserve other targets, and keep evaluator/reference contracts immutable. A bounded
improvement is valid without demanding perfect detection in every class. Clinical thresholds
remain your domain responsibility; controlled-defect scores are not clinical label acceptance.
Never modify production labels directly.""",
    "rule_curator": """You curate reusable CMR/LGE QC rules and the clinical casebook. Convert
repeated evidence into the narrowest measurable rule, list contraindications and validation. Do
not broaden thresholds from a single ambiguous case.""",
    "skill_maintainer": """You maintain the CMR+LGE workflow skill. Propose concise updates only
when observed failures reveal missing routing, invariants, tools or escalation criteria. Avoid
duplicating logs or accumulating patient-specific instructions.""",
    "efficiency_auditor": """You evaluate agent and workflow efficiency. Identify redundant
passes, weak handoffs and avoidable compute/token cost while preserving issue recall and the
false-repair safety ceiling. Propose measurable optimizations.""",
}

CONCISE_OUTPUT = """Keep the JSON compact: at most 3 items in every list and at most 180
characters per string. Do not repeat evidence, restate the schema, or include implementation code
inside prose fields. Completeness means filling every required field, not writing a long report."""

SUPERVISOR_PROMPT = """You are the top-level supervisor for a governed CMR+LGE QC system.
Prioritize specialist and maintenance proposals, assign owners, cap retries and define acceptance
metrics. Prefer autonomous bounded experiments and reusable rule/tool improvements over human
review, but never authorize direct edits to production labels, credentials or clinical ground truth.
The dependency order is LGE myocardium, then LGE infarct, then paired-slice selection, then CMR
myocardium only on that persisted pair set. Review only the active-stage specialist assessment;
parallel actions must remain inside that stage. If LGE lacks a production consumer, prioritize
building and validating that control-plane connection without inventing a medical rule. A CMR
production blocker is deferred until the final paired-CMR stage.
When workflow_dependency.integration_required is true, stop proposing measurement-only scans.
Follow workflow_dependency.required_output exactly. When lge_generator_required is true, the
eligible action is an isolated_code_experiment owned by qc_code_optimizer implementing and testing
scripts/lge_candidate_generator.py with generate_candidates(data, output_dir). It reads supplied
snapshot images/masks and emits repair/accept_existing/needs_review proposals with reasons.
The trusted controller owns independent scoring, holdout, apply and audit; never edit that controller.
Otherwise the legacy integration action implements scripts/lge_feedback_loop.py and its continuous
runtime/lge_iterations/latest.json audit contract. The consumer must use the supplied active-stage
LGE input and exact stage fields (lge_myo_pending/work/resolved or lge_mi equivalents), reconcile
against the supplied queue length, and must not reuse CMR holdout counts or hard-code an earlier
shadow sample size. An experiments-only consumer is invalid.
The run_evaluation function is pure: it returns a report and the framework driver persists it.
If frozen_split.status is UNAVAILABLE, the report must keep frozen_split_status=UNAVAILABLE and
holdout_excluded=0; never invent or load a different split to make the gate pass.
After a prior measured shadow result identifies a concrete consumer and blocker, you may assign an
isolated snapshot patch to that named production-path file; this is not permission to write production.
If evidence_summary.check_iteration.production_blocker exists, you MUST prioritize one
isolated_code_experiment assigned to its named owner and target_path before approving any new
measurement-only shadow. Preserve the named deterministic gate; let the domain owner infer the rule.
If deferred_production_blocker exists instead, bounded attempts for that unchanged target are
exhausted: do not rename or retry it. Treat those conflicts as abstained residuals and select a
different independently measurable issue class; this deferral is not permission to route labels.
Code, rule and skill changes must be isolated and validated before
promotion. Use iteration_history to prevent repeated low-value loops and route the next unresolved
issue class. Assign isolated_code_experiment only to qc_code_optimizer, rule_draft only to
rule_curator, and skill_draft only to skill_maintainer. Do not rename or restate a prior action
to evade the deterministic duplicate guard; propose a genuinely different bounded experiment
or return no actions. For a real continuation, set continuation_of to the exact prior action_id,
addresses_metric to one exact unmet_metrics string from its reviewer feedback, and
continuation_delta to the concrete new measurement or artifact this action will produce.
These fields are not permission to repeat the same objective or fabricate validation results.
For a bounded read-only experiment executable by the assigned owner in an
isolated snapshot, set auto_execute true; the independent reviewer still decides whether it may
run. Use false only when a specific human-only dependency makes isolated execution impossible.
Only action classes listed in action_ownership have autonomous executors. Never set auto_execute
true for known_tool, human_review, stop, or any other class absent from that mapping.
Human review is the final fallback only after bounded retries are exhausted or the
required clinical evidence cannot be derived. Write every action objective and every acceptance metric in Simplified Chinese so operators can read them without translation. Return only the requested structured object."""


def mailbox(db: Database, run_id: int, role: str, thread_id: str) -> list[dict[str, Any]]:
    """Return the validated content addressed to one role in one thread."""
    return [{
        "message_id": item["message_id"],
        "sender_role": item["sender_role"],
        "message_type": item["message_type"],
        "payload": item["payload"],
    } for item in db.agent_inbox(run_id, role, thread_id=thread_id)]


SUPERVISOR_PROMPT += "\nWhen protocol_repair.required is true, route framework_repair to workflow_maintainer and name configs/lge_protocol_aliases.json. This owner can only repair transport aliases; it cannot alter medical logic, labels, budgets, permissions or validators. Do not send protocol transport faults to doctors. Unsupported faults remain explicit."


def score_agent_output(role: str, value: dict[str, Any]) -> AgentScorecard:
    text = json.dumps(value, ensure_ascii=False).lower()
    evidence_terms = value.get("supporting_issue_types") or value.get("evidence_issue_types") or []
    evidence_coverage = min(1.0, 0.35 + 0.15 * len(evidence_terms))
    actionable_fields = sum(bool(value.get(key)) for key in (
        "suggested_experiment", "proposed_changes", "validation_plan", "actions",
    ))
    actionability = min(1.0, 0.25 + 0.25 * actionable_fields)
    risk_terms = sum(term in text for term in ("risk", "contraindication", "rollback", "human"))
    risk_awareness = min(1.0, 0.3 + 0.2 * risk_terms)
    confidence = float(value.get("confidence", 0.5))
    confidence_calibration = max(0.0, 1.0 - abs(confidence - 0.7))
    composite = (evidence_coverage + actionability + risk_awareness + confidence_calibration) / 4
    return AgentScorecard(
        role=role,
        output_valid=True,
        evidence_coverage=evidence_coverage,
        actionability=actionability,
        risk_awareness=risk_awareness,
        confidence_calibration=confidence_calibration,
        composite_score=composite,
        notes=["Deterministic first-pass rubric; promotion metrics remain authoritative."],
    )


def filter_proposals(proposals: list) -> list:
    """Keep one proposal per action class; drop no-op advice.

    The four maintenance roles share most of their prompt, so on identical
    evidence they tend to restate the same change. Forwarding four near-copies
    makes the supervisor's input look broad while it is really one idea.
    """
    kept: list = []
    for item in proposals:
        if getattr(item, "action_class", "") == "no_change":
            continue
        if any(other.action_class == item.action_class and
               difflib.SequenceMatcher(None, other.problem.lower(), item.problem.lower()).ratio() > 0.75
               for other in kept):
            continue
        kept.append(item)
    return kept


def filter_repeated_actions(actions: list, iteration_history: list[dict[str, Any]],
                            threshold: float = 0.78) -> tuple[list, list[dict[str, str]]]:
    """Block semantic re-proposals even when agents invent a new action ID.

    History is deliberately deterministic and local; an LLM does not get to
    declare its own near-duplicate proposal novel. Numbers in run/action IDs
    are normalized while clinically meaningful words and action class remain.
    """
    previous: list[tuple[str, str, str]] = []
    prior_by_id: dict[str, dict[str, Any]] = {}
    for run in iteration_history:
        for item in run.get("actions", []):
            text = re.sub(r"\d+", "#", str(item.get("objective", "")).lower())
            text = re.sub(r"\s+", " ", text).strip()
            previous.append((str(item.get("action_class", "")), text,
                             str(item.get("action_id", ""))))
            prior_by_id[str(item.get("action_id", ""))] = item
    kept, blocked = [], []
    for action in actions:
        current = re.sub(r"\d+", "#", str(action.objective).lower())
        current = re.sub(r"\s+", " ", current).strip()
        claimed = getattr(action, "continuation_of", None)
        addressed = getattr(action, "addresses_metric", None)
        delta = (getattr(action, "continuation_delta", None) or "").strip()
        if claimed or addressed or delta:
            prior = prior_by_id.get(claimed or "", {})
            feedback = prior.get("review_feedback") or {}
            addressed_check = str(addressed or "").split("==", 1)[0].strip()
            deterministic_failed_checks = {
                name
                for suite in (prior.get("validation_result") or {}).get("suites", [])
                if suite.get("suite") == "lge_stage_consumer_contract"
                for name, passed in (suite.get("checks") or {}).items()
                if passed is False
            }
            failed_consumer_continuation = (
                claimed
                and prior.get("status") in {"HUMAN_REQUIRED", "REJECTED"}
                and action_builds_stage_consumer(model_dump(action))
                and addressed_check in deterministic_failed_checks
                and len(delta) >= 12
            )
            if not (claimed
                    and prior.get("status") in {
                        "HUMAN_REQUIRED", "REJECTED", "PROMOTED_SHADOW", "PROMOTED"
                    }
                    and ((addressed in (feedback.get("required_validation_metrics") or [])
                          and len(delta) >= 12)
                         or failed_consumer_continuation)):
                blocked.append({"action_id": action.action_id,
                                "matches": claimed or "", "similarity": "n/a",
                                "reason": "invalid_continuation_reference"})
                continue
        match = None
        best = 0.0
        for action_class, old, action_id in previous:
            if action_class and action_class != action.action_class:
                continue
            score = difflib.SequenceMatcher(None, old, current).ratio()
            if score > best:
                best, match = score, action_id
        if match and best >= threshold:
            prior_match = prior_by_id.get(match, {})
            exact_failed_consumer_retry = (
                bool(claimed) and claimed == match
                and (prior_match.get("status") == "HUMAN_REQUIRED"
                     or failed_consumer_continuation)
                and action_builds_stage_consumer(model_dump(action))
                and len(delta) >= 12
            )
            continuation = (
                bool(claimed)
                and current != next(old for cls, old, aid in previous if aid == match)
            )
            if continuation or exact_failed_consumer_retry:
                kept.append(action)  # Still requires the independent review.
            else:
                blocked.append({"action_id": action.action_id, "matches": match,
                                "similarity": f"{best:.3f}",
                                "reason": "repeated_or_unverified_continuation"})
        else:
            kept.append(action)
    return kept, blocked


def select_analysis_roles(evidence: dict[str, Any], analysis: FailureAnalysis) -> tuple[list[str], list[str]]:
    """Schedule only the currently eligible stage in the LGE-first DAG."""
    if (evidence.get("protocol_repair") or {}).get("required"):
        return [], ["workflow_maintainer"]
    affected = {str(item).lower() for item in evidence.get("modalities", [])}
    affected &= {"cmr", "lge"}
    if not affected:
        affected = {str(item).lower() for item in analysis.affected_modalities}
        affected &= {"cmr", "lge"}

    stage = str((evidence.get("workflow_dependency") or {}).get("stage") or "")
    if stage == "lge_myo" and "lge" in affected:
        specialists = ["lge_myo_specialist"]
    elif stage == "lge_mi" and "lge" in affected:
        specialists = ["lge_mi_specialist"]
    elif stage == "paired_selection" and "lge" in affected:
        specialists = ["lge_myo_specialist", "lge_mi_specialist"]
    elif stage == "cmr_paired" and "cmr" in affected:
        specialists = ["cmr_specialist"]
    elif "lge" in affected:
        specialists = ["lge_myo_specialist"]
    else:
        specialists = ["cmr_specialist"] if "cmr" in affected else []

    history = evidence.get("iteration_history", [])
    cycle = len(history) + 1
    failure_text = " ".join((analysis.failure_type, analysis.hypothesis,
                             analysis.suggested_experiment)).lower()
    workflow = evidence.get("workflow_dependency") or {}
    if workflow.get("integration_required"):
        maintainers = ["qc_code_optimizer"]
    else:
        maintainers = ["rule_curator"]
        if any(term in failure_text for term in (
                "code", "router", "routing", "workflow", "consumer", "service", "bug")):
            maintainers.insert(0, "qc_code_optimizer")
    if (not workflow.get("integration_required")
            and any(term in failure_text for term in ("workflow", "skill", "knowledge", "handoff"))):
        maintainers.append("skill_maintainer")
    if cycle % 10 == 0 or any(term in failure_text for term in (
            "efficiency", "token", "cost", "latency", "timeout")):
        maintainers.append("efficiency_auditor")
    return specialists, maintainers


def supervisor_output_needs_retry(supervisor: SupervisorDecision,
                                  candidate_actions: list[SupervisorAction],
                                  blocked_actions: list[dict[str, Any]],
                                  autonomy_retry: bool) -> bool:
    """Reject semantically impossible execute/experiment decisions with no actions."""
    empty_experiment = (supervisor.decision in {"execute", "experiment"}
                        and not supervisor.actions)
    return empty_experiment or bool(not candidate_actions and blocked_actions) or autonomy_retry


def filter_workflow_actions(actions: list[SupervisorAction], workflow: dict[str, Any]
                            ) -> tuple[list[SupervisorAction], list[dict[str, Any]]]:
    current_stage = str(workflow.get("stage") or "")
    allowed: list[SupervisorAction] = []
    blocked: list[dict[str, Any]] = []
    for action in actions:
        payload = model_dump(action)
        payload["workflow_stage"] = current_stage
        if workflow.get("protocol_repair_required") and action.action_class != "framework_repair":
            blocked.append({"action_id":action.action_id,"reason":"protocol_repair_before_domain_work"})
            continue
        if action.action_class == "framework_repair":
            from app.lge_protocol import action_allowed
            if workflow.get("protocol_repair_required") and action_allowed(payload):
                allowed.append(action)
            else:
                blocked.append({"action_id":action.action_id,"reason":"framework_repair_scope_or_evidence_missing"})
            continue
        stale_contract = (workflow.get("integration_required")
                          and action_uses_stale_lge_shadow_contract(payload))
        if (action_matches_workflow_stage(payload, current_stage)
                and (not workflow.get("integration_required")
                     or (action_builds_stage_consumer(payload, require_generator=bool(workflow.get("lge_generator_required")))
                         and not stale_contract))):
            allowed.append(action)
        else:
            integration_blocked = (workflow.get("integration_required")
                                   and not action_builds_stage_consumer(payload, require_generator=bool(workflow.get("lge_generator_required"))))
            blocked.append({
                "action_id": action.action_id,
                "reason": ("stale_stage_denominator" if stale_contract
                           else "stage_consumer_required" if integration_blocked
                           else "workflow_stage_mismatch"),
                "action_stage": semantic_action_workflow_stage(payload),
                "active_stage": current_stage,
                "instruction": (workflow.get("required_output")
                                if integration_blocked or stale_contract
                                else "Only propose work for the active dependency stage."),
            })
    return allowed, blocked


def stable_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\|([A-Za-z_][A-Za-z0-9_]*))?\}", value)
        if match:
            first, fallback = match.group(1), match.group(2)
            resolved = os.getenv(first) or (os.getenv(fallback) if fallback else None)
            if not resolved:
                raise RuntimeError(
                    f"missing model environment variable: {first}"
                    + (f" (and fallback {fallback})" if fallback else ""))
            return resolved
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


async def run_shadow_analysis(db: Database, run_id: int, provider_config: Path,
                              agent_config: Path) -> dict[str, Any]:
    details = db.run_details(run_id)
    if details["status"] not in {RunState.QC_FAILED.value, RunState.EVIDENCE_READY.value}:
        raise ValueError("shadow analysis requires a QC_FAILED or EVIDENCE_READY run")
    agents = expand_env(yaml.safe_load(agent_config.read_text(encoding="utf-8")))["agents"]
    registry = ProviderRegistry(provider_config)
    evidence = build_evidence_package(details)
    input_hash = stable_hash(evidence)
    analysis_thread = f"run-{run_id}:analysis"
    db.transition(run_id, RunState.FAILURE_ANALYSIS, current_step="shadow_failure_analysis")

    try:
        db.post_agent_message(
            run_id, thread_id=analysis_thread, sender_role="evidence_builder",
            recipient_role="failure_triage", message_type="evidence_ready", payload=evidence,
        )
        triage_cfg = agents["failure_triage"]
        triage_provider = registry.get(triage_cfg["provider"])
        triage_input = {"role": "failure_triage", "_max_output_tokens": 2500,
                        "role_memory": evidence.get("role_continuity", {}).get(
                            "roles", {}).get("failure_triage", {}),
                        "handoff_messages": mailbox(
                            db, run_id, "failure_triage", analysis_thread), **evidence}
        analysis = await invoke_traced(
            db=db, run_id=run_id, thread_id=analysis_thread,
            role="failure_triage", provider_name=triage_cfg["provider"],
            provider=triage_provider,
            system_prompt=TRIAGE_PROMPT + "\n" + CONCISE_OUTPUT,
            input_data=triage_input,
            output_schema=FailureAnalysis,
            model=triage_cfg["model"],
        )
        db.save_agent_output(run_id, "failure_triage", triage_cfg["provider"], triage_cfg["model"],
                             "SUCCEEDED", input_hash, model_dump(analysis))

        specialist_roles, maintenance_roles = select_analysis_roles(evidence, analysis)
        for role in specialist_roles + maintenance_roles:
            db.post_agent_message(
                run_id, thread_id=analysis_thread, sender_role="failure_triage",
                recipient_role=role, message_type="failure_analysis",
                payload=model_dump(analysis),
            )

        async def call_specialist(role: str):
            cfg = agents[role]
            provider = registry.get(cfg["provider"])
            branch_scope = next((item for item in evidence.get("workstreams", [])
                                 if item.get("specialist_role") == role), {})
            specialist_input = {
                "role": role,
                "_max_output_tokens": 1600,
                "role_memory": evidence.get("role_continuity", {}).get(
                    "roles", {}).get(role, {}),
                "evidence_summary": evidence,
                "triage": model_dump(analysis),
                "branch_scope": branch_scope,
                "branch_instruction": (
                    "Respect workflow_dependency strictly. Work only on the active stage; never "
                    "start full-cohort CMR QC before LGE myocardium, LGE MI and paired-slice "
                    "selection are complete. If the active production consumer is missing, "
                    "propose a bounded read-only integration/evaluation step."
                ),
                "handoff_messages": mailbox(db, run_id, role, analysis_thread),
                "mode": "shadow_no_mutation",
            }
            item_hash = stable_hash(specialist_input)
            try:
                output = await invoke_traced(
                    db=db, run_id=run_id, thread_id=analysis_thread,
                    role=role, provider_name=cfg["provider"], provider=provider,
                    system_prompt=SPECIALIST_PROMPTS[role] + "\n" + CONCISE_OUTPUT,
                    input_data=specialist_input,
                    output_schema=SpecialistAssessment,
                    model=cfg["model"],
                )
                db.save_agent_output(run_id, role, cfg["provider"], cfg["model"], "SUCCEEDED",
                                     item_hash, model_dump(output))
                return output
            except Exception as exc:
                db.save_agent_output(
                    run_id, role, cfg["provider"], cfg["model"], "FAILED", item_hash, {},
                    f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                )
                return None

        specialist_results = await asyncio.gather(
            *(call_specialist(role) for role in specialist_roles)
        )
        specialist_outputs = [item for item in specialist_results if item is not None]
        for item in specialist_outputs:
            for role in maintenance_roles:
                db.post_agent_message(
                    run_id, thread_id=analysis_thread, sender_role=item.role,
                    recipient_role=role, message_type="specialist_assessment",
                    payload=model_dump(item),
                )

        async def call_maintainer(role: str):
            cfg = agents[role]
            provider = registry.get(cfg["provider"])
            maintenance_input = {
                "role": role,
                "_max_output_tokens": 3200,
                "role_memory": evidence.get("role_continuity", {}).get(
                    "roles", {}).get(role, {}),
                "evidence_summary": evidence,
                "triage": model_dump(analysis),
                "specialist_assessments": [model_dump(item) for item in specialist_outputs],
                "workflow_dependency": evidence.get("workflow_dependency", {}),
                "handoff_messages": mailbox(db, run_id, role, analysis_thread),
                "constraints": {
                    "production_writes": False,
                    "isolated_experiments_only": True,
                    "max_attempts_per_action": 3,
                },
            }
            item_hash = stable_hash(maintenance_input)
            try:
                output = await invoke_traced(
                    db=db, run_id=run_id, thread_id=analysis_thread,
                    role=role, provider_name=cfg["provider"], provider=provider,
                    system_prompt=MAINTENANCE_PROMPTS[role] + "\n" + CONCISE_OUTPUT,
                    input_data=maintenance_input,
                    output_schema=MaintenanceProposal,
                    model=cfg["model"],
                )
                db.save_agent_output(run_id, role, cfg["provider"], cfg["model"], "SUCCEEDED",
                                     item_hash, model_dump(output))
                return output
            except Exception as exc:
                db.save_agent_output(
                    run_id, role, cfg["provider"], cfg["model"], "FAILED", item_hash, {},
                    f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                )
                return None

        maintenance_results = await asyncio.gather(
            *(call_maintainer(role) for role in maintenance_roles)
        )
        maintenance_outputs = filter_proposals(
            [item for item in maintenance_results if item is not None]
        )
        for item in maintenance_outputs:
            db.post_agent_message(
                run_id, thread_id=analysis_thread, sender_role=item.role,
                recipient_role="supervisor", message_type="maintenance_proposal",
                payload=model_dump(item),
            )

        supervisor_cfg = agents["supervisor"]
        supervisor_provider = registry.get(supervisor_cfg["provider"])
        supervisor_input = {
            "role": "supervisor",
            "_max_output_tokens": 4000,
            "role_memory": evidence.get("role_continuity", {}).get(
                "roles", {}).get("supervisor", {}),
            "evidence_summary": evidence,
            "triage": model_dump(analysis),
            "specialist_assessments": [model_dump(item) for item in specialist_outputs],
            "workflow_dependency": evidence.get("workflow_dependency", {}),
            "maintenance_proposals": [model_dump(item) for item in maintenance_outputs],
            "handoff_messages": mailbox(db, run_id, "supervisor", analysis_thread),
            "evaluation_policy": {
                "primary": ["target_issue_recall", "false_repair_rate", "new_critical_qc_issues"],
                "non_regression": ["myo_dice", "mi_dice"],
                "efficiency": ["wall_time", "gpu_hours", "llm_calls", "manual_escalation_rate"],
                "max_attempts": 3,
            },
            "action_ownership": ACTION_OWNERS,
        }
        supervisor = await invoke_traced(
            db=db, run_id=run_id, thread_id=analysis_thread,
            role="supervisor", provider_name=supervisor_cfg["provider"],
            provider=supervisor_provider,
            system_prompt=SUPERVISOR_PROMPT + "\n" + CONCISE_OUTPUT,
            input_data=supervisor_input,
            output_schema=SupervisorDecision,
            model=supervisor_cfg["model"],
        )
        db.save_agent_output(run_id, "supervisor", supervisor_cfg["provider"],
                             supervisor_cfg["model"], "SUCCEEDED",
                             stable_hash(supervisor_input), model_dump(supervisor))
        budget = max(1, min(int(getattr(supervisor, "max_parallel_actions", 2)), 4))
        candidate_actions, blocked_actions = filter_repeated_actions(
            supervisor.actions[:budget], evidence.get("iteration_history", []))
        candidate_actions, ownership_blocked = enforce_action_ownership(candidate_actions)
        blocked_actions.extend(ownership_blocked)
        candidate_actions, executor_blocked = enforce_autonomous_executor_contract(candidate_actions)
        blocked_actions.extend(executor_blocked)
        candidate_actions, workflow_blocked = filter_workflow_actions(
            candidate_actions, evidence["workflow_dependency"])
        blocked_actions.extend(workflow_blocked)
        autonomy_retry = bool(candidate_actions) and not any(
            action.auto_execute for action in candidate_actions
        ) and any(action.action_class in ACTION_OWNERS for action in candidate_actions)
        empty_experiment = (supervisor.decision in {"execute", "experiment"}
                            and not supervisor.actions)
        if supervisor_output_needs_retry(
                supervisor, candidate_actions, blocked_actions, autonomy_retry):
            initial_supervisor, initial_candidates = supervisor, candidate_actions
            guard_feedback = list(blocked_actions)
            if empty_experiment:
                guard_feedback.append({
                    "reason": "experiment_without_actions",
                    "instruction": (
                        "Select one feasible action from the supplied proposals, or change the "
                        "decision to stop_no_gain/needs_human_review. Do not reference a phantom action."
                    ),
                })
            retry_input = {**supervisor_input,
                           "guard_feedback": guard_feedback,
                           "previous_supervisor_decision": model_dump(supervisor),
                           "retry_instruction": (
                               "The experiment decision contained no executable action. Select one feasible bounded action from the supplied proposals, or change the decision to stop_no_gain/needs_human_review. Do not reference a phantom action."
                               if empty_experiment else
                               "All prior actions were rejected. Read progress_ledger and the matched prior action's reviewer feedback. "
                               "Propose a genuinely different bounded action, or explicitly continue a blocked action "
                               "with its exact unmet metric and concrete new measurement. If neither is possible, return no actions."
                               if not candidate_actions else
                               "The valid bounded experiment is marked manual. Reconsider whether its named owner can run it in an isolated snapshot. Set auto_execute true only if feasible; otherwise retain false and explain the specific human-only dependency in rationale.")}
            supervisor = await invoke_traced(
                db=db, run_id=run_id, thread_id=analysis_thread,
                role="supervisor", provider_name=supervisor_cfg["provider"],
                provider=supervisor_provider,
                system_prompt=SUPERVISOR_PROMPT + "\n" + CONCISE_OUTPUT,
                input_data=retry_input, output_schema=SupervisorDecision,
                model=supervisor_cfg["model"],
            )
            db.save_agent_output(run_id, "supervisor", supervisor_cfg["provider"],
                                 supervisor_cfg["model"], "SUCCEEDED",
                                 stable_hash(retry_input), model_dump(supervisor))
            budget = max(1, min(int(getattr(supervisor, "max_parallel_actions", 2)), 4))
            candidate_actions, retry_blocked = filter_repeated_actions(
                supervisor.actions[:budget], evidence.get("iteration_history", []))
            candidate_actions, ownership_blocked = enforce_action_ownership(candidate_actions)
            blocked_actions.extend(retry_blocked)
            blocked_actions.extend(ownership_blocked)
            candidate_actions, executor_blocked = enforce_autonomous_executor_contract(candidate_actions)
            blocked_actions.extend(executor_blocked)
            candidate_actions, workflow_blocked = filter_workflow_actions(
                candidate_actions, evidence["workflow_dependency"])
            blocked_actions.extend(workflow_blocked)
            if autonomy_retry and not candidate_actions:
                supervisor, candidate_actions = initial_supervisor, initial_candidates
        if blocked_actions:
            db.save_agent_output(
                run_id, "duplicate_guard", "deterministic", "semantic-v1", "SUCCEEDED",
                stable_hash({"history": evidence.get("iteration_history", []),
                             "actions": blocked_actions}),
                {"blocked_actions": blocked_actions},
            )
        dependency = evidence.get("workflow_dependency", {})
        workflow_stage = dependency.get("stage")
        workflow_contract = dependency.get("contract")

        def staged_action_payload(action: SupervisorAction) -> dict[str, Any]:
            payload = model_dump(action)
            payload["workflow_stage"] = workflow_stage
            payload["workflow_dependency_contract"] = workflow_contract
            return payload

        vetted_supervisor = model_dump(supervisor)
        vetted_supervisor["actions"] = [
            staged_action_payload(action) for action in candidate_actions
        ]
        db.post_agent_message(
            run_id, thread_id=analysis_thread, sender_role="supervisor",
            recipient_role="independent_reviewer", message_type="supervisor_decision",
            payload=vetted_supervisor,
        )
        for action in candidate_actions:
            db.post_agent_message(
                run_id, thread_id=f"run-{run_id}:{action.action_id}",
                sender_role="supervisor", recipient_role="independent_reviewer",
                message_type="action_proposal", payload=staged_action_payload(action),
            )

        review_cfg = agents["independent_reviewer"]
        review_provider = registry.get(review_cfg["provider"])
        review_input = {"role": "independent_reviewer", "_max_output_tokens": 1500,
                        "role_memory": evidence.get("role_continuity", {}).get(
                            "roles", {}).get("independent_reviewer", {}),
                        "review_stage": "pre_execution_isolated_experiment",
                        "execution_scope": "owner_snapshot_no_production_write",
                        "evidence_summary": evidence,
                        "proposal": model_dump(analysis),
                        "specialist_assessments": [model_dump(item) for item in specialist_outputs],
                        "maintenance_proposals": [model_dump(item) for item in maintenance_outputs],
                        "supervisor_decision": vetted_supervisor,
                        "handoff_messages": mailbox(
                            db, run_id, "independent_reviewer", analysis_thread),
                        "mode": "shadow_no_mutation"}
        review = await invoke_traced(
            db=db, run_id=run_id, thread_id=analysis_thread,
            role="independent_reviewer", provider_name=review_cfg["provider"],
            provider=review_provider,
            system_prompt=REVIEW_PROMPT + "\n" + PLAN_REVIEW_POLICY + "\n" + CONCISE_OUTPUT,
            input_data=review_input,
            output_schema=ReviewDecision,
            model=review_cfg["model"],
        )
        db.save_agent_output(run_id, "independent_reviewer", review_cfg["provider"], review_cfg["model"],
                             "SUCCEEDED", stable_hash(review_input), model_dump(review))
        reviewer_accepted = review.decision == "accept_experiment"
        for action in candidate_actions:
            action_payload = staged_action_payload(action)
            action_payload["review_approved"] = reviewer_accepted
            action_payload["review_decision"] = review.decision
            action_thread = f"run-{run_id}:{action.action_id}"
            db.post_agent_message(
                run_id, thread_id=action_thread, sender_role="supervisor",
                recipient_role=action.owner_role, message_type="assigned_action",
                payload=action_payload,
            )
            db.post_agent_message(
                run_id, thread_id=action_thread, sender_role="independent_reviewer",
                recipient_role=action.owner_role, message_type="action_review",
                payload=model_dump(review),
            )
            if action.action_class == "human_review" or not action.auto_execute:
                action_status = "HUMAN_REQUIRED"
            elif reviewer_accepted:
                action_status = "APPROVED"
            else:
                action_status = "REJECTED"
            db.save_maintenance_action(run_id, action_payload, status=action_status)
        scored = {
            "failure_triage": model_dump(analysis),
            **{item.role: model_dump(item) for item in specialist_outputs},
            **{item.role: model_dump(item) for item in maintenance_outputs},
            "supervisor": model_dump(supervisor),
            "independent_reviewer": model_dump(review),
        }
        scorecards = []
        for role, output in scored.items():
            score = score_agent_output(role, output)
            scorecards.append(model_dump(score))
            db.save_agent_scorecard(run_id, role, model_dump(score))

        has_bounded_action = any(
            action.action_class in {"isolated_code_experiment", "rule_draft", "skill_draft"}
            for action in candidate_actions
        )
        target = (RunState.EXPERIMENT_PROPOSED
                  if reviewer_accepted and has_bounded_action
                  else RunState.NEEDS_REVIEW)
        db.transition(run_id, target, current_step="shadow_analysis_complete")
        return {"evidence": evidence, "analysis": model_dump(analysis),
                "specialists": [model_dump(item) for item in specialist_outputs],
                "maintenance": [model_dump(item) for item in maintenance_outputs],
                "supervisor": model_dump(supervisor),
                "review": model_dump(review),
                "scorecards": scorecards,
                "final_state": target.value}
    except Exception as exc:
        db.transition(run_id, RunState.ERROR, current_step="shadow_failure_analysis",
                      error_message=str(exc))
        raise
