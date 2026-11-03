"""Opt-in normal Autopilot dispatch for typed, independently reviewed training.

No configuration is enabled by this module. Operator configuration belongs in
configs/segmentation_dispatch.json; Owner output cannot grant capabilities,
increase session limits, select a command or manufacture an approval file.
This is prepared-plan dispatch, NOT autonomous network/artifact generation.
New job/architecture hashes still require an operator configuration update.
"""
from __future__ import annotations

import json
import math
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path

from agents.governance import SEGMENTATION_OWNERS
from contracts.compat import model_dump, model_validate
from contracts.models import SegmentationCyclePlan
from app.control_checkpoint import save_journal

CONFIG = "configs/segmentation_dispatch.json"
INPUT_FIELDS = ("target", "registry_root", "parent_id", "policy_id", "job",
                "architecture", "dataset_manifest", "cohort")


def configuration(root: Path) -> dict:
    path = Path(root) / CONFIG
    if not path.exists():
        return {"enabled": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("invalid_segmentation_dispatch_config")
    if not value["enabled"]:
        return {"enabled": False}
    if (set(value) - {"automatic_reviews"} != {"schema", "enabled", "max_actions", "walltime_seconds", "plan_inputs"}
            or type(value.get("automatic_reviews", False)) is not bool
            or value["schema"] != "segmentation-dispatch-v1"
            or type(value["max_actions"]) is not int or not 1 <= value["max_actions"] <= 3
            or type(value["walltime_seconds"]) is not int or not 1 <= value["walltime_seconds"] <= 21600
            or not isinstance(value["plan_inputs"], list) or not 1 <= len(value["plan_inputs"]) <= 16):
        raise ValueError("invalid_segmentation_dispatch_config")
    for item in value["plan_inputs"]:
        if (not isinstance(item, dict) or set(item) != set(INPUT_FIELDS)
                or item["target"] not in SEGMENTATION_OWNERS
                or not isinstance(item["registry_root"], str) or not Path(item["registry_root"]).is_absolute()
                or any(not isinstance(item[k], str) or not item[k].strip() for k in ("parent_id", "policy_id"))):
            raise ValueError("invalid_segmentation_plan_inputs")
        for key in ("job", "architecture", "dataset_manifest", "cohort"):
            binding = item[key]
            if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                    or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
                    or not isinstance(binding["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", binding["sha256"])):
                raise ValueError("invalid_segmentation_plan_binding")
    return value


def proposal_context(root: Path, run_id: int) -> dict:
    """Trusted artifact descriptions only, never manifests, patient rows or reviews."""
    try:
        config = configuration(root)
        if not config["enabled"]:
            return {"enabled": False}
        return {"enabled": True, "run_id": run_id, "plan_inputs": config["plan_inputs"],
                "target_owners": SEGMENTATION_OWNERS, "max_actions": config["max_actions"],
                "walltime_seconds": config["walltime_seconds"]}
    except (OSError, ValueError, TypeError):
        return {"enabled": False, "error_code": "INVALID_CONFIGURATION"}


def checked_plan(payload: dict, run_id: int, config: dict) -> dict:
    plan = model_dump(model_validate(SegmentationCyclePlan, payload.get("segmentation_cycle_plan")))
    owner = SEGMENTATION_OWNERS[plan["target"]]
    if (payload.get("action_class") != "segmentation_training"
            or payload.get("owner_role") != owner or plan["owner_role"] != owner or plan["owner"] != owner
            or plan["run_id"] != run_id or plan["action_id"] != payload.get("action_id")
            or not config.get("enabled")
            or {k: plan[k] for k in INPUT_FIELDS} not in config.get("plan_inputs", [])):
        raise ValueError("segmentation_plan_not_bound_to_configured_owner_inputs")
    return plan


def proposal_allowed(payload: dict, context: dict, specialists: list) -> bool:
    try:
        plan = checked_plan(payload, context["run_id"], context)
        for item in specialists:
            item = model_dump(item) if hasattr(item, "dict") else item
            if (item.get("role") == plan["owner_role"] and item.get("applicable") is True
                    and item.get("segmentation_cycle_plan") == plan):
                return True
    except (KeyError, ValueError, TypeError):
        pass
    return False


def _review_request(db, plan, kind, binding):
    """Durable request only, addressed to the existing independently configured role."""
    from app.segmentation_cycle import _hash
    thread = f"run-{plan['run_id']}:{plan['action_id']}"
    digest = _hash(binding)
    for message in db.agent_inbox(plan["run_id"], "independent_reviewer", thread_id=thread):
        if (message["sender_role"] == "segmentation_controller" and message["message_type"] == kind
                and message["payload"].get("request_sha256") == digest):
            return message["message_id"]
    return db.post_agent_message(plan["run_id"], thread_id=thread,
        sender_role="segmentation_controller", recipient_role="independent_reviewer",
        message_type=kind, payload=dict(binding, request_sha256=digest))


def _provenance_messages(db, plan, outcome):
    """Reference existing independent attestations; never create their facts."""
    from app.segmentation_cycle import plan_sha256
    thread = f"run-{plan['run_id']}:{plan['action_id']}"
    rows = db.agent_inbox(plan["run_id"], plan["owner_role"], thread_id=thread)
    rows = [m for m in rows if m["sender_role"] in {"independent_evaluator", "independent_reviewer"}]
    expected = {
        "reference_provenance": {"cohort_sha256": plan["cohort"]["sha256"], "reference_sha256": outcome["report"]["reference_sha256"]},
        "parent_exposure": {"parent_id": plan["parent_id"], "cohort_sha256": plan["cohort"]["sha256"],
                            "training_evaluation_patient_disjoint": "ESTABLISHED"},
        "split_evidence": {"dataset_sha256": plan["dataset_manifest"]["sha256"], "cohort_sha256": plan["cohort"]["sha256"],
                           "validation_test_patient_disjoint": "ESTABLISHED", "training_evaluation_patient_disjoint": "ESTABLISHED"},
    }
    found = {}
    for name, fields in expected.items():
        candidates = [r for r in rows if r["message_type"] == "segmentation_" + name]
        fields = dict(fields, status="VERIFIED", plan_sha256=plan_sha256(plan))
        if not candidates or any(candidates[-1]["payload"].get(k) != v for k, v in fields.items()):
            return None
        found[name + "_message_id"] = candidates[-1]["message_id"]
    return found


def _review_step(root, db, plan, outcome, state, path, config, checkpoint, providers_path, agents_path):
    from app import segmentation_cycle as cycle, segmentation_registry as registry
    phase = {"AWAITING_EVALUATION": "evaluation", "AWAITING_ACTIVATION_APPROVAL": "activation"}.get(outcome["stage"])
    if not phase:
        return False
    if (not outcome.get("report") or not outcome.get("candidate_id")
            or outcome.get("report_sha256") != cycle._hash(outcome["report"])):
        state["review_status"] = "MEASUREMENT_REQUIRED"
        return False
    candidate = registry.resolve_candidate(Path(plan["registry_root"]), outcome["candidate_id"])
    binding = {"phase": phase, "plan_sha256": cycle.plan_sha256(plan), "report_sha256": outcome["report_sha256"],
               "candidate_id": outcome["candidate_id"], "parent_id": plan["parent_id"],
               "policy_id": plan["policy_id"], "target": plan["target"],
               "candidate_artifact_hashes": {k: v["sha256"] for k, v in candidate["artifacts"].items()},
               "metrics": outcome["report"]["metrics"]}
    if phase == "activation":
        assessment = outcome.get("assessment") or {}
        if assessment.get("passed") is not True or not assessment.get("assessment_id"):
            state["review_status"] = "ASSESSMENT_REQUIRED"
            return False
        binding["assessment_id"] = assessment["assessment_id"]
    request_id = _review_request(db, plan, "segmentation_" + phase + "_request", binding)
    sources = _provenance_messages(db, plan, outcome)
    if sources is None:
        state["review_status"] = "MANUAL_EVIDENCE_REQUIRED"
        return False
    if not config.get("automatic_reviews", False):
        state["review_status"] = "MANUAL_REVIEW_REQUESTED"
        return False
    attempts = state.setdefault("review_attempts", {})
    if phase in attempts:
        state["review_status"] = ("MANUAL_REVIEW_REQUIRED" if attempts[phase].get("status") == "RESPONDED"
                                  else "REVIEW_ALREADY_REQUESTED")
        return False
    if not checkpoint():
        state["review_status"] = "REVIEW_BUDGET_BLOCKED"
        return False
    from agents.implementation_review import review_segmentation_stage
    attempts[phase] = {"request_id": request_id, "status": "STARTED"}
    save_journal(path, state)  # paid-call claim is durable before invoking the configured role
    request = dict(binding, source_message_ids=sources)
    if phase == "activation":
        request["assessment"] = outcome["assessment"]
    review_segmentation_stage(db=db, plan=plan, request=request, request_id=request_id,
                              providers_path=providers_path, agents_path=agents_path)
    attempts[phase]["status"] = "RESPONDED"
    state["review_status"] = "REVIEW_RESPONDED"
    save_journal(path, state)
    # Only run_cycle's read-only DB adapters may recognize the persisted decision.
    return True


def control_ready(root: Path, *, now=None) -> bool:
    """Read existing grants only. No arm/resume/reserve that can reset budgets."""
    try:
        operator = json.loads((root / "runtime/control/operator_state.json").read_text(encoding="utf-8"))
        budget = json.loads((root / "runtime/control/run_budget.json").read_text(encoding="utf-8"))
        stamp = time.time() if now is None else now
        return (operator.get("paused") is False and budget.get("active") is True
                and budget.get("status") == "RUNNING" and type(budget["deadline"]) in (int, float)
                and math.isfinite(budget["deadline"]) and stamp < budget["deadline"]
                and all(type(budget["used"][k]) is int and type(budget["limits"][k]) is int
                        and 0 <= budget["used"][k] < budget["limits"][k] for k in ("analyses", "requests")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def readiness(root: Path) -> dict:
    """Path-free board projection; cached metadata/DB only, never a dataset scan.

    Schema: enabled(bool), status/reason_code(str), missing(list[str]),
    queue({inspected, has_more, approved, running, waiting_approval}), current
    (null or {status,target}). Queue counts cover at most 100 pending actions;
    None means the DB could not be observed, not an invented zero.
    """
    root = Path(root)
    result = {"enabled": False, "status": "DISABLED", "reason_code": "CONFIG_MISSING",
              "missing": ["trusted_config"],
              "queue": {"inspected": None, "has_more": None, "approved": None,
                        "running": None, "waiting_approval": None}, "current": None}
    try:
        if not (root / CONFIG).exists():
            return result
        config = configuration(root)
        if not config["enabled"]:
            result.update(reason_code="CONFIG_DISABLED", missing=["enabled_trusted_config"])
            return result
        result["enabled"] = True
    except (OSError, ValueError, TypeError):
        result.update(status="BLOCKED", reason_code="CONFIG_INVALID")
        return result
    try:
        from app.segmentation_host import ReadOnlyDatabase
        from agents.governance import agent_action_authorized, implementation_review_payload
        from app.segmentation_cycle import Stage, implementation_sha256
        db = ReadOnlyDatabase(root / "runtime/workflow.sqlite")
        with db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM maintenance_actions WHERE action_class='segmentation_training' "
                "AND status IN ('APPROVED','RUNNING','SUCCEEDED') ORDER BY run_id DESC,action_id LIMIT 101"
            ).fetchall()
        queue = {"inspected": min(100, len(rows)), "has_more": len(rows) > 100,
                 "approved": 0, "running": 0, "waiting_approval": 0}
        for raw in rows[:100]:
            row = dict(raw)
            payload = json.loads(row["payload_json"])
            if row["status"] == "RUNNING":
                queue["running"] += 1
            try:
                plan = checked_plan(payload, row["run_id"], config)
                review = implementation_review_payload(db, row["run_id"], row["action_id"], plan["owner_role"])
                approved = (payload.get("auto_execute") is True
                            and agent_action_authorized(db, row["run_id"], row, payload)
                            and review and review.get("decision") == "accept_experiment"
                            and not review.get("required_validation_metrics")
                            and review.get("implementation_sha256") == implementation_sha256(plan))
            except (KeyError, ValueError, TypeError):
                approved = False
            queue["approved" if approved else "waiting_approval"] += 1
            if result["current"] is None and row["status"] == "RUNNING":
                from app.segmentation_cycle import _hash
                claim = root / "runtime/segmentation_dispatch" / (_hash({"run_id": row["run_id"], "action_id": row["action_id"]}) + ".json")
                if claim.is_file():
                    state = json.loads(claim.read_text(encoding="utf-8"))
                    target = (payload.get("segmentation_cycle_plan") or {}).get("target")
                    if target in SEGMENTATION_OWNERS and state.get("status") in {s.value for s in Stage} | {"EXECUTING", "AWAITING_MODEL_SOURCES"}:
                        result["current"] = {"status": state["status"], "target": target}
                        reason = state.get("review_status")
                        if reason in {"MANUAL_EVIDENCE_REQUIRED", "MANUAL_REVIEW_REQUESTED", "REVIEW_ALREADY_REQUESTED",
                                      "REVIEW_BUDGET_BLOCKED", "REVIEW_RESPONDED", "MODEL_SOURCES_REQUIRED", "MANUAL_REVIEW_REQUIRED",
                                      "MEASUREMENT_REQUIRED", "ASSESSMENT_REQUIRED"}:
                            result["current"]["review_status"] = reason
        result["queue"] = queue
        summary_path = root / "runtime/segmentation_training/latest.json"
        if result["current"] is None and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") in {stage.value for stage in Stage} and summary.get("target") in SEGMENTATION_OWNERS:
                result["current"] = {"status": summary["status"], "target": summary["target"]}
        if not control_ready(root):
            result.update(status="PAUSED", reason_code="PAUSED_OR_BUDGET_UNAVAILABLE",
                          missing=["active_budget_or_operator_resume"])
        elif result["current"] and result["current"].get("review_status") in {
                "MANUAL_EVIDENCE_REQUIRED", "MANUAL_REVIEW_REQUESTED", "REVIEW_ALREADY_REQUESTED", "MODEL_SOURCES_REQUIRED", "MANUAL_REVIEW_REQUIRED"}:
            result.update(status="BLOCKED", reason_code=result["current"]["review_status"],
                          missing=["independent_provenance" if result["current"]["review_status"] == "MANUAL_EVIDENCE_REQUIRED"
                                   else "baseline_source_bundle" if result["current"]["review_status"] == "MODEL_SOURCES_REQUIRED"
                                   else "independent_review_response"])
        elif not queue["approved"]:
            result.update(status="BLOCKED", reason_code="APPROVED_PLAN_MISSING", missing=["exact_plan_approval"])
        else:
            active = result["current"] and result["current"]["status"] in {"TRAINING", "PREDICTING", "ACTIVATING"}
            result.update(status="RUNNING" if active else "READY", reason_code="RUNNING" if active else "READY", missing=[])
    except Exception:
        result.update(status="BLOCKED", reason_code="GOVERNANCE_UNAVAILABLE", missing=["governance_database"])
    return result


def dispatch(root: Path, db, run_id: int, action_id: str, *, providers_path: Path,
             agents_path: Path) -> dict:
    """Normal authorized action entry, using the same trusted host as its CLI.

    The caller is Autopilot, never an HTTP payload. The host owns subprocess tree
    cleanup and checkpoints; its immutable plan is reread from the trusted DB.
    An OS lock serializes sessions. Claim before requesting review or execution;
    ambiguous interrupted work cannot silently buy another review or retrain.
    """
    from app import segmentation_cycle as cycle
    from app import segmentation_training as training
    from agents.governance import agent_action_authorized, implementation_review_payload
    from app.dataset_control import _private_dir, _safe_path
    result = lambda status, **extra: dict(status=status, **extra)
    action_checked, path, state = False, None, None
    try:
        root = training._absolute(str(root))
        config = configuration(root)
        if not config["enabled"]:
            return result("DISABLED")
        if not control_ready(root):
            return result("PAUSED")
        row = db.get_maintenance_action(run_id, action_id)
        payload = row["payload"]
        if not payload.get("auto_execute") or not agent_action_authorized(db, run_id, row, payload):
            return result("AWAITING_APPROVAL")
        action_checked = True
        plan = checked_plan(payload, run_id, config)
        # Host is deliberately imported only after feature and authority checks.
        from app.segmentation_host import make_controller
        checked, _, _, _ = cycle._preflight(plan)
        folder = _safe_path(root / "runtime/segmentation_dispatch")
        for key in ("dataset_root", "output_dir"):
            source = Path(checked["job"][key])
            if folder.is_relative_to(source) or source.is_relative_to(folder):
                raise ValueError("dispatch_output_overlaps_job")
        _private_dir(folder)
        with training._job_lock(folder / ".lock"):
            key = cycle._hash({"run_id": run_id, "action_id": action_id})
            path = _safe_path(folder / (key + ".json"))
            digest = cycle.plan_sha256(plan)
            state = training.read_json(path) if path.exists() else None
            if state and state.get("plan_sha256") != digest:
                raise ValueError("action_plan_changed")
            if state and state["status"] in {"FAILED", "CANCELLED", "ACTIVATED", "RETAIN_PARENT"}:
                return result(state["status"])
            if state and state["status"] in {"REVIEWING", "EXECUTING"}:
                state.update(status="FAILED", error_code="INTERRUPTED_NO_RETRY")
                save_journal(path, state)
                db.update_maintenance_action(run_id, action_id, status="REJECTED")
                return result("FAILED", error_code="INTERRUPTED_NO_RETRY")
            if not control_ready(root):
                return result("PAUSED")
            if state is None:
                budget = training.read_json(root / "runtime/control/run_budget.json")
                started = budget.get("started_at")
                if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
                    return result("PAUSED", error_code="INVALID_SESSION")
                ledger_path = _safe_path(folder / "sessions.json")
                ledger = training.read_json(ledger_path) if ledger_path.exists() else {}
                session_id = cycle._hash({"started_at": started})
                session = ledger.setdefault(session_id, {"limit": config["max_actions"], "actions": []})
                cap = min(session["limit"], config["max_actions"])
                if key in session["actions"]:
                    return result("FAILED", error_code="MISSING_CLAIM_NO_RETRY")
                if len(session["actions"]) >= cap:
                    return result("BUDGET_EXHAUSTED")
                session["actions"].append(key)
                save_journal(ledger_path, ledger)
                state = {"plan_sha256": digest, "status": "CLAIMED", "run_id": run_id,
                         "action_id": action_id, "deadline": min(time.time() + config["walltime_seconds"], budget["deadline"])}
                save_journal(path, state)

            def checkpoint():
                return (time.time() < state["deadline"] and control_ready(root)
                        and configuration(root).get("enabled") is True)

            if time.time() >= state["deadline"]:
                state.update(status="FAILED", error_code="TIME_LIMIT")
                save_journal(path, state)
                db.update_maintenance_action(run_id, action_id, status="REJECTED")
                return result("FAILED", error_code="TIME_LIMIT")
            if not checkpoint():
                return result("PAUSED")
            prior = implementation_review_payload(db, run_id, action_id, plan["owner_role"])
            if prior is None:
                from agents.implementation_review import review_implementation
                state["status"] = "REVIEWING"
                save_journal(path, state)
                # Real configured reviewer call in normal operation. Tests mock the
                # provider boundary; no dict/filename is treated as approval.
                review_implementation(db=db, run_id=run_id, action=payload, implementation=plan,
                    validation={"passed": True, "promotion_scope": "segmentation_training",
                                "gate": "segmentation_cycle_preflight", "plan_sha256": digest},
                    providers_path=providers_path, agents_path=agents_path)
                prior = implementation_review_payload(db, run_id, action_id, plan["owner_role"])
            if (not prior or prior.get("decision") != "accept_experiment"
                    or prior.get("required_validation_metrics")
                    or prior.get("implementation_sha256") != cycle.implementation_sha256(plan)):
                state["status"] = "FAILED"
                save_journal(path, state)
                db.update_maintenance_action(run_id, action_id, status="REJECTED")
                return result("FAILED", error_code="EXACT_PLAN_REVIEW_REQUIRED")
            if not checkpoint():
                state["status"] = "PAUSED"
                save_journal(path, state)
                return result("PAUSED")
            from app.segmentation_host import SourceUnavailable
            try:
                controller = make_controller(root, plan, database=db.path,
                                             prediction_seconds=config["walltime_seconds"])
            except SourceUnavailable:
                _review_request(db, plan, "segmentation_source_request", {
                    "plan_sha256": digest, "parent_id": plan["parent_id"], "target": plan["target"]})
                state.update(status="AWAITING_MODEL_SOURCES", review_status="MODEL_SOURCES_REQUIRED")
                save_journal(path, state)
                db.update_maintenance_action(run_id, action_id, status="RUNNING")
                return result("AWAITING_MODEL_SOURCES", review_status="MODEL_SOURCES_REQUIRED")
            host_control = controller.control
            def bounded_control(request):
                if not checkpoint():
                    return cycle.ControlDecision(cycle.Control.PAUSE, "dispatch_pause_or_deadline")
                return host_control(request)
            controller = replace(controller, control=bounded_control)
            state["status"] = "EXECUTING"
            save_journal(path, state)
            db.update_maintenance_action(run_id, action_id, status="RUNNING")
            for _ in range(3):  # evaluation review, activation review, terminal reconciliation
                outcome = cycle.run_cycle(root, plan, controller=controller, execute=True)
                if not _review_step(root, db, plan, outcome, state, path, config, checkpoint,
                                    providers_path, agents_path):
                    break
            stage = cycle.Stage(outcome["stage"]).value
            state["status"] = stage
            save_journal(path, state)
            db_status = ("VALIDATED" if stage in {"ACTIVATED", "RETAIN_PARENT"}
                         else "REJECTED" if stage in {"FAILED", "CANCELLED"} else "RUNNING")
            db.update_maintenance_action(run_id, action_id, status=db_status)
            return result(stage, target=plan["target"], plan_sha256=digest,
                          review_status=state.get("review_status"))
    except training.JobBusy:
        return result("BUSY")
    except Exception:
        # Raw paths and provider messages stay out of the result. A durable claim
        # also suppresses retry if persisting this terminal error itself fails.
        if action_checked:
            try:
                if path is not None:
                    with path.with_suffix(".log").open("a", encoding="utf-8") as log:
                        traceback.print_exc(file=log)
                if state is not None and path is not None:
                    state.update(status="FAILED", error_code="SEGMENTATION_DISPATCH_FAILED")
                    save_journal(path, state)
                db.update_maintenance_action(run_id, action_id, status="REJECTED")
            except Exception:
                pass
        return result("FAILED", error_code="SEGMENTATION_DISPATCH_FAILED")
