"""Trusted mechanical controller; no role dispatch, medical policy or self-approval.

Integration contract (all JSON hashes are sorted, compact, ASCII SHA-256):

* ``run_cycle(root, plan, controller=TrustedController(...), execute=False)``
  is read-only by default. root is the project/control root. Registry location
  belongs to the immutable plan. One run_id/action_id owns exactly one plan and
  at most one training launch, across processes and restarts.
* Plan schema ``segmentation-cycle-v1`` has run_id (positive int), action_id,
  owner, owner_role, target, registry_root (absolute), parent_id, policy_id and
  four {path, sha256} bindings: job, architecture, dataset_manifest, cohort.
  No approval flags, callable names, command strings or metric values are fields.
* dataset_manifest is ``dataset_manifest_for_job(job)`` serialized by the
  external producer BEFORE approval. It binds normalized records AND splits.
  cohort is {schema: segmentation-cycle-cohort-v1, target, cases}, where each
  case is {case_id, patient_id, image: binding, reference: binding}. Policy
  benchmark_sha256 is the cohort FILE hash, dataset_sha256 the manifest FILE
  hash. References retain external provenance; this module never declares gold.
  Registration uses the trainer's effective_config.json, verified against the
  approved numeric projection/epoch cap and, for MI, its hashed Myo snapshot.
* authorize reads authenticated stored reviews. ``DatabaseAuthority(db)`` uses
  payload.segmentation_cycle_plan, original_action_approved, and the latest
  independent implementation_review. implementation_sha256 must equal the
  existing governance stable_hash of the EXACT plan (``implementation_sha256``
  below). Activation additionally needs a segmentation_activation_review with
  decision=accept_experiment, plan_sha256, assessment_id, candidate_id, policy_id.
* control is mandatory, read-only, called before launches and during training
  polling. It must check the externally granted budget; it must never arm or
  extend it. Local operator_state/run_budget are also checked fail-closed when
  present. Prediction callbacks MUST call checkpoint before launching and while
  polling, terminate their children on interruption, and return only after exit.
* predict receives verified immutable model artifacts and frozen image bindings,
  never references, and writes only within request.output_dir. It can adapt
  predict_command(..., model_id=request.model_id). No subprocess is launched
  by the runner for prediction without this explicitly supplied trusted adapter.
* evaluate reads independently stored evidence for the computed report; it does
  NOT supply metrics. Evidence binds plan/report, authenticates reference and
  BOTH models' exposure/split provenance. Missing/unknown evidence waits. A
  patient set calculation cannot establish an imported parent's unknown exposure.

Callbacks, DB and runtime/registry directories are trusted controller capabilities,
not authentication for an untrusted HTTP caller. Do not deserialize callbacks or
accept Owner-authored receipts as their return values. Failed/interrupted work is
terminal; an explicit new externally reviewed action is required to retry. A
pause before work is resumable; a pause during training cancels the single attempt.
Missing approvals/evidence are resumable without retraining or re-predicting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from app import segmentation_registry as registry
from app import segmentation_training as training


class CycleError(ValueError):
    pass


class Stage(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PAUSED = "PAUSED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    REGISTERED = "REGISTERED"
    PREDICTING = "PREDICTING"
    AWAITING_EVALUATION = "AWAITING_EVALUATION"
    ASSESSED = "ASSESSED"
    AWAITING_ACTIVATION_APPROVAL = "AWAITING_ACTIVATION_APPROVAL"
    ACTIVATING = "ACTIVATING"
    ACTIVATED = "ACTIVATED"
    RETAIN_PARENT = "RETAIN_PARENT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Control(str, Enum):
    RUN = "RUN"
    PAUSE = "PAUSE"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class CyclePlan:
    """Owner artifact description only; authority is supplied separately.

    Binding objects are {path, sha256}. ``to_dict`` is the exact JSON object
    stored in DB payload.segmentation_cycle_plan and reviewed as implementation.
    run_cycle also accepts this JSON object directly for CLI integration.
    """
    run_id: int
    action_id: str
    owner: str
    owner_role: str
    target: str
    registry_root: str
    parent_id: str
    policy_id: str
    job: dict
    architecture: dict
    dataset_manifest: dict
    cohort: dict
    schema: str = "segmentation-cycle-v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ControlDecision:
    action: Control
    reason: str = ""


@dataclass(frozen=True)
class ControlRequest:
    root: Path
    plan_sha256: str
    run_id: int
    action_id: str
    phase: str


@dataclass(frozen=True)
class ApprovalRequest:
    plan: dict
    plan_sha256: str
    phase: str  # execute | activate
    subject_sha256: str  # plan hash | assessment ID
    candidate_id: str | None = None


@dataclass(frozen=True)
class StoredApproval:
    record_id: str
    actor: str
    plan_sha256: str
    phase: str
    subject_sha256: str


@dataclass(frozen=True)
class PredictionRequest:
    model_id: str
    target: str
    registry_root: Path
    artifacts: dict
    cohort_sha256: str
    cases: list[dict]  # image bindings only; identical for parent and candidate
    output_dir: Path


@dataclass(frozen=True)
class PredictionReceipt:
    model_id: str
    cohort_sha256: str
    predictions: dict  # case_id -> {path, sha256}


@dataclass(frozen=True)
class EvaluationRequest:
    plan: dict
    plan_sha256: str
    candidate_id: str
    report_sha256: str
    report: dict


@dataclass(frozen=True)
class IndependentEvidence:
    record_id: str
    actor: str
    plan_sha256: str
    report_sha256: str
    reference_provenance_id: str
    parent_exposure_evidence_id: str
    split_evidence_id: str
    validation_test_patient_disjoint: bool
    training_evaluation_patient_disjoint: bool


@dataclass(frozen=True)
class TrustedController:
    authorize: Callable[[ApprovalRequest], StoredApproval | None]
    control: Callable[[ControlRequest], ControlDecision]
    predict: Callable[[PredictionRequest, Callable[[], None]], PredictionReceipt] | None = None
    evaluate: Callable[[EvaluationRequest], IndependentEvidence | None] | None = None


def _require(ok, reason):
    if not ok:
        raise CycleError(reason)


def _hash(value):
    return hashlib.sha256(training._canonical(value)).hexdigest()


def plan_sha256(plan: CyclePlan | dict) -> str:
    return _hash(plan.to_dict() if type(plan) is CyclePlan else plan)


def implementation_sha256(plan: CyclePlan | dict) -> str:
    """Existing agents.shadow_analysis.stable_hash format, without provider imports."""
    value = plan.to_dict() if type(plan) is CyclePlan else plan
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


class DatabaseAuthority:
    """Read-only adapter to existing DB governance; never inserts approval messages."""
    def __init__(self, db):
        self.db = db

    def __call__(self, request: ApprovalRequest) -> StoredApproval | None:
        from agents.governance import original_action_approved, EXECUTABLE_ACTION_STATUSES
        p = request.plan
        if (request.plan_sha256 != plan_sha256(p) or
                (request.phase == "execute" and request.subject_sha256 != request.plan_sha256)):
            return None
        try:
            row = self.db.get_maintenance_action(p["run_id"], p["action_id"])
        except KeyError:
            return None
        payload = row["payload"]
        if (row["status"] not in EXECUTABLE_ACTION_STATUSES or row["owner_role"] != p["owner_role"]
                or payload.get("segmentation_cycle_plan") != p
                or not original_action_approved(self.db, p["run_id"], row, payload)):
            return None
        inbox = self.db.agent_inbox(p["run_id"], p["owner_role"],
                                   thread_id=f"run-{p['run_id']}:{p['action_id']}")
        reviews = [m for m in inbox if m["sender_role"] == "independent_reviewer"
                   and m["message_type"] == "implementation_review"]
        if not reviews:
            return None
        review = reviews[-1]
        body = review["payload"]
        if (body.get("decision") != "accept_experiment" or body.get("required_validation_metrics")
                or body.get("implementation_sha256") != implementation_sha256(p)):
            return None
        if request.phase == "activate":
            reviews = [m for m in inbox if m["sender_role"] == "independent_reviewer"
                       and m["message_type"] == "segmentation_activation_review"]
            if not reviews:
                return None
            review = reviews[-1]
            body = review["payload"]
            expected = {"decision": "accept_experiment", "plan_sha256": request.plan_sha256,
                        "assessment_id": request.subject_sha256, "candidate_id": request.candidate_id,
                        "policy_id": p["policy_id"]}
            if any(body.get(k) != v for k, v in expected.items()):
                return None
        elif request.phase != "execute":
            return None
        return StoredApproval("db:agent_messages:" + str(review["message_id"]),
                              review["sender_role"], request.plan_sha256,
                              request.phase, request.subject_sha256)


def dataset_manifest_for_job(job: dict) -> dict:
    """Pure producer helper; caller freezes/reviews this before constructing a plan."""
    normalized = training.validate_job(job)["job"]
    return dict(schema="segmentation-cycle-data-v1", **{
        k: normalized[k] for k in ("target", "dataset_root", "splits", "records")})


def _bound_json(binding):
    checked = training._file(binding)
    data = Path(checked["path"]).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == checked["sha256"], "binding_changed")
    return training._decode_json(data.decode("utf-8"))


def _preflight(plan):
    training._keys(plan, ("schema", "run_id", "action_id", "owner", "owner_role", "target",
                          "registry_root", "parent_id", "policy_id", "job", "architecture",
                          "dataset_manifest", "cohort"))
    _require(plan["schema"] == "segmentation-cycle-v1", "plan_schema")
    _require(type(plan["run_id"]) is int and plan["run_id"] > 0, "run_id")
    for key in ("action_id", "owner", "owner_role"):
        training._text(plan[key])
    _require(plan["owner_role"] not in ("independent_reviewer", "independent_evaluator", "policy_approver"),
             "owner_role_conflict")
    reg = training._absolute(plan["registry_root"])
    parent = registry.resolve_candidate(reg, plan["parent_id"])
    job = training.validate_job(_bound_json(plan["job"]))
    j = job["job"]
    _require(j["target"] == parent["target"] == plan["target"], "target_mismatch")
    _require(_bound_json(plan["dataset_manifest"]) == dataset_manifest_for_job(j),
             "dataset_manifest_must_bind_exact_records_and_splits")
    architecture = training._file(plan["architecture"])
    source = Path(architecture["path"])
    _require(source.is_relative_to(Path(j["vendor"]["root"])) and
             j["vendor"]["files"].get(source.relative_to(j["vendor"]["root"]).as_posix()) == architecture["sha256"],
             "architecture_not_in_approved_vendor")
    if "parent_weights" in j:
        _require(j["parent_weights"]["sha256"] == parent["artifacts"]["weights"]["sha256"], "parent_weights_mismatch")
    cohort = _bound_json(plan["cohort"])
    training._keys(cohort, ("schema", "target", "cases"))
    _require(cohort["schema"] == "segmentation-cycle-cohort-v1" and cohort["target"] == j["target"], "cohort_schema")
    _require(type(cohort["cases"]) is list and cohort["cases"], "empty_cohort")
    seen = set()
    exposed = set(j["splits"]["train"] + j["splits"]["validation"])
    exposed_images = {r["image"]["sha256"] for r in j["records"] if r["patient"] in exposed}
    for row in cohort["cases"]:
        training._keys(row, ("case_id", "patient_id", "image", "reference"))
        training._text(row["case_id"])
        training._text(row["patient_id"])
        _require(row["case_id"] not in seen, "duplicate_cohort_case")
        seen.add(row["case_id"])
        for key in ("image", "reference"):
            training._file(row[key])
        _require(row["patient_id"] not in exposed and row["image"]["sha256"] not in exposed_images,
                 "training_or_selection_cohort_overlap")
    policy = registry._read_record(reg, "policies", plan["policy_id"])
    registry._policy(policy)
    definition = policy["definition"]
    _require(definition["target"] == j["target"] and definition["dataset_sha256"] == plan["dataset_manifest"]["sha256"]
             and definition["benchmark_sha256"] == plan["cohort"]["sha256"], "policy_binding_mismatch")
    _require(policy["approver"]["actor"].casefold() != plan["owner"].casefold(), "owner_cannot_approve_policy")
    _require(set(definition["metrics"]) <= {"dice", "positive_reference_dice", "empty_reference_fp_rate", "coverage"},
             "policy_requires_unavailable_metrics")
    return job, cohort, parent, policy


class _Interrupted(RuntimeError):
    def __init__(self, decision):
        self.decision = decision
        super().__init__(decision.reason or decision.action.value)


def _checkpoint(root, plan, digest, controller, phase):
    operator = root / "runtime/control/operator_state.json"
    budget_path = root / "runtime/control/run_budget.json"
    if operator.exists() and training.read_json(operator).get("paused"):
        raise _Interrupted(ControlDecision(Control.PAUSE, "operator_pause"))
    if budget_path.exists():
        b = training.read_json(budget_path)
        if (b.get("active") is not True or b.get("status") != "RUNNING"
                or time.time() >= b.get("deadline", 0)):
            raise _Interrupted(ControlDecision(Control.PAUSE, "budget_not_running_or_expired"))
    decision = controller.control(ControlRequest(root, digest, plan["run_id"], plan["action_id"], phase))
    _require(type(decision) is ControlDecision and type(decision.action) is Control, "typed_control_decision_required")
    if decision.action != Control.RUN:
        raise _Interrupted(decision)


def _approval(controller, plan, digest, phase, subject, candidate=None):
    approval = controller.authorize(ApprovalRequest(copy.deepcopy(plan), digest, phase, subject, candidate))
    if approval is None:
        return None
    _require(type(approval) is StoredApproval, "stored_approval_required_not_boolean")
    _require(approval.plan_sha256 == digest and approval.phase == phase and approval.subject_sha256 == subject,
             "approval_binding_mismatch")
    training._text(approval.record_id)
    training._text(approval.actor)
    _require(approval.actor.casefold() not in {plan["owner"].casefold(), plan["owner_role"].casefold()},
             "owner_cannot_approve")
    return asdict(approval)


def _train(checked, checkpoint):
    """Monitor existing synchronous run_job; its marker cancels its process tree."""
    run_dir = Path(checked["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(training.run_job, checked["job"], execute=True)
        interrupted = None
        try:
            while True:
                try:
                    result = future.result(timeout=0.1)
                    break
                except TimeoutError:
                    if future.done():
                        result = future.result()
                        break
                    if interrupted is None:
                        try:
                            checkpoint()
                        except BaseException as exc:
                            interrupted = exc
                            (run_dir / "CANCEL").touch(exist_ok=True)
        except BaseException:
            # Also interrupt the worker if the caller is interrupted while
            # waiting on the future, before the executor joins its thread.
            (run_dir / "CANCEL").touch(exist_ok=True)
            raise
        if interrupted:
            raise interrupted
        return result


def _effective_config(checked):
    """Verify the actual trainer artifact, including its pinned MI snapshot."""
    expected = copy.deepcopy(checked["config"])
    job, directory = checked["job"], Path(checked["run_dir"])
    if job["target"] == "lge_mi":
        dependency = training._file({"path": str(directory / "inputs/myo_weights" /
                                                Path(job["myo_weights"]["path"]).name),
                                     "sha256": job["myo_weights"]["sha256"]})
        expected["myocardium_model"] = {"weights": dependency["path"], "sha256": dependency["sha256"]}
    # The trainer writes canonical JSON. Exact bytes also reject hidden/private
    # config sections, unapproved epochs and a substituted Myo dependency path.
    return training._file({"path": str(directory / "effective_config.json"), "sha256": _hash(expected)})


def _training_record(plan, checked, receipt):
    j = checked["job"]
    _require(receipt.get("status") == "TRAINED" and receipt.get("job_sha256") == checked["job_sha256"], "training_receipt_mismatch")
    for key, expected in (("vendor_sources_sha256", _hash(j["vendor"]["files"])),
                          ("config_sha256", j["config"]["sha256"]),
                          ("projected_config_sha256", _hash(checked["config"])),
                          ("parent_weights_sha256", j.get("parent_weights", {}).get("sha256")),
                          ("myo_weights_sha256", j.get("myo_weights", {}).get("sha256"))):
        _require(receipt.get(key) == expected, "training_receipt_mismatch:" + key)
    weights = training._file({"path": receipt["checkpoint"], "sha256": receipt["checkpoint_sha256"]})
    _require(Path(weights["path"]).is_relative_to(Path(checked["run_dir"]) / "artifacts"), "checkpoint_outside_attempt")
    effective_config = _effective_config(checked)
    hashes = {"weights": weights["sha256"], "config": effective_config["sha256"],
              "architecture": plan["architecture"]["sha256"], "dataset_manifest": plan["dataset_manifest"]["sha256"]}
    return {"status": "SUCCEEDED", "training_source": "fine_tuning" if "parent_weights" in j else "training",
            "owner_role": plan["owner_role"], "attempt_id": checked["job_sha256"], "metrics": {},
            "artifact_hashes": hashes, "vendor_sources_sha256": receipt["vendor_sources_sha256"],
            "job_sha256": checked["job_sha256"], "plan_sha256": plan_sha256(plan),
            "receipt": copy.deepcopy(receipt)}


def _predictions(receipt, request):
    _require(type(receipt) is PredictionReceipt, "typed_prediction_receipt_required")
    _require(receipt.model_id == request.model_id and receipt.cohort_sha256 == request.cohort_sha256,
             "prediction_model_or_cohort_mismatch")
    _require(type(receipt.predictions) is dict and set(receipt.predictions) == {r["case_id"] for r in request.cases},
             "incomplete_prediction_cohort")
    paths = set()
    for binding in receipt.predictions.values():
        path = Path(training._file(binding)["path"])
        _require(path.is_relative_to(request.output_dir) and path not in paths, "prediction_outside_output_or_duplicate")
        paths.add(path)
    return asdict(receipt)


def run_cycle(root: Path, plan: CyclePlan | dict, *, controller: TrustedController, execute: bool = False) -> dict:
    """Return persisted typed stage/latest state; validation errors raise before work.

    State: root/runtime/segmentation_cycles/<hash(run_id,action_id)>/latest.json.
    Each transition retains a numbered snapshot in stages/. Call again to resume
    waiting stages. FAILED/CANCELLED/RETAIN_PARENT/ACTIVATED never launch again.
    Runtime and callback payloads are private; expose only aggregates in the UI.
    runtime/segmentation_training/latest.json is a separate aggregate projection:
    status/stage, target, job_id, elapsed_seconds and a generic error_code only.
    """
    _require(type(execute) is bool and type(controller) is TrustedController, "trusted_controller_required")
    _require(callable(controller.authorize) and callable(controller.control), "authority_and_control_required")
    plan = plan.to_dict() if type(plan) is CyclePlan else plan
    plan = training._decode_json(training._canonical(plan).decode("utf-8"))
    root = training._absolute(str(root))
    checked, cohort, parent, policy = _preflight(plan)
    digest = plan_sha256(plan)
    folder = root / "runtime/segmentation_cycles" / _hash({k: plan[k] for k in ("run_id", "action_id")})
    protected = [Path(checked["job"][key]) for key in ("dataset_root", "output_dir")]
    protected += [Path(checked["job"]["vendor"]["root"]), Path(plan["registry_root"])]
    protected += [Path(row[key]["path"]) for row in cohort["cases"] for key in ("image", "reference")]
    protected += [Path(plan[key]["path"]) for key in ("job", "architecture", "dataset_manifest", "cohort")]
    protected += [Path(checked["job"][key]["path"]) for key in ("config", "parent_weights", "myo_weights")
                  if key in checked["job"]]
    _require(all(not folder.is_relative_to(p) and not p.is_relative_to(folder) for p in protected),
             "cycle_outputs_overlap_bound_inputs_or_other_stores")
    latest = training._absolute(str(folder / "latest.json"))
    state = {"schema": "segmentation-cycle-state-v1", "plan_sha256": digest, "stage": Stage.PREFLIGHT.value,
             "run_id": plan["run_id"], "action_id": plan["action_id"], "job_sha256": checked["job_sha256"],
             "sequence": 0, "training_started": False, "prediction_started": [], "predictions": {},
             "candidate_id": None, "missing_evidence": []}
    if latest.exists():
        state = training.read_json(latest)
        _require(state["plan_sha256"] == digest, "action_already_bound_to_another_plan")
        Stage(state["stage"])
    if not execute:
        return dict(state, execute=False)
    folder.mkdir(parents=True, exist_ok=True)
    with training._job_lock(folder / ".lock"):
        if latest.exists():
            state = training.read_json(latest)
            _require(state["plan_sha256"] == digest, "action_already_bound_to_another_plan")
            Stage(state["stage"])
        def save(stage, **updates):
            state.update(updates, stage=stage.value, sequence=state["sequence"] + 1)
            stages = training._absolute(str(folder / "stages"))
            stages.mkdir(exist_ok=True)
            training._atomic_json(stages / (str(state["sequence"]).zfill(6) + ".json"), state)
            training._atomic_json(latest, state)
            board_path = training._absolute(str(root / "runtime/segmentation_training/latest.json"))
            board_path.parent.mkdir(parents=True, exist_ok=True)
            # Whitelist only aggregate fields: exception text, action IDs,
            # artifact paths, patient IDs and model receipts stay private.
            summary = {"schema": "segmentation-cycle-summary-v1", "status": stage.value,
                       "stage": stage.value, "target": plan["target"], "job_id": checked["job_sha256"],
                       "elapsed_seconds": max(0.0, time.time() - state.get("started_at", time.time()))}
            if stage in (Stage.FAILED, Stage.CANCELLED):
                summary["error_code"] = "cycle_failed" if stage == Stage.FAILED else "cycle_cancelled"
            training._atomic_json(board_path, summary)
            return copy.deepcopy(state)
        if state["stage"] in {s.value for s in (Stage.FAILED, Stage.CANCELLED, Stage.RETAIN_PARENT, Stage.ACTIVATED)}:
            return state
        work_in_progress = False
        state.setdefault("started_at", time.time())
        reg = Path(plan["registry_root"])
        def checkpoint(phase):
            _checkpoint(root, plan, digest, controller, phase)
        try:
            checkpoint("resume")
            approval = _approval(controller, plan, digest, "execute", digest)
            if approval is None:
                return save(Stage.AWAITING_APPROVAL, missing_evidence=["stored_exact_plan_approval"])
            state["execution_approval"] = approval
            if not state.get("training"):
                receipt_path = Path(checked["run_dir"]) / "status.json"
                if state["training_started"]:
                    # Never call run_job again after an uncertain launch, even if
                    # no receipt exists. A completed receipt can be reconciled.
                    _require(receipt_path.exists(), "interrupted_training_launch_no_retry")
                    receipt = training.read_json(receipt_path)
                    _require(receipt.get("status") != "TRAINING", "interrupted_training_no_retry")
                else:
                    checkpoint("training_launch")
                    save(Stage.TRAINING, training_started=True, missing_evidence=[])
                    work_in_progress = True
                    last_progress = time.monotonic()
                    def poll_training():
                        nonlocal last_progress
                        checkpoint("training_poll")
                        if time.monotonic() - last_progress >= 1:
                            save(Stage.TRAINING)
                            last_progress = time.monotonic()
                    receipt = _train(checked, poll_training)
                    work_in_progress = False
                if receipt.get("status") != "TRAINED":
                    return save(Stage.CANCELLED if receipt.get("status") == "CANCELLED" else Stage.FAILED,
                                training=receipt, reason=receipt.get("reason", "training_failed"))
                _training_record(plan, checked, receipt)
                save(Stage.TRAINED, training=receipt)
            if not state["candidate_id"]:
                checkpoint("register")
                _preflight(plan)  # recheck approved bytes after training
                record = _training_record(plan, checked, state["training"])
                candidate = registry.register_candidate(reg, target=plan["target"],
                    weights=Path(state["training"]["checkpoint"]), config=Path(_effective_config(checked)["path"]),
                    architecture=Path(plan["architecture"]["path"]), dataset_manifest=Path(plan["dataset_manifest"]["path"]),
                    parent_id=plan["parent_id"], owner=plan["owner"], training_record=record)
                save(Stage.REGISTERED, candidate_id=candidate["id"])
            candidate = registry.resolve_candidate(reg, state["candidate_id"])
            if controller.predict is None:
                return save(Stage.AWAITING_EVALUATION, missing_evidence=["trusted_prediction_adapter"])
            images = [{k: copy.deepcopy(row[k]) for k in ("patient_id", "case_id", "image")} for row in cohort["cases"]]
            for role, model in (("parent", parent), ("candidate", candidate)):
                request = PredictionRequest(model["candidate_id"], plan["target"], reg, copy.deepcopy(model["artifacts"]),
                    plan["cohort"]["sha256"], copy.deepcopy(images), folder / (role + "_predictions"))
                if role not in state["predictions"]:
                    _require(role not in state["prediction_started"], "interrupted_prediction_no_retry:" + role)
                    checkpoint(role + "_prediction_launch")
                    save(Stage.PREDICTING, prediction_started=state["prediction_started"] + [role])
                    work_in_progress = True
                    receipt = controller.predict(copy.deepcopy(request), lambda: checkpoint(role + "_prediction_poll"))
                    state["predictions"][role] = _predictions(receipt, request)
                    work_in_progress = False
                    save(Stage.PREDICTING)
                else:
                    _predictions(PredictionReceipt(**state["predictions"][role]), request)
            checkpoint("compare")
            _preflight(plan)
            from app.segmentation_evaluation import compare_predictions
            cases = [dict(patient_id=row["patient_id"], case_id=row["case_id"], reference=row["reference"],
                          **{role: state["predictions"][role]["predictions"][row["case_id"]]
                             for role in ("parent", "candidate")}) for row in cohort["cases"]]
            report = compare_predictions(cases, target=plan["target"])
            report_hash = _hash(report)
            save(Stage.AWAITING_EVALUATION, report=report, report_sha256=report_hash,
                 missing_evidence=["independent_reference_and_exposure_evidence"])
            if controller.evaluate is None:
                return state
            evidence = controller.evaluate(EvaluationRequest(copy.deepcopy(plan), digest, candidate["candidate_id"],
                                                             report_hash, copy.deepcopy(report)))
            if evidence is None:
                return state
            _require(type(evidence) is IndependentEvidence, "typed_independent_evidence_required")
            _require(evidence.plan_sha256 == digest and evidence.report_sha256 == report_hash, "evidence_binding_mismatch")
            training._text(evidence.actor)
            _require(evidence.actor.casefold() not in {plan["owner"].casefold(), plan["owner_role"].casefold()}, "evaluator_not_independent")
            fields = ("record_id", "reference_provenance_id", "parent_exposure_evidence_id", "split_evidence_id")
            missing = [f for f in fields if not isinstance(getattr(evidence, f), str) or not getattr(evidence, f).strip()]
            missing += [f for f in ("validation_test_patient_disjoint", "training_evaluation_patient_disjoint")
                        if getattr(evidence, f) is not True]
            if missing:
                return save(Stage.AWAITING_EVALUATION, missing_evidence=missing)
            names = policy["definition"]["metrics"]
            absent = [name for name in names if report["metrics"][name]["candidate_count"] == 0]
            if absent:
                return save(Stage.AWAITING_EVALUATION, missing_evidence=["metric_stratum:" + n for n in absent])
            comparison = {"candidate_id": candidate["candidate_id"], "parent_id": parent["candidate_id"],
                "target": plan["target"], "dataset_sha256": plan["dataset_manifest"]["sha256"],
                "benchmark_sha256": plan["cohort"]["sha256"],
                "candidate_artifact_hashes": {k: v["sha256"] for k, v in candidate["artifacts"].items()},
                "parent_artifact_hashes": {k: v["sha256"] for k, v in parent["artifacts"].items()},
                "validation_test_patient_disjoint": evidence.validation_test_patient_disjoint,
                "training_evaluation_patient_disjoint": evidence.training_evaluation_patient_disjoint,
                "evaluator": {"actor": evidence.actor, "role": "independent_evaluator", "status": "SUCCEEDED"},
                "measurement": {k: report[k] for k in ("source", "self_reported", "candidate_predictions_sha256",
                                                       "parent_predictions_sha256", "reference_sha256")},
                "metrics": {plan["target"]: {k: report["metrics"][k] for k in names}}}
            checkpoint("assessment")
            _require(_hash(compare_predictions(cases, target=plan["target"])) == report_hash,
                     "measurement_changed_during_evidence_lookup")
            assessment = registry.record_assessment(reg, candidate["candidate_id"], comparison, policy_id=plan["policy_id"])
            save(Stage.ASSESSED, assessment=assessment, independent_evidence=asdict(evidence), missing_evidence=[])
            if not assessment["passed"]:
                return save(Stage.RETAIN_PARENT, reason="assessment_failed")
            approval = _approval(controller, plan, digest, "activate", assessment["assessment_id"], candidate["candidate_id"])
            if approval is None:
                return save(Stage.AWAITING_ACTIVATION_APPROVAL, missing_evidence=["stored_assessment_activation_approval"])
            checkpoint("activate")
            _preflight(plan)
            _require(_hash(compare_predictions(cases, target=plan["target"])) == report_hash,
                     "measurement_changed_during_approval_lookup")
            active = registry.resolve_active(reg, plan["target"])
            if (active and active["candidate_id"] == candidate["candidate_id"] and
                    active["selection"]["assessment_id"] == assessment["assessment_id"]):
                return save(Stage.ACTIVATED, selection=active["selection"], activation_approval=approval)
            save(Stage.ACTIVATING, activation_approval=approval)
            selection = registry.activate(reg, candidate["candidate_id"],
                                          {"assessment_id": assessment["assessment_id"]}, plan["parent_id"])
            return save(Stage.ACTIVATED, selection=selection)
        except _Interrupted as exc:
            terminal = work_in_progress or exc.decision.action == Control.CANCEL
            return save(Stage.CANCELLED if terminal else Stage.PAUSED, reason=str(exc))
        except Exception as exc:
            return save(Stage.FAILED, reason=type(exc).__name__ + ": " + str(exc))
