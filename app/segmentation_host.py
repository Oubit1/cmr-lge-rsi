"""Concrete trusted host capabilities for the mechanical segmentation cycle.

``make_controller(root, plan, database=path)`` connects existing DB governance,
read-only global budget checks, immutable-model prediction and stored evidence.
It dispatches no roles, inserts no messages, grants no budget and writes no labels.
The host/DB/runtime must be inaccessible to Owner code for mutation. DB sender
roles are the existing governance trust boundary, not cryptographic signatures.

Prediction reuses scripts.segment_from_registry.predict_command(model_id=...).
Only hashed image copies enter its input folder, under stable 96-bit SHA-256 case
names (collisions are rejected). Short names keep Windows runtime paths bounded.
Each model gets its own verified source bundle under registry/source_bundles,
bound once to its model ID, all artifact hashes and full Python source hash.
freeze_model_sources(registry, id, vendor_root=...) mechanically freezes an
existing baseline's actual vendor tree; it remains BASELINE_UNMEASURED. This is
source identity, not clinical approval, and requires no per-plan DB attestation.
The CLI's --baseline-vendor supplies that target directory once; preflight only
verifies it, execution snapshots it. Later plans reuse the immutable bundle.
Trained models can also resolve their prior run_job receipt's exact job-hash run
directory and vendor snapshot, verified against stored weights/config/source
hashes, then archive it once. No parent code is inferred from the current job.
make_controller checks parent availability before training; SourceUnavailable
means AWAITING_MODEL_SOURCES in the CLI, not a failed training attempt. Parent and
candidate architectures/configs can differ when each has its matching bundle.
The supported vendor layout is masks/<case>.nii[.gz]; CMR's older predictions/ layout
is also accepted, but both folders together are ambiguous and rejected.
Missing, ambiguous, extra or nested mask outputs fail closed; no filename
guessing, thresholding, reference copying or label installation is performed.
Workers start behind a gate, inside a Windows Job Object or POSIX process group;
all descendants are stopped on exit, timeout or checkpoint interruption. This is
execution of trusted reviewed local code, not a same-UID security sandbox.

DB evidence contract, in thread run-{run_id}:{action_id}, recipient owner_role:
only independent_evaluator/independent_reviewer sender roles are accepted, and
neither may equal the plan Owner identity/role. The latest message of each type
wins; an older verified record cannot override a later rejection/unknown record.

segmentation_evaluation_evidence payload:
  schema: segmentation-evaluation-evidence-v1; status: VERIFIED;
  plan_sha256, report_sha256, candidate_id, parent_id, policy_id, target;
  reference_provenance_message_id, parent_exposure_message_id,
  split_evidence_message_id (positive integer DB message IDs).

The referenced independent messages must be the latest of their respective
types in that same thread, with status=VERIFIED and the exact plan_sha256:
  segmentation_reference_provenance: cohort_sha256, reference_sha256
  segmentation_parent_exposure: parent_id, cohort_sha256,
      training_evaluation_patient_disjoint: ESTABLISHED
  segmentation_split_evidence: dataset_sha256, cohort_sha256,
      validation_test_patient_disjoint: ESTABLISHED,
      training_evaluation_patient_disjoint: ESTABLISHED

These are attestations from an independently authenticated workflow, not facts
inferred by this host from a filename or a baseline import. UNKNOWN, missing or
Owner-authored evidence yields WAITING, never a manufactured disjointness flag.
Reference provenance does not become doctor gold through this interface.
Actual metrics always come from segmentation_evaluation.compare_predictions.
Activation separately requires the cycle's segmentation_activation_review.

Comparison and promotion are distinct. Aggregate paired metrics are available
before this evidence gate, including for repeated validation used in checkpoint
selection by the parent, provided the cohort passes the cycle's preflight. That
preflight currently rejects overlap with the candidate job's train OR validation
patients/images. Standalone compare_predictions can measure such repeated
validation, but this cycle does not yet represent that selection-only scope.
A known train/validation exposure history is not an untouched final
test. The current registry requires both literal disjointness flags and has no
validation-only promotion_scope; absence of a final-test partition must remain
unestablished, not be converted to a vacuously true flag. This host does not add
or weaken that policy. The CLI exposes comparison_status/metrics even while
evidence_status is WAITING, and makes no untouched-test/generalization claim.

Global controls are read on each checkpoint. A missing/malformed/inactive budget,
expired deadline, exhausted allowance or operator pause blocks work. The host
never calls reserve/arm/resume, spends provider allowances, or extends deadlines.
Prediction walltime is at most the approved job max_seconds, additionally bounded
by the current global deadline. A lower host cap can be supplied by deployment.
"""
from __future__ import annotations

import copy
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app import segmentation_cycle as cycle
from app import segmentation_registry as registry
from app import segmentation_training as training
from app.database import Database
from scripts.segment_from_registry import predict_command


class HostError(ValueError):
    """A deterministic host contract failure; contains no worker log contents."""


class SourceUnavailable(HostError):
    """A verified per-model source bundle must be supplied by the host reviewer."""


def _require(ok, code):
    if not ok:
        raise HostError(code)


class ReadOnlyDatabase(Database):
    """No logical DB writes/migrations. SQLite may create WAL coordination sidecars.

    Do not use immutable=1: that would ignore live WAL reviews/revocations.
    """
    def __init__(self, path: Path):
        self.path = training._absolute(str(path))
        _require(self.path.is_file(), "governance_database_missing")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            yield connection
        finally:
            connection.close()


class GlobalControl:
    def __init__(self, root: Path, plan: dict):
        self.root = training._absolute(str(root))
        self.plan = copy.deepcopy(plan)

    def __call__(self, request: cycle.ControlRequest) -> cycle.ControlDecision:
        pause = lambda code: cycle.ControlDecision(cycle.Control.PAUSE, code)
        if (request.root != self.root or request.plan_sha256 != cycle.plan_sha256(self.plan)
                or request.run_id != self.plan["run_id"] or request.action_id != self.plan["action_id"]):
            return pause("host_control_binding_mismatch")
        try:
            operator = self.root / "runtime/control/operator_state.json"
            if operator.exists():
                value = training.read_json(operator)
                if type(value) is not dict or type(value.get("paused")) is not bool:
                    return pause("operator_state_invalid")
                if value["paused"]:
                    return pause("operator_pause")
            budget = training.read_json(self.root / "runtime/control/run_budget.json")
            if type(budget) is not dict or budget.get("active") is not True or budget.get("status") != "RUNNING":
                return pause("budget_not_running")
            deadline = budget.get("deadline")
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.time() >= deadline:
                return pause("budget_deadline")
            for kind in ("analyses", "requests"):
                used, limit = budget.get("used", {}).get(kind), budget.get("limits", {}).get(kind)
                if type(used) is not int or type(limit) is not int or used < 0 or limit <= 0:
                    return pause("budget_counters_invalid")
                if used >= limit:
                    return pause("budget_" + kind + "_exhausted")
        except (OSError, ValueError, TypeError, AttributeError):
            return pause("budget_or_operator_state_unreadable")
        return cycle.ControlDecision(cycle.Control.RUN)


@dataclass(frozen=True)
class EvidenceLookup:
    status: str
    missing_evidence: tuple[str, ...] = ()
    evidence: cycle.IndependentEvidence | None = None


class DatabaseEvidence:
    """Read independently stored provenance; never accept metric or approval input."""
    ROLES = frozenset({"independent_evaluator", "independent_reviewer"})

    def __init__(self, db: ReadOnlyDatabase):
        self.db = db
        self.last_lookup = EvidenceLookup("NOT_CHECKED")

    def inspect(self, request: cycle.EvaluationRequest) -> EvidenceLookup:
        p = request.plan
        wait = lambda code: EvidenceLookup("WAITING", (code,))
        if request.plan_sha256 != cycle.plan_sha256(p) or request.report_sha256 != cycle._hash(request.report):
            return wait("evaluation_request_binding_mismatch")
        try:
            row = self.db.get_maintenance_action(p["run_id"], p["action_id"])
            if row["owner_role"] != p["owner_role"] or row["payload"].get("segmentation_cycle_plan") != p:
                return wait("stored_plan_mismatch")
            messages = self.db.agent_inbox(p["run_id"], p["owner_role"],
                                          thread_id=f"run-{p['run_id']}:{p['action_id']}")
        except (KeyError, OSError, ValueError, sqlite3.Error):
            return wait("independent_evidence_store_unavailable")
        messages = [m for m in messages if m["sender_role"] in self.ROLES and
                    m["sender_role"].casefold() not in {p["owner"].casefold(), p["owner_role"].casefold()}]
        def latest(kind):
            rows = [m for m in messages if m["message_type"] == kind]
            return rows[-1] if rows else None
        receipt = latest("segmentation_evaluation_evidence")
        if receipt is None:
            return wait("independent_evaluation_message_missing")
        body = receipt["payload"]
        expected = {"schema": "segmentation-evaluation-evidence-v1", "status": "VERIFIED",
                    "plan_sha256": request.plan_sha256, "report_sha256": request.report_sha256,
                    "candidate_id": request.candidate_id, "parent_id": p["parent_id"],
                    "policy_id": p["policy_id"], "target": p["target"]}
        if type(body) is not dict or any(body.get(k) != v for k, v in expected.items()):
            return wait("independent_evaluation_not_verified_for_exact_report")
        references = (
            ("reference_provenance", {"cohort_sha256": p["cohort"]["sha256"],
                                      "reference_sha256": request.report["reference_sha256"]}),
            ("parent_exposure", {"parent_id": p["parent_id"], "cohort_sha256": p["cohort"]["sha256"],
                                  "training_evaluation_patient_disjoint": "ESTABLISHED"}),
            ("split_evidence", {"dataset_sha256": p["dataset_manifest"]["sha256"],
                                 "cohort_sha256": p["cohort"]["sha256"],
                                 "validation_test_patient_disjoint": "ESTABLISHED",
                                 "training_evaluation_patient_disjoint": "ESTABLISHED"}),
        )
        ids = {}
        for name, fields in references:
            identifier = body.get(name + "_message_id")
            message = latest("segmentation_" + name)
            if type(identifier) is not int or identifier <= 0 or not message or message["message_id"] != identifier:
                return wait(name + "_independent_record_missing_or_superseded")
            payload = message["payload"]
            fields = dict(fields, status="VERIFIED", plan_sha256=request.plan_sha256)
            if isinstance(payload, dict):
                for flag in ("validation_test_patient_disjoint", "training_evaluation_patient_disjoint"):
                    if fields.get(flag) == "ESTABLISHED" and payload.get(flag) != "ESTABLISHED":
                        # Known exposure can be non-disjoint. Report the unmet
                        # disjointness requirement, not "unknown exposure".
                        return wait(("parent_" if name == "parent_exposure" else "") + flag + "_not_established")
            if type(payload) is not dict or any(payload.get(k) != v for k, v in fields.items()):
                return wait(name + "_not_established")
            ids[name] = "db:agent_messages:" + str(identifier)
        evidence = cycle.IndependentEvidence("db:agent_messages:" + str(receipt["message_id"]),
            receipt["sender_role"], request.plan_sha256, request.report_sha256,
            ids["reference_provenance"], ids["parent_exposure"], ids["split_evidence"], True, True)
        return EvidenceLookup("EVIDENCE_VERIFIED", evidence=evidence)

    def __call__(self, request: cycle.EvaluationRequest) -> cycle.IndependentEvidence | None:
        self.last_lookup = self.inspect(request)
        return self.last_lookup.evidence


def _source_files(root):
    root = training._absolute(str(root))
    _require(root.is_dir(), "model_source_directory_missing")
    files = {}
    for path in root.rglob("*"):
        training._absolute(str(path))
        if path.is_file() and path.suffix == ".py":
            files[path.relative_to(root).as_posix()] = training.sha256_file(path)
    return files


def _verify_sources(model, root, files, expected):
    root = training._absolute(str(root))
    training._sha(expected)
    _require(type(files) is dict and files and cycle._hash(files) == expected, "model_source_manifest_mismatch")
    if model["kind"] == "trained":
        _require(expected == model["training_record"].get("vendor_sources_sha256"), "trained_model_source_binding_mismatch")
    _require(_source_files(root) == files, "model_source_bundle_changed")
    model_file = {"cmr_myo": "cine_seg/model.py", "lge_myo": "lge_seg/model.py",
                  "lge_mi": "lge_mi_prior_seg/model.py"}[model["target"]]
    _require(files.get(model_file) == model["artifacts"]["architecture"]["sha256"], "model_source_architecture_mismatch")
    _require(training.VENDORS[model["target"]][1] in files, "model_source_entrypoint_missing")
    return {"root": root, "files": copy.deepcopy(files), "sha256": expected}


def _catalog_sources(registry_root, model):
    folder = registry._path(Path(registry_root), "source_bundles", model["candidate_id"])
    if not folder.exists():
        return None
    body = registry._load(registry._path(folder, "bundle.json"))
    registry._check_record(body)
    _require(body.get("schema") == "segmentation-source-bundle-v1" and body.get("model_id") == model["candidate_id"]
             and body.get("status") == ("BASELINE_UNMEASURED" if model["kind"] == "baseline" else "TRAINED_SOURCE_BOUND")
             and body.get("artifact_hashes") == {k: v["sha256"] for k, v in model["artifacts"].items()},
             "source_catalog_model_binding_mismatch")
    return _verify_sources(model, folder / "vendor", body["files"], body["vendor_sources_sha256"])


def freeze_model_sources(registry_root: Path, model_id: str, *, vendor_root: Path) -> dict:
    """CONTROL ONLY: once-per-model source identity snapshot; no medical approval.

    The caller supplies the existing baseline's actual target vendor directory.
    Architecture/config/weights remain the verified registry binding. A baseline
    stays BASELINE_UNMEASURED; this adds no training exposure or accuracy claim.
    Trained models additionally require the recorded full vendor-manifest hash.
    """
    root = training._absolute(str(registry_root))
    model = registry.resolve_candidate(root, model_id)
    files = _source_files(vendor_root)
    source = _verify_sources(model, vendor_root, files, cycle._hash(files))
    body = registry._record({"schema": "segmentation-source-bundle-v1", "model_id": model_id,
        "status": "BASELINE_UNMEASURED" if model["kind"] == "baseline" else "TRAINED_SOURCE_BOUND",
        "artifact_hashes": {k: v["sha256"] for k, v in model["artifacts"].items()},
        "files": files, "vendor_sources_sha256": source["sha256"]})
    with registry._locked(root):
        existing = _catalog_sources(root, model)
        if existing is not None:
            _require(existing["sha256"] == source["sha256"], "model_source_catalog_is_immutable")
            return existing
        collection = registry._path(root, "source_bundles")
        collection.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sources-", dir=str(root)) as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir()
            for relative, sha in files.items():
                training._snapshot({"path": str(source["root"] / relative), "sha256": sha}, stage / "vendor" / relative)
            _verify_sources(model, stage / "vendor", files, source["sha256"])
            _require(registry.resolve_candidate(root, model_id) == model, "model_changed_during_source_freeze")
            registry._atomic_json(stage / "bundle.json", body, immutable=True)
            for path in (stage / "vendor").rglob("*.py"):
                path.chmod(0o444)
            os.rename(str(stage), str(collection / model_id))
        return _catalog_sources(root, model)


class ModelSources:
    """Resolve each model independently; no per-plan or clinical source approval."""
    def __init__(self, registry_root, *, baseline_id, baseline_vendor=None):
        self.registry_root = Path(registry_root)
        self.baseline_id, self.baseline_vendor = baseline_id, baseline_vendor

    def resolve(self, model):
        archived = _catalog_sources(self.registry_root, model)
        if archived is not None:
            return archived
        if model["kind"] == "baseline":
            if model["candidate_id"] != self.baseline_id or self.baseline_vendor is None:
                raise SourceUnavailable("baseline_vendor_bundle_missing")
            files = _source_files(self.baseline_vendor)
            return _verify_sources(model, self.baseline_vendor, files, cycle._hash(files))
        record = model["training_record"]
        receipt = record.get("receipt")
        if not isinstance(receipt, dict) or not record.get("job_sha256"):
            raise SourceUnavailable("trained_source_receipt_missing")
        job_id = training._sha(record["job_sha256"])
        checkpoint = training._absolute(receipt.get("checkpoint"))
        # This is the run_job storage contract, not a search for similar models:
        # output/<exact job SHA>/artifacts/<checkpoint>, with vendor beside it.
        runs = [p for p in checkpoint.parents if p.name == job_id and checkpoint.is_relative_to(p / "artifacts")]
        _require(len(runs) == 1 and receipt.get("status") == "TRAINED" and receipt.get("job_sha256") == job_id
                 and receipt.get("checkpoint_sha256") == model["artifacts"]["weights"]["sha256"], "trained_source_receipt_mismatch")
        run = runs[0]
        _require(training.read_json(run / "status.json") == receipt, "training_receipt_changed")
        training._file({"path": str(run / "effective_config.json"), "sha256": model["artifacts"]["config"]["sha256"]})
        expected = record.get("vendor_sources_sha256")
        _require(receipt.get("vendor_sources_sha256") == expected, "trained_source_manifest_mismatch")
        files = _source_files(run / "vendor")
        return _verify_sources(model, run / "vendor", files, expected)


def _environment():
    # Do not forward provider tokens or arbitrary Python import/startup settings.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
               "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LD_LIBRARY_PATH",
               "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
    result = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def _run_process(command, directory, checkpoint, max_seconds):
    """Start gated so Windows descendants cannot escape before Job assignment."""
    bootstrap = ('import os,runpy,sys; gate=sys.stdin.buffer.readline(); '
                 'gate == b"start\\n" or sys.exit(125); sys.argv=sys.argv[1:]; '
                 'sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); '
                 'runpy.run_path(sys.argv[0],run_name="__main__")')
    process, group = None, None
    log_path = directory / "worker.log"
    checkpoint()
    started = time.monotonic()
    try:
        if os.name == "nt":
            group = training._WindowsJob()
        with log_path.open("xb") as log:
            process = subprocess.Popen([sys.executable, "-B", "-c", bootstrap, *command[1:]],
                stdin=subprocess.PIPE, stdout=log, stderr=log, cwd=str(directory), env=_environment(),
                shell=False, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if group is not None:
                group.attach(process)
            checkpoint()
            process.stdin.write(b"start\n")
            process.stdin.flush()
            while True:
                checkpoint()
                _require(time.monotonic() - started < max_seconds, "prediction_deadline_exceeded")
                _require(log_path.stat().st_size <= 16 * 1024 * 1024, "prediction_log_limit")
                code = process.poll()
                if code is not None:
                    _require(code == 0, "prediction_worker_failed")
                    break
                time.sleep(0.05)
    finally:
        training._stop_group(process, group)


class RegistryPredictor:
    """Fixed vendor adapter, not a configurable command/LLM-generated executable."""
    MASK_FOLDERS = {"cmr_myo": ("masks", "predictions"), "lge_myo": ("masks",), "lge_mi": ("masks",)}

    def __init__(self, plan: dict, *, baseline_vendor: Path | None = None, max_seconds: float | None = None):
        self.plan = copy.deepcopy(plan)
        self.checked = training.validate_job(cycle._bound_json(plan["job"]))
        self.sources = ModelSources(plan["registry_root"], baseline_id=plan["parent_id"], baseline_vendor=baseline_vendor)
        seconds = self.checked["job"]["resources"]["max_seconds"]
        if max_seconds is not None:
            _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and max_seconds > 0,
                     "invalid_prediction_walltime")
            seconds = min(seconds, max_seconds)
        self.max_seconds = seconds

    def __call__(self, request: cycle.PredictionRequest, checkpoint) -> cycle.PredictionReceipt:
        checkpoint()
        p = self.plan
        _require(request.registry_root == Path(p["registry_root"]) and request.target == p["target"]
                 and request.cohort_sha256 == p["cohort"]["sha256"], "prediction_request_binding_mismatch")
        # Reload the approved cohort, never pass its references to the worker.
        cohort = cycle._bound_json(p["cohort"])
        images = [{k: row[k] for k in ("case_id", "patient_id", "image")} for row in cohort["cases"]]
        _require(request.cases == images, "prediction_request_cohort_mismatch")
        model = registry.resolve_candidate(request.registry_root, request.model_id)
        if request.model_id != p["parent_id"]:
            _require(model["parent_id"] == p["parent_id"] and
                     model["training_record"].get("plan_sha256") == cycle.plan_sha256(p), "candidate_plan_mismatch")
        _require(request.artifacts == model["artifacts"], "prediction_artifacts_mismatch")
        source_bundle = self.sources.resolve(model)
        # Persist source identity once, so subsequent plans need neither the live
        # vendor tree nor fresh metadata/attestation for an already bound model.
        checkpoint()
        source_bundle = freeze_model_sources(request.registry_root, request.model_id, vendor_root=source_bundle["root"])
        job = training.validate_job(cycle._bound_json(p["job"]))["job"]
        _require(job == self.checked["job"], "approved_job_changed")
        output = training._absolute(str(request.output_dir))
        protected = [Path(job["dataset_root"]), source_bundle["root"], request.registry_root]
        protected += [Path(row["image"]["path"]) for row in images]
        _require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected),
                 "prediction_output_overlaps_sources")
        output.mkdir(parents=True, exist_ok=False)
        snapshots, mapping = [], {}
        for row in sorted(images, key=lambda r: r["case_id"]):
            checkpoint()
            name = str(row["image"]["path"])
            _require(name.endswith((".nii", ".nii.gz")), "prediction_image_requires_nifti")
            token = cycle._hash({"case_id": row["case_id"]})[:24]
            _require(token not in mapping, "case_name_collision")
            suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
            snapshots.append(training._snapshot(row["image"], output / "inputs" / (token + suffix)))
            mapping[token] = row["case_id"]
        vendor = output / "e"
        folder = training.VENDORS[request.target][0]
        for relative, digest in source_bundle["files"].items():
            checkpoint()
            snapshots.append(training._snapshot({"path": str(source_bundle["root"] / relative), "sha256": digest},
                                                 vendor / folder / relative))
        predictions = output / "outputs"
        command, receipt = predict_command(request.registry_root, vendor, request.target,
            output / "inputs", predictions, device=job["resources"]["device"], model_id=request.model_id)
        _require(receipt["model_id"] == request.model_id, "command_model_mismatch")
        _run_process(command, output, checkpoint, self.max_seconds)
        checkpoint()
        for item in snapshots:
            training._file(item)
        # Also detect changes to original sources and immutable model artifacts.
        for row in images:
            training._file(row["image"])
        _require(self.sources.resolve(model) == source_bundle, "model_sources_changed_during_prediction")
        _require(registry.resolve_candidate(request.registry_root, request.model_id) == model, "model_changed_during_prediction")
        mask_roots = [training._absolute(str(predictions / name)) for name in self.MASK_FOLDERS[request.target]]
        mask_roots = [path for path in mask_roots if path.is_dir()]
        _require(len(mask_roots) == 1, "prediction_masks_missing_or_ambiguous_layout")
        mask_root = mask_roots[0]
        found = {}
        for path in mask_root.rglob("*"):
            training._absolute(str(path))
            if not path.is_file() or not path.name.endswith((".nii", ".nii.gz")):
                continue
            token = path.name[:-7] if path.name.endswith(".nii.gz") else path.name[:-4]
            _require(path.parent == mask_root and token in mapping and token not in found,
                     "unexpected_or_duplicate_prediction_mask")
            found[token] = {"path": str(path), "sha256": training.sha256_file(path)}
        _require(set(found) == set(mapping), "incomplete_prediction_cohort")
        result = {mapping[token]: found[token] for token in sorted(found)}
        training._atomic_json(output / "case_map.json", {"schema": "segmentation-host-predictions-v1",
            "model_id": request.model_id, "cohort_sha256": request.cohort_sha256,
            "cases": [{"case_id": mapping[token], "input_name": token, "prediction": found[token]} for token in sorted(found)]})
        return cycle.PredictionReceipt(request.model_id, request.cohort_sha256, result)


def make_controller(root: Path, plan: cycle.CyclePlan | dict, *, database: Path,
                    prediction_seconds: float | None = None, baseline_vendor: Path | None = None) -> cycle.TrustedController:
    """Pure construction/preflight. Execution still requires run_cycle(execute=True)."""
    plan = plan.to_dict() if type(plan) is cycle.CyclePlan else copy.deepcopy(plan)
    cycle._preflight(plan)
    db = ReadOnlyDatabase(database)
    predictor = RegistryPredictor(plan, baseline_vendor=baseline_vendor, max_seconds=prediction_seconds)
    predictor.sources.resolve(registry.resolve_candidate(Path(plan["registry_root"]), plan["parent_id"]))
    return cycle.TrustedController(cycle.DatabaseAuthority(db), GlobalControl(root, plan), predictor, DatabaseEvidence(db))


def public_result(state: dict, controller: cycle.TrustedController, plan: dict) -> dict:
    """Small CLI/UI result; patient paths, exception text and reports stay private."""
    result = {k: state[k] for k in ("stage", "plan_sha256", "job_sha256", "candidate_id", "report_sha256") if k in state}
    result["missing_evidence"] = state.get("missing_evidence", [])
    report = state.get("report")
    result["comparison_status"] = "MEASURED" if report else "NOT_MEASURED"
    if report:
        # Patient IDs, paths and per-patient rows remain in the private journal.
        result["comparison_metrics"] = copy.deepcopy(report["metrics"])
        result["comparison_cohort"] = copy.deepcopy(report["cohort"])
        result["comparison_scope"] = "FROZEN_COHORT_ONLY"
        result["untouched_final_test_status"] = "NOT_ESTABLISHED_BY_THIS_COMPARISON"
    if state.get("assessment"):
        result["assessment_id"] = state["assessment"]["assessment_id"]
    evaluator = controller.evaluate
    lookup = evaluator.last_lookup
    if state.get("report") and state.get("candidate_id"):
        lookup = evaluator.inspect(cycle.EvaluationRequest(plan, cycle.plan_sha256(plan), state["candidate_id"],
            state["report_sha256"], state["report"]))
    result["evidence_status"] = lookup.status
    result["evidence_missing"] = list(lookup.missing_evidence)
    return result
