from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

from app.artifact_registry import collect_qc_artifacts
from app.database import Database
from app.state_machine import RunState
from app.task_runner import TaskRunner


class WorkflowOrchestrator:
    # Operator-owned runtime configuration, outside Owner source proposals.
    # Optional JSON with exactly two absolute paths: model_registry and
    # orchestrator_root. Absence keeps legacy annotation. No arbitrary argv.
    MODEL_CONFIG = "runtime/control/segmentation_inference.json"

    def __init__(self, db: Database, project_root: Path, skill_root: Path, python: Path):
        self.db = db
        self.project_root = project_root.resolve()
        self.skill_root = skill_root.resolve()
        self.python = python.resolve()
        self.runner = TaskRunner(db)

    def _model_config(self) -> dict | None:
        path = self.project_root / self.MODEL_CONFIG
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if (not isinstance(value, dict) or set(value) != {"model_registry", "orchestrator_root"}
                or any(not isinstance(v, str) or not v.strip() or not Path(v).is_absolute()
                       for v in value.values())):
            raise ValueError("invalid trusted segmentation inference configuration")
        registry = Path(value["model_registry"]).resolve()
        controller = Path(value["orchestrator_root"]).resolve()
        if not registry.is_dir() or not controller.is_dir() or not (controller / "scripts/segment_from_registry.py").is_file():
            raise ValueError("trusted segmentation registry or orchestrator script missing")
        return {"model_registry": str(registry), "orchestrator_root": str(controller)}

    def submit(self, *, dataset_root: Path, raw_input: Path | None, execute_mode: str,
               device: str, workers: int, run_annotation: bool) -> int:
        dataset_root = dataset_root.resolve()
        config = {
            "dataset_root": str(dataset_root),
            "raw_input": str(raw_input.resolve()) if raw_input else None,
            "device": device,
            "workers": workers,
            "run_annotation": run_annotation,
        }
        if run_annotation and execute_mode != "audit":
            models = self._model_config()
            if models is not None:
                config["segmentation_inference"] = models
        dataset_id = self.db.register_dataset(dataset_root.parent.name or dataset_root.name,
                                              dataset_root, raw_input)
        run_id = self.db.create_run(dataset_id, execute_mode, config)
        self.db.transition(run_id, RunState.QUEUED, current_step="queued")
        return run_id

    def build_pipeline_command(self, run: dict) -> list[str]:
        config = run["config"]
        command = [
            str(self.python),
            str(self.skill_root / "scripts/run_full_qc_pipeline.py"),
            "--root", run["processed_path"],
            "--device", config["device"],
            "--workers", str(config["workers"]),
        ]
        if config.get("raw_input"):
            command += ["--input", config["raw_input"]]
        if not config.get("run_annotation", False):
            command.append("--skip-annotation")
        if run["execute_mode"] == "audit":
            command.append("--audit-only")
        elif config.get("run_annotation", False) and "segmentation_inference" in config:
            trusted = self._model_config()
            if trusted is None or config["segmentation_inference"] != trusted:
                raise ValueError("trusted segmentation inference configuration changed; resubmit the run")
            command += ["--model-registry", trusted["model_registry"],
                        "--orchestrator-root", trusted["orchestrator_root"]]
        return command

    def execute(self, run_id: int) -> dict:
        run = self.db.get_run(run_id)
        if run["status"] not in {RunState.QUEUED.value}:
            raise ValueError(f"run {run_id} is {run['status']}, expected QUEUED")
        if not self.db.claim_run(run_id):
            raise RuntimeError(f"run {run_id} was already claimed by another worker")
        try:
            command = self.build_pipeline_command(run)
        except (OSError, ValueError):
            # A changed/missing trusted opt-in must neither fall back to legacy
            # nor strand this run in RUNNING or at the front of the queue.
            self.db.transition(run_id, RunState.ERROR, current_step="pipeline_configuration",
                               error_message="invalid or changed trusted pipeline configuration")
            raise
        run_dir = self.project_root / "artifacts" / f"run_{run_id:06d}"
        # Command construction/submission does not need GPU platform bindings.
        from app.gpu_scheduler import GpuLease, normalize_gpu_id
        gpu_id = normalize_gpu_id(run["config"]["device"]) if run["config"].get("run_annotation") else None
        lease = GpuLease(self.project_root / "runtime/gpu_locks", gpu_id).acquire() if gpu_id else nullcontext(None)
        with lease:
            task_id, returncode = self.runner.run(
                run_id, "full_qc_pipeline", command, run_dir / "full_qc_pipeline.log",
                cwd=self.project_root,
                env={"CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id else None,
                gpu_id=gpu_id,
            )
        if returncode != 0:
            self.db.transition(run_id, RunState.ERROR, current_step="full_qc_pipeline",
                               error_message=f"pipeline exited with {returncode}")
            return self.db.run_details(run_id)
        artifact_summary = collect_qc_artifacts(self.db, run_id, task_id, Path(run["processed_path"]))
        target = RunState.QC_FAILED if artifact_summary["issue_rows"] else RunState.QC_PASSED
        self.db.transition(run_id, target, current_step="qc_complete")
        (run_dir / "artifact_summary.json").write_text(
            json.dumps(artifact_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return self.db.run_details(run_id)
