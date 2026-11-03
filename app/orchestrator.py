from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

from app.artifact_registry import collect_qc_artifacts
from app.database import Database
from app.gpu_scheduler import GpuLease, normalize_gpu_id
from app.state_machine import RunState
from app.task_runner import TaskRunner


class WorkflowOrchestrator:
    def __init__(self, db: Database, project_root: Path, skill_root: Path, python: Path):
        self.db = db
        self.project_root = project_root.resolve()
        self.skill_root = skill_root.resolve()
        self.python = python.resolve()
        self.runner = TaskRunner(db)

    def submit(self, *, dataset_root: Path, raw_input: Path | None, execute_mode: str,
               device: str, workers: int, run_annotation: bool) -> int:
        dataset_root = dataset_root.resolve()
        dataset_id = self.db.register_dataset(dataset_root.parent.name or dataset_root.name,
                                              dataset_root, raw_input)
        config = {
            "dataset_root": str(dataset_root),
            "raw_input": str(raw_input.resolve()) if raw_input else None,
            "device": device,
            "workers": workers,
            "run_annotation": run_annotation,
        }
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
        return command

    def execute(self, run_id: int) -> dict:
        run = self.db.get_run(run_id)
        if run["status"] not in {RunState.QUEUED.value}:
            raise ValueError(f"run {run_id} is {run['status']}, expected QUEUED")
        if not self.db.claim_run(run_id):
            raise RuntimeError(f"run {run_id} was already claimed by another worker")
        run_dir = self.project_root / "artifacts" / f"run_{run_id:06d}"
        command = self.build_pipeline_command(run)
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
