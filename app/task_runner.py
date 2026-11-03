from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.database import Database


SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def sanitized_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    # Deterministic preprocessing/QC subprocesses do not need cloud secrets.
    # Keep provider credentials only in the orchestrator process that performs
    # an explicit shadow-agent call.
    env = {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in SECRET_MARKERS)
    }
    if extra:
        env.update(extra)
    return env


class TaskRunner:
    def __init__(self, db: Database):
        self.db = db

    def run(self, run_id: int, task_type: str, command: list[str], log_path: Path,
            cwd: Path | None = None, env: dict[str, str] | None = None,
            gpu_id: str | None = None) -> tuple[int, int]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        task_id = self.db.create_task(run_id, task_type, command, log_path, gpu_id)
        self.db.update_task(task_id, "RUNNING")
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("command=" + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                env=sanitized_environment(env),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            self.db.mark_task_process(task_id, process.pid)
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        self.db.touch_task(task_id)
            except BaseException:
                process.terminate()
                try:
                    returncode = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                self.db.update_task(task_id, "FAILED", exit_code=returncode)
                raise
        self.db.update_task(task_id, "SUCCEEDED" if returncode == 0 else "FAILED",
                            exit_code=returncode)
        return task_id, returncode
