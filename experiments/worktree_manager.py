from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    def __init__(self, repository: Path, worktree_root: Path):
        self.repository = repository.resolve()
        self.worktree_root = worktree_root.resolve()

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repository), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def validate_repository(self) -> None:
        result = self._git("rev-parse", "--is-inside-work-tree", check=False)
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise ValueError(f"not a Git repository: {self.repository}")

    def create(self, experiment_id: int, base_ref: str = "HEAD") -> Path:
        self.validate_repository()
        target = self.worktree_root / f"exp_{experiment_id:06d}"
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        branch = f"cmr-lge-exp-{experiment_id:06d}"
        self._git("worktree", "add", "-b", branch, str(target), base_ref)
        return target
