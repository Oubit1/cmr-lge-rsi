from __future__ import annotations

import fcntl
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class GpuLease:
    """Simple cross-process GPU lock. Selection remains explicit or 'cpu'."""

    def __init__(self, lock_root: Path, gpu_id: str):
        self.lock_root = lock_root
        self.gpu_id = gpu_id

    @contextmanager
    def acquire(self) -> Iterator[str]:
        self.lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.lock_root / f"gpu_{self.gpu_id}.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self.gpu_id
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def select_free_gpu() -> str | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        return None
    candidates = []
    for line in result.stdout.splitlines():
        try:
            index, free_memory, utilization = [int(part.strip()) for part in line.split(",")]
        except (ValueError, TypeError):
            continue
        candidates.append((free_memory, -utilization, -index, str(index)))
    return max(candidates)[-1] if candidates else None


def normalize_gpu_id(device: str) -> str | None:
    value = device.lower().strip()
    if value == "cpu":
        return None
    if value == "auto":
        return select_free_gpu()
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    return value if value.isdigit() else None
