from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any


DEFAULT_SOCKET = Path("/run/cmr-lge-label-executor/apply.sock")


class LabelExecutorUnavailable(RuntimeError):
    pass


def request_label_apply(*, dataset_root: Path, work_root: Path, edit_id: str,
                        reviewer: str, confirmation: str,
                        socket_path: Path | None = None) -> dict[str, Any]:
    """Submit a bounded apply request; this process never opens a label for writing."""
    target = socket_path or Path(os.getenv("CMR_LGE_LABEL_EXECUTOR_SOCKET", str(DEFAULT_SOCKET)))
    request = {
        "operation": "apply_staged_edit",
        "dataset_root": str(dataset_root.resolve()),
        "work_root": str(work_root.resolve()),
        "edit_id": edit_id,
        "reviewer": reviewer,
        "confirmation": confirmation,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10)
            client.connect(str(target))
            client.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
            chunks = []
            size = 0
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 1_048_576:
                    raise LabelExecutorUnavailable("label executor response is too large")
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except (FileNotFoundError, ConnectionError, socket.timeout, OSError) as exc:
        raise LabelExecutorUnavailable(
            f"separate label executor is unavailable at {target}"
        ) from exc
    try:
        response = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise LabelExecutorUnavailable("label executor returned an invalid response") from exc
    if not response.get("ok"):
        raise ValueError(str(response.get("error") or "label executor rejected request"))
    return dict(response["result"])
