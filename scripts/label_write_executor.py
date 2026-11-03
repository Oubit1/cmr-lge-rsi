#!/usr/bin/env python3
"""Minimal production-label writer intended for a separate Unix identity."""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.manual_annotation import apply_edit  # noqa: E402


def append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class ApplyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        response: dict[str, Any]
        try:
            raw = self.rfile.readline(1_048_577)
            if not raw or len(raw) > 1_048_576:
                raise ValueError("request must be one JSON line below 1 MiB")
            payload = json.loads(raw.decode("utf-8"))
            server = self.server
            if payload.get("operation") != "apply_staged_edit":
                raise ValueError("unsupported label operation")
            dataset = Path(str(payload.get("dataset_root") or "")).resolve()
            work = Path(str(payload.get("work_root") or "")).resolve()
            if dataset != server.dataset_root or work != server.work_root:  # type: ignore[attr-defined]
                raise ValueError("request roots do not match executor allowlist")
            result = apply_edit(
                dataset, work, str(payload.get("edit_id") or ""),
                reviewer=str(payload.get("reviewer") or ""),
                confirmation=str(payload.get("confirmation") or ""),
            )
            response = {"ok": True, "result": result}
        except Exception as exc:  # fail closed at the privilege boundary
            response = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}"[:1000]}
        append_audit(self.server.audit_log, {  # type: ignore[attr-defined]
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "peer": "local_unix_socket",
            "ok": bool(response.get("ok")),
            "edit_id": str(locals().get("payload", {}).get("edit_id") or ""),
            "error": response.get("error"),
        })
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")


class ExecutorServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: Path, dataset_root: Path, work_root: Path,
                 audit_log: Path):
        self.dataset_root = dataset_root.resolve()
        self.work_root = work_root.resolve()
        self.audit_log = audit_log.resolve()
        super().__init__(str(socket_path), ApplyHandler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()
    old_umask = os.umask(0o117)
    try:
        with ExecutorServer(args.socket, args.dataset, args.work_root, args.audit_log) as server:
            os.chmod(args.socket, 0o660)
            server.serve_forever()
    finally:
        os.umask(old_umask)
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
