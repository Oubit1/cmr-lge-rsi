from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel, ValidationError

from contracts.compat import model_schema, model_validate
from providers.base import ModelProvider
from providers.openai_compatible import _record_call_event, _usage

T = TypeVar("T", bound=BaseModel)


class CodexCliProvider(ModelProvider):
    """Run the authenticated official Codex CLI in isolated read-only mode."""

    def __init__(self, name: str = "codex_subscription", binary_env: str = "CODEX_CLI_PATH",
                 reasoning_effort: str | dict[str, str] | None = None):
        self.name = name
        self.binary_env = binary_env
        self.reasoning_effort = reasoning_effort or {}

    def _binary(self) -> str:
        binary = os.environ.get(self.binary_env) or shutil.which("codex")
        if not binary:
            raise RuntimeError("Codex CLI executable not found")
        return binary

    async def generate_structured(self, *, system_prompt: str, input_data: Dict[str, Any],
                                  output_schema: Type[T], model: str,
                                  timeout_seconds: int = 180) -> T:
        binary = self._binary()
        status = await asyncio.create_subprocess_exec(binary, "login", "status",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    env={k:v for k,v in os.environ.items() if not k.upper().endswith("_API_KEY")})
        try:
            status_out, _ = await asyncio.wait_for(status.communicate(), timeout=min(timeout_seconds, 10))
        except asyncio.TimeoutError as exc:
            status.kill()
            await status.wait()
            raise RuntimeError("Codex CLI ChatGPT login check timed out") from exc
        if status.returncode or b"ChatGPT" not in status_out:
            raise RuntimeError("Codex CLI must be logged in with a ChatGPT account")
        schema = model_schema(output_schema)
        public_input = {k: v for k, v in input_data.items() if not str(k).startswith("_")}
        prompt = (system_prompt + "\n\nUser input:\n" + json.dumps(public_input, ensure_ascii=False)
                  + "\n\nRequired result schema:\n" + json.dumps(schema, ensure_ascii=False)
                  + "\nReturn the result as JSON encoded inside the response_json string. "
                    "The decoded object must satisfy the required result schema. Do not use tools.")
        callback = input_data.get("_call_event_callback")
        invocation = str(input_data.get("_invocation_id") or "")
        call = {"timestamp": datetime.now(timezone.utc).isoformat(), "provider": self.name,
                "model": model, "requested_model": model, "fallback_reason": None,
                "role": str(input_data.get("role") or input_data.get("target") or "unspecified"),
                "task_kind": str(input_data.get("_task_kind") or "general"),
                "input_bytes": len(json.dumps(public_input, ensure_ascii=False).encode()), "attempt": 1}
        if invocation:
            call["invocation_id"] = invocation
        from app.workflow_budget import reserve
        budget = input_data.get("_request_budget_callback")
        if callable(budget):
            budget()
        else:
            reserve("requests", root=Path(input_data.get("_budget_root") or Path(__file__).resolve().parents[1]))

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="codex-provider-") as isolated:
            schema_path = Path(isolated) / "schema.json"
            output_path = Path(isolated) / "result.json"
            # Domain contracts include arbitrary dictionaries and optional fields.
            # Preserve those contracts, using a strict transport envelope rather
            # than silently changing their semantics to satisfy OpenAI's subset.
            envelope = {"type": "object", "properties": {"response_json": {"type": "string"}},
                        "required": ["response_json"], "additionalProperties": False}
            schema_path.write_text(json.dumps(envelope), encoding="utf-8")
            args = [binary, "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check", "--sandbox", "read-only",
                "--model", model, "--output-schema", str(schema_path),
                "--output-last-message", str(output_path), "--cd", isolated,
                "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
                "-c", "features.multi_agent=false", "-c", 'web_search="disabled"']
            effort = input_data.get("_reasoning_effort")
            if effort is None:
                effort = (self.reasoning_effort.get(str(input_data.get("role") or input_data.get("target") or ""))
                          if isinstance(self.reasoning_effort, dict) else self.reasoning_effort)
            if effort:
                args.extend(["-c", f'model_reasoning_effort="{effort}"'])
            args.append("-")
            env = {k:v for k,v in os.environ.items() if not k.upper().endswith("_API_KEY")}
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=isolated,
                env=env,
                **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout_seconds)
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
                err = TimeoutError("Codex CLI request timed out") if isinstance(exc, asyncio.TimeoutError) else RuntimeError("Codex CLI request cancelled")
                _record_call_event({**call, "status": "ERROR", "latency_ms": round((time.monotonic()-started)*1000),
                                    "error": str(err), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, callback)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise RuntimeError(str(err)) from err
            if proc.returncode:
                # Do not surface stderr: CLI diagnostics can contain sensitive context.
                err = RuntimeError(f"Codex CLI exited with status {proc.returncode}")
                _record_call_event({**call, "status": "ERROR", "latency_ms": round((time.monotonic()-started)*1000),
                                    "error": str(err), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, callback)
                raise err
            try:
                transport = json.loads(output_path.read_text(encoding="utf-8"))
                result = model_validate(output_schema, json.loads(transport["response_json"]))
            except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                _record_call_event({**call, "status": "ERROR", "latency_ms": round((time.monotonic()-started)*1000),
                                    "error": f"{type(exc).__name__}: structured output invalid",
                                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, callback)
                raise RuntimeError("Codex CLI returned invalid structured output") from exc
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                    if event.get("type") == "turn.completed":
                        usage = _usage({"usage": event.get("usage", {})})
                except (json.JSONDecodeError, AttributeError):
                    continue
            _record_call_event({**call, "status": "SUCCEEDED", "latency_ms": round((time.monotonic()-started)*1000),
                                **usage}, callback)
            return result
