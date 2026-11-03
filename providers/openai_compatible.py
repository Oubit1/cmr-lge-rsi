from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from contracts.compat import model_schema, model_validate
from providers.base import ModelProvider


T = TypeVar("T", bound=BaseModel)
_TELEMETRY_LOCK = threading.Lock()
_DEFAULT_TELEMETRY_PATH = Path(__file__).resolve().parents[1] / "runtime" / "agent_calls.jsonl"


def _write_call_event(event: dict[str, Any]) -> None:
    """Write call metadata only; prompts, paths, images, masks and keys are excluded."""
    path = Path(os.getenv("CMR_LGE_AGENT_CALL_LOG", str(_DEFAULT_TELEMETRY_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TELEMETRY_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _record_call_event(event: dict[str, Any], callback: Any = None) -> None:
    """Persist telemetry and optionally attach it to an orchestrator invocation."""
    _write_call_event(event)
    if callback is not None:
        try:
            callback(event)
        except Exception:
            # Audit persistence must never alter the provider result. The JSONL
            # record remains the recovery source if the database is unavailable.
            pass


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }


def _healthy_model(requested: str, force: bool = False) -> tuple[str, str | None]:
    """Temporarily bypass a recently verified inaccessible model."""
    fallback = os.getenv("ALIYUN_MODEL", "")
    if force or not fallback or requested == fallback:
        return requested, None
    path = _DEFAULT_TELEMETRY_PATH.parent / "agent_model_health.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        checked = datetime.fromisoformat(report["checked_at"])
        if (datetime.now(timezone.utc) - checked).total_seconds() > 6 * 3600:
            return requested, None
        row = next((x for x in report.get("results", []) if x.get("model") == requested), None)
        if row and row.get("status") == "ERROR":
            return fallback, f"health fallback from {requested}"
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return requested, None


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


class OpenAICompatibleProvider(ModelProvider):
    """Adapter for OpenAI-compatible Chat Completions endpoints.

    Uses only the Python standard library for transport so it works in the
    existing medical-imaging environment without mutating shared packages.
    Outputs are always validated by Pydantic.
    """

    def __init__(self, name: str, api_key_env: str, base_url: str | None = None,
                 base_url_env: str | None = None, json_mode: bool = False):
        self.name = name
        self.api_key_env = api_key_env
        self.base_url = os.getenv(base_url_env, "") if base_url_env else (base_url or "")
        self.base_url_env = base_url_env
        self.json_mode = json_mode

    def _credentials(self) -> tuple[str, str]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        if self.base_url_env and not self.base_url:
            raise RuntimeError(f"missing provider base URL environment variable: {self.base_url_env}")
        base_url = self.base_url or "https://api.openai.com/v1"
        return api_key, base_url.rstrip("/")

    @staticmethod
    async def _post(endpoint: str, api_key: str, payload: dict, timeout_seconds: int) -> dict:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if response.is_error:
            raise RuntimeError(f"provider HTTP {response.status_code}: {response.text[:1000]}")
        return response.json()

    async def generate_structured(self, *, system_prompt: str, input_data: Dict[str, Any],
                                  output_schema: Type[T], model: str,
                                  timeout_seconds: int = 180) -> T:
        api_key, base_url = self._credentials()
        requested_model = model
        resolved_model, fallback_reason = _healthy_model(
            requested_model, force=bool(input_data.get("force_model")))
        fallback_model = os.getenv("ALIYUN_MODEL", "").strip()
        candidates = [(resolved_model, fallback_reason)]
        if (not input_data.get("force_model")
                and input_data.get("_fallback_on_error", True)
                and fallback_model and resolved_model != fallback_model):
            candidates.append(
                (fallback_model, f"runtime fallback after {resolved_model} failed")
            )
        schema = model_schema(output_schema)
        callback = input_data.get("_call_event_callback")
        invocation_id = str(input_data.get("_invocation_id") or "")
        public_input = {key: value for key, value in input_data.items()
                        if not str(key).startswith("_")}
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({
                "input": public_input,
                "required_output_schema": schema,
                "instruction": "Return one JSON object only. Do not include markdown or hidden reasoning.",
            }, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        schema_attempts = 1 + max(0, min(1, int(input_data.get("_schema_retries", 1))))
        for active_model, active_fallback_reason in candidates:
            messages = list(base_messages)
            for attempt in range(schema_attempts):
                request = dict(
                    model=active_model,
                    messages=messages,
                    temperature=0,
                    timeout=timeout_seconds,
                )
                # Alibaba hybrid-thinking models may spend the entire output
                # budget on reasoning and leave no JSON content. Model Studio
                # recommends disabling thinking for structured output. This is
                # an Aliyun-only extension and must not leak to OpenAI endpoints.
                if self.name == "aliyun":
                    request["enable_thinking"] = bool(
                        input_data.get("_enable_thinking", False)
                    )
                if input_data.get("_max_output_tokens"):
                    request["max_tokens"] = int(input_data["_max_output_tokens"])
                if self.json_mode:
                    request["response_format"] = {"type": "json_object"}
                started = time.monotonic()
                call = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": self.name,
                    "model": active_model,
                    "requested_model": requested_model,
                    "fallback_reason": active_fallback_reason,
                    "role": str(input_data.get("role") or input_data.get("target") or "unspecified"),
                    "task_kind": str(input_data.get("_task_kind") or "general"),
                    "input_bytes": len(json.dumps(public_input, ensure_ascii=False).encode("utf-8")),
                    "attempt": attempt + 1,
                }
                if invocation_id:
                    call["invocation_id"] = invocation_id
                # Claim every actual HTTP request, including fallback/schema retries.
                # This is outside the transport handler: budget denial must not fallback.
                from app.workflow_budget import reserve
                request_budget=input_data.get("_request_budget_callback")
                if callable(request_budget):request_budget()
                else:reserve("requests", root=Path(input_data.get("_budget_root") or Path(__file__).resolve().parents[1]))
                try:
                    response = await self._post(
                        f"{base_url}/chat/completions", api_key, request, timeout_seconds
                    )
                except Exception as exc:
                    last_error = exc
                    _record_call_event({
                        **call, "status": "ERROR",
                        "latency_ms": round((time.monotonic() - started) * 1000),
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    }, callback)
                    # A transport/model error is not repaired by repeating the
                    # same request. Move once to the audited base-model fallback.
                    break
                try:
                    content = response["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as exc:
                    last_error = RuntimeError(
                        "provider response lacks choices[0].message.content")
                    break
                try:
                    parsed = model_validate(output_schema, _extract_json(content))
                    _record_call_event({
                        **call, "status": "SUCCEEDED",
                        "latency_ms": round((time.monotonic() - started) * 1000),
                        "request_id": str(response.get("id") or ""), **_usage(response),
                    }, callback)
                    return parsed
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_error = exc
                    _record_call_event({
                        **call, "status": "SCHEMA_RETRY",
                        "latency_ms": round((time.monotonic() - started) * 1000),
                        "request_id": str(response.get("id") or ""), **_usage(response),
                    }, callback)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                                     "The response failed schema validation. Return a corrected JSON object only."})
        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        raise RuntimeError(
            f"all provider model candidates failed for {requested_model}: {detail}")
