from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from app.database import Database
from providers.base import ModelProvider


T = TypeVar("T", bound=BaseModel)


async def invoke_traced(*, db: Database, run_id: int, thread_id: str, role: str,
                        provider_name: str, provider: ModelProvider, model: str,
                        system_prompt: str, input_data: dict[str, Any],
                        output_schema: Type[T], timeout_seconds: int = 180) -> T:
    """Invoke one role while preserving its exact public input and provider events."""
    invocation_id = uuid.uuid4().hex
    public_input = {
        key: value for key, value in input_data.items()
        if not str(key).startswith("_")
    }
    db.begin_agent_invocation(
        invocation_id, run_id=run_id, thread_id=thread_id, role=role,
        provider=provider_name, requested_model=model,
        input_hash=hashlib.sha256(
            json.dumps(public_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(), input_data=input_data,
    )

    def record(event: dict[str, Any]) -> None:
        db.record_agent_invocation_event(invocation_id, event)

    traced_input = dict(input_data)
    traced_input["_budget_root"] = str(db.path.parent.parent)
    traced_input["_invocation_id"] = invocation_id
    traced_input["_call_event_callback"] = record
    try:
        result = await provider.generate_structured(
            system_prompt=system_prompt, input_data=traced_input,
            output_schema=output_schema, model=model, timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        db.finish_agent_invocation(
            invocation_id, status="FAILED",
            error_message=f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
        )
        raise
    if db.agent_invocation_event_count(invocation_id) == 0:
        if provider_name == "mock":
            db.record_agent_invocation_event(invocation_id, {
                "attempt": 1, "model": model, "status": "SUCCEEDED",
                "fallback_reason": "deterministic mock provider",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            })
        else:
            error = "provider returned without a persisted invocation event"
            db.finish_agent_invocation(invocation_id, status="FAILED", error_message=error)
            raise RuntimeError(error)
    db.finish_agent_invocation(invocation_id, status="SUCCEEDED")
    return result
