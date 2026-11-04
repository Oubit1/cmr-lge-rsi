from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Type

import yaml
from pydantic import BaseModel

from contracts.compat import model_validate
from providers.registry import ProviderRegistry


class Reply(BaseModel):
    text: str


def _configs(root: Path) -> tuple[Path, Path]:
    agents = Path(os.environ.get("CMR_LGE_AGENTS_CONFIG", root / "configs/agents.aliyun-roles.yaml"))
    providers = Path(os.environ.get("CMR_LGE_PROVIDERS_CONFIG", root / "configs/providers.yaml"))
    return agents, providers


def subscription_active(role: str, root: Path) -> bool:
    agents_path, providers_path = _configs(root)
    if not agents_path.is_file() or not providers_path.is_file():
        return False
    agents = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
    providers = yaml.safe_load(providers_path.read_text(encoding="utf-8")) or {}
    name = (agents.get("agents", {}).get(role) or {}).get("provider")
    return bool(name and (providers.get("providers", {}).get(name) or {}).get("adapter") == "codex_cli")


def generate_reply(role: str, messages: list[dict], *, root: Path, timeout: int,
                   budget_callback, task_prompt: str = "") -> dict:
    agents_path, providers_path = _configs(root)
    agents = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
    role_cfg = (agents.get("agents") or {}).get(role) or {}
    provider_name = role_cfg.get("provider")
    if not provider_name:
        raise RuntimeError(f"agent role is not configured: {role}")
    registry = ProviderRegistry(providers_path)
    model = str(role_cfg.get("model") or "")
    prompt = task_prompt or "Return a helpful response to the conversation."
    called = False
    callback_lock = threading.Lock()
    def reserve_one_request() -> None:
        nonlocal called
        with callback_lock:
            if called:
                raise RuntimeError("board request budget callback may be used only once")
            called = True
        budget_callback()
    events = []
    data = {"messages": messages, "task_prompt": prompt, "role": role,
            "_call_event_callback": events.append,
            "_request_budget_callback": reserve_one_request, "_fallback_on_error": False}
    result = asyncio.run(registry.get(provider_name).generate_structured(
        system_prompt=prompt, input_data=data, output_schema=Reply, model=model,
        timeout_seconds=timeout))
    text = result.text if isinstance(result, Reply) else model_validate(Reply, result).text
    usage = {k: events[-1].get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if events else {}
    return {"choices": [{"message": {"content": text}}], "model": model, "usage": usage}
