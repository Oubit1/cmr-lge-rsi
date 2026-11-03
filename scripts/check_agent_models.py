#!/usr/bin/env python3
"""Make one de-identified structured call to every configured unique model."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.shadow_analysis import expand_env  # noqa: E402
from providers.registry import ProviderRegistry  # noqa: E402


class HealthReply(BaseModel):
    ok: bool
    role: str
    message: str


async def check(providers: Path, agents: Path) -> dict[str, Any]:
    registry = ProviderRegistry(providers)
    config = expand_env(yaml.safe_load(agents.read_text(encoding="utf-8")) or {})
    seen: set[tuple[str, str]] = set()
    targets: list[tuple[str, str, str]] = []
    for role, item in config.get("agents", {}).items():
        key = (str(item["provider"]), str(item["model"]))
        if key in seen:
            continue
        seen.add(key)
        targets.append((role, key[0], key[1]))

    async def one(role: str, provider_name: str, model: str) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        row: dict[str, Any] = {"role": role, "provider": provider_name, "model": model}
        try:
            reply = await registry.get(provider_name).generate_structured(
                system_prompt="Return the requested health-check JSON. Do not perform medical analysis.",
                input_data={"role": role, "task": "configuration_health_check",
                            "contains_medical_data": False, "force_model": True},
                output_schema=HealthReply, model=model, timeout_seconds=60,
            )
            row.update(status="OK" if reply.ok else "INVALID", message=reply.message[:120])
        except Exception as exc:
            row.update(status="ERROR", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        row["latency_ms"] = round((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        return row

    results = list(await asyncio.gather(*(one(*target) for target in targets)))
    report = {"schema_version": "agent-model-health-v1",
              "checked_at": datetime.now(timezone.utc).isoformat(), "results": results,
              "all_ok": bool(results) and all(row["status"] == "OK" for row in results)}
    output = ROOT / "runtime" / "agent_model_health.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", type=Path, default=ROOT / "configs/providers.yaml")
    parser.add_argument("--agents", type=Path, default=ROOT / "configs/agents.aliyun-roles.yaml")
    args = parser.parse_args()
    report = asyncio.run(check(args.providers, args.agents))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_ok"] else 2)


if __name__ == "__main__":
    main()
