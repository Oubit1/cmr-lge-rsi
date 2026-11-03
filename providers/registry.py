from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from providers.base import ModelProvider
from providers.mock import MockProvider
from providers.openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._providers: dict[str, ModelProvider] = {}

    def get(self, name: str) -> ModelProvider:
        if name in self._providers:
            return self._providers[name]
        item: dict[str, Any] = self.config.get("providers", {}).get(name, {})
        adapter = item.get("adapter")
        if adapter == "mock":
            provider: ModelProvider = MockProvider()
        elif adapter == "openai_compatible_chat":
            provider = OpenAICompatibleProvider(
                name=name,
                api_key_env=item["api_key_env"],
                base_url=item.get("base_url"),
                base_url_env=item.get("base_url_env"),
                json_mode=bool(item.get("json_mode", False)),
            )
        else:
            raise KeyError(f"unknown provider or adapter: {name}/{adapter}")
        self._providers[name] = provider
        return provider
