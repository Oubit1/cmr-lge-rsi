from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class ModelProvider(ABC):
    name: str

    @abstractmethod
    async def generate_structured(self, *, system_prompt: str, input_data: Dict[str, Any],
                                  output_schema: Type[T], model: str,
                                  timeout_seconds: int = 180) -> T:
        raise NotImplementedError
