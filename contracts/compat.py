from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def model_validate(model: Type[T], value: Any) -> T:
    if hasattr(model, "model_validate"):
        return model.model_validate(value)
    return model.parse_obj(value)


def model_dump(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    return value.dict(by_alias=True)


def model_schema(model: Type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()
