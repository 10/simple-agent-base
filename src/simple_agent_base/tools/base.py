from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, ConfigDict, create_model

from simple_agent_base.errors import ToolDefinitionError
from simple_agent_base.types import JSONObject, ToolDefinition

TOOL_DEFINITION_ATTR = "__simple_agent_base_tool_definition__"


def build_arguments_model(func: Callable[..., object]) -> type[BaseModel]:
    signature = inspect.signature(func)
    fields: dict[str, tuple[object, object]] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            raise ToolDefinitionError(f"Tool '{func.__name__}' cannot use *args or **kwargs.")

        if parameter.annotation is inspect.Signature.empty:
            raise ToolDefinitionError(
                f"Tool '{func.__name__}' must declare a type annotation for '{parameter.name}'."
            )

        default = ... if parameter.default is inspect.Signature.empty else parameter.default
        fields[parameter.name] = (parameter.annotation, default)

    model_name = f"{func.__name__.title().replace('_', '')}Arguments"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def build_tool_definition(func: Callable[..., object]) -> ToolDefinition:
    arguments_model = build_arguments_model(func)
    parameters = cast(JSONObject, arguments_model.model_json_schema())
    doc = inspect.getdoc(func) or ""
    first_doc_line = doc.strip().splitlines()[0].strip() if doc.strip() else ""

    return ToolDefinition(
        name=func.__name__,
        description=first_doc_line or f"Run the {func.__name__} tool.",
        parameters=parameters,
        func=func,
        arguments_model=arguments_model,
        is_async=inspect.iscoroutinefunction(func),
    )


def get_tool_definition(func: Callable[..., object]) -> ToolDefinition | None:
    return cast(ToolDefinition | None, getattr(func, TOOL_DEFINITION_ATTR, None))


def dump_tool_output(value: object) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, BaseModel):
        payload: object = value.model_dump(mode="json")
    else:
        payload = value

    return json.dumps(payload, ensure_ascii=False, default=str)
