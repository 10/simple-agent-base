from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import cast

from simple_agent_base.errors import ToolExecutionError, ToolRegistrationError
from simple_agent_base.types import JSONObject, ToolCallRequest, ToolDefinition, ToolExecutionResult

from .base import build_tool_definition, dump_tool_output, get_tool_definition


class ToolRegistry:
    def __init__(self, tools: Iterable[Callable[..., object]] | None = None) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

        for tool in tools or ():
            self.register(tool)

    def register(self, tool_fn: Callable[..., object]) -> None:
        definition = get_tool_definition(tool_fn) or build_tool_definition(tool_fn)

        if definition.name in self._definitions:
            raise ToolRegistrationError(f"Tool '{definition.name}' is already registered.")

        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ToolRegistrationError(f"Tool '{name}' is not registered.") from exc

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def to_openai_tools(self) -> list[JSONObject]:
        return [
            {
                "type": "function",
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
                "strict": True,
            }
            for definition in self._definitions.values()
        ]

    async def execute(self, call: ToolCallRequest) -> ToolExecutionResult:
        definition = self.get(call.name)

        try:
            validated = definition.arguments_model.model_validate(call.arguments)
            arguments = cast(dict[str, object], validated.model_dump(mode="python"))
            if definition.is_async:
                raw_output = await definition.func(**arguments)
            else:
                raw_output = await asyncio.to_thread(definition.func, **arguments)
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{call.name}' failed: {exc}") from exc

        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            arguments=arguments,
            output=dump_tool_output(raw_output),
            raw_output=raw_output,
        )
