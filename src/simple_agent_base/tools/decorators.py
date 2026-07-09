from __future__ import annotations

from collections.abc import Callable

from .base import TOOL_DEFINITION_ATTR, build_tool_definition


def tool(
    func: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., object] | Callable[[Callable[..., object]], Callable[..., object]]:
    def decorator(inner: Callable[..., object]) -> Callable[..., object]:
        definition = build_tool_definition(inner)
        if name:
            definition.name = name
        if description:
            definition.description = description
        setattr(inner, TOOL_DEFINITION_ATTR, definition)
        return inner

    if func is None:
        return decorator

    return decorator(func)
