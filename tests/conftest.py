from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from simple_agent_base.errors import ProviderError
from simple_agent_base.providers.base import ProviderEvent, ProviderResponse
from simple_agent_base.types import ConversationItem


class FakeProvider:
    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        *,
        stream_sequences: list[list[ProviderEvent]] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.stream_sequences = list(stream_sequences or [])
        self.calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "input_items": list(input_items),
                "tools": list(tools),
                "response_model": response_model,
            }
        )
        if not self.responses:
            raise ProviderError("No more fake responses configured.")
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        call = {
            "input_items": list(input_items),
            "tools": list(tools),
            "response_model": response_model,
        }
        self.calls.append(call)
        if not self.stream_sequences:
            raise ProviderError("No more fake stream sequences configured.")
        for event in self.stream_sequences.pop(0):
            yield event

    async def close(self) -> None:
        return None
