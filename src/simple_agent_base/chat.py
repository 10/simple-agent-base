from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from simple_agent_base.transcript import build_transcript, messages_from_items, persist_chat_items
from simple_agent_base.types import (
    AgentEvent,
    AgentRunResult,
    ChatMessage,
    ChatSnapshot,
    ConversationItem,
    JSONObject,
    MessageInput,
)

if TYPE_CHECKING:
    from simple_agent_base.agent import Agent


class ChatSession:
    def __init__(
        self,
        agent: Agent,
        items: Sequence[ConversationItem] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._agent = agent
        self._items = list(items or [])
        self._system_prompt = system_prompt

    @property
    def history(self) -> list[ChatMessage]:
        return messages_from_items(self._items)

    @property
    def items(self) -> list[ConversationItem]:
        return list(self._items)

    def snapshot(self) -> ChatSnapshot:
        return ChatSnapshot(
            items=self.items,
            system_prompt=self._system_prompt,
        )

    def export(self) -> JSONObject:
        return self.snapshot().model_dump(mode="json")

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        resolved_system_prompt = self._agent._resolve_system_prompt_with_default(
            system_prompt,
            self._system_prompt,
        )
        transcript = build_transcript(
            input_data,
            system_prompt=resolved_system_prompt,
            prefix_items=self._items,
        )
        result = await self._agent._run_transcript(transcript, response_model=response_model)
        self._items = persist_chat_items(
            transcript,
            system_prompt=resolved_system_prompt,
        )
        return result

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        resolved_system_prompt = self._agent._resolve_system_prompt_with_default(
            system_prompt,
            self._system_prompt,
        )
        transcript = build_transcript(
            input_data,
            system_prompt=resolved_system_prompt,
            prefix_items=self._items,
        )

        async for event in self._agent._stream_transcript(transcript, response_model=response_model):
            if event.type == "completed":
                self._items = persist_chat_items(
                    transcript,
                    system_prompt=resolved_system_prompt,
                )
            yield event

    def reset(self) -> None:
        self._items.clear()

    def run_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        return self._agent._run_sync_call(
            lambda: self.run(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="run_sync()",
            async_hint="await chat.run(...)",
        )

    def stream_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[AgentEvent]:
        return self._agent._stream_sync_call(
            lambda: self.stream(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="stream_sync()",
            async_hint="async for event in chat.stream(...)",
        )
