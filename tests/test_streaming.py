from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from conftest import FakeProvider
from simple_agent_base import Agent, AgentConfig, ChatMessage, ChatSnapshot, UsageMetadata, tool
from simple_agent_base.errors import ToolExecutionError
from simple_agent_base.providers.base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderHostedToolCallEvent,
    ProviderReasoningDeltaEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
    ProviderToolArgumentsDeltaEvent,
)


class ExplodingStreamingProvider(FakeProvider):
    async def stream_response(self, **_kwargs: Any) -> AsyncIterator[ProviderEvent]:
        raise RuntimeError("stream failed")
        yield


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


@tool
async def slow_ping(message: str) -> str:
    """Echo a message back after a short delay."""
    await asyncio.sleep(0.05)
    return f"pong: {message}"


@tool
async def very_slow_ping(message: str) -> str:
    """Echo a message back after a longer delay."""
    await asyncio.sleep(0.2)
    return f"pong: {message}"


@tool
async def explode(message: str) -> str:
    """Always fail."""
    raise ValueError(message)


class Summary(BaseModel):
    title: str
    bullets: list[str]


REASONING_PROMPT = "Think carefully, then answer with exactly reasoning-ok."
REASONING_SUMMARY = "Checking the exact-output constraint."


def make_reasoning_stream(provider_summary: str = REASONING_SUMMARY) -> list[ProviderEvent]:
    return [
        ProviderReasoningDeltaEvent(delta=provider_summary),
        ProviderCompletedEvent(
            response=ProviderResponse(
                output_text="reasoning-ok",
                reasoning_summary=provider_summary,
            )
        ),
    ]


@pytest.mark.asyncio
async def test_stream_yields_text_delta_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderTextDeltaEvent(delta="Hel"),
                ProviderTextDeltaEvent(delta="lo"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Hello",
                    )
                ),
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = [event async for event in agent.stream("Say hello.")]

    assert [event.delta for event in events if event.type == "text_delta"] == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_stream_yields_reasoning_delta_events() -> None:
    provider = FakeProvider(stream_sequences=[make_reasoning_stream()])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = [event async for event in agent.stream(REASONING_PROMPT)]

    assert [event.delta for event in events if event.type == "reasoning_delta"] == [REASONING_SUMMARY]


def test_stream_sync_yields_text_delta_and_completed_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderTextDeltaEvent(delta="Hel"),
                ProviderTextDeltaEvent(delta="lo"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Hello",
                    )
                ),
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = list(agent.stream_sync("Say hello."))

    assert [event.delta for event in events if event.type == "text_delta"] == ["Hel", "lo"]
    assert events[-1].type == "completed"
    assert events[-1].result is not None
    assert events[-1].result.output_text == "Hello"


def test_stream_sync_yields_reasoning_delta_events() -> None:
    provider = FakeProvider(stream_sequences=[make_reasoning_stream()])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = list(agent.stream_sync(REASONING_PROMPT))

    assert [event.delta for event in events if event.type == "reasoning_delta"] == [REASONING_SUMMARY]


@pytest.mark.asyncio
async def test_stream_sync_raises_inside_running_event_loop() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=FakeProvider(stream_sequences=[]))

    with pytest.raises(RuntimeError, match="stream_sync\\(\\) cannot be used inside a running event loop"):
        list(agent.stream_sync("Say hello."))


@pytest.mark.asyncio
async def test_stream_prepends_run_level_system_prompt() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderTextDeltaEvent(delta="Hello"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Hello",
                    )
                ),
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    _ = [event async for event in agent.stream("Say hello.", system_prompt="Be concise.")]

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "Be concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Say hello.",
        },
    ]


@pytest.mark.asyncio
async def test_stream_completed_result_includes_reasoning_summary() -> None:
    provider = FakeProvider(stream_sequences=[make_reasoning_stream()])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = [event async for event in agent.stream(REASONING_PROMPT)]

    assert events[-1].type == "completed"
    assert events[-1].result is not None
    assert events[-1].result.reasoning_summary == REASONING_SUMMARY


@pytest.mark.asyncio
async def test_stream_completed_result_includes_aggregated_usage() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Hello",
                        usage=UsageMetadata(input_tokens=5, output_tokens=2, total_tokens=7),
                    )
                ),
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = [event async for event in agent.stream("Say hello.")]

    assert events[-1].type == "completed"
    assert events[-1].result is not None
    assert events[-1].result.usage == UsageMetadata(
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
    )
    assert events[-1].result.usage_by_response == [
        UsageMetadata(input_tokens=5, output_tokens=2, total_tokens=7)
    ]


@pytest.mark.asyncio
async def test_stream_aggregates_usage_after_tool_turn() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                        usage=UsageMetadata(input_tokens=9, output_tokens=3, total_tokens=12),
                    )
                )
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Done",
                        usage=UsageMetadata(input_tokens=7, output_tokens=4, total_tokens=11),
                    )
                )
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    events = [event async for event in agent.stream("Use ping.")]

    assert events[-1].type == "completed"
    assert events[-1].result is not None
    assert len(events[-1].result.usage_by_response) == 2
    assert events[-1].result.usage is not None
    assert events[-1].result.usage.input_tokens == 16
    assert events[-1].result.usage.output_tokens == 7
    assert events[-1].result.usage.total_tokens == 23


@pytest.mark.asyncio
async def test_stream_yields_tool_lifecycle_and_completed_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                        output_items=[
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                )
            ],
            [
                ProviderTextDeltaEvent(delta="Done"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Done",
                    )
                ),
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    events = [event async for event in agent.stream("Use ping.")]

    assert [event.type for event in events] == [
        "tool_call_started",
        "tool_call_completed",
        "text_delta",
        "completed",
    ]
    assert events[1].tool_result is not None
    assert events[1].tool_result.output == "pong: hello"
    assert events[-1].result is not None
    assert events[-1].result.output_text == "Done"


@pytest.mark.asyncio
async def test_stream_yields_tool_arguments_delta_before_tool_lifecycle() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderToolArgumentsDeltaEvent(
                    item_id="fc_1",
                    call_id="call_1",
                    name="ping",
                    delta='{"message":"hel',
                ),
                ProviderToolArgumentsDeltaEvent(
                    item_id="fc_1",
                    call_id="call_1",
                    name="ping",
                    delta='lo"}',
                ),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                        output_items=[
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                ),
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Done",
                    )
                ),
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    events = [event async for event in agent.stream("Use ping.")]

    assert [event.type for event in events] == [
        "tool_arguments_delta",
        "tool_arguments_delta",
        "tool_call_started",
        "tool_call_completed",
        "completed",
    ]
    argument_events = [event for event in events if event.type == "tool_arguments_delta"]
    assert events[2].tool_call is not None
    assert "".join(event.delta or "" for event in argument_events) == events[2].tool_call.raw_arguments
    assert [event.tool_item_id for event in argument_events] == ["fc_1", "fc_1"]
    assert [event.tool_name for event in argument_events] == ["ping", "ping"]


@pytest.mark.asyncio
async def test_stream_yields_web_search_call_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderHostedToolCallEvent(
                    type="hosted_tool_call_started",
                    item_id="ws_1",
                    tool_type="web_search_call",
                    status="in_progress",
                    item={
                        "id": "ws_1",
                        "type": "web_search_call",
                        "status": "in_progress",
                    },
                ),
                ProviderHostedToolCallEvent(
                    type="hosted_tool_call_updated",
                    item_id="ws_1",
                    tool_type="web_search_call",
                    status="searching",
                ),
                ProviderHostedToolCallEvent(
                    type="hosted_tool_call_completed",
                    item_id="ws_1",
                    tool_type="web_search_call",
                    status="completed",
                    item={
                        "id": "ws_1",
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "longevity clinic san diego",
                        },
                    },
                ),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Done",
                        output_items=[
                            {
                                "type": "web_search_call",
                                "id": "ws_1",
                                "status": "completed",
                            }
                        ],
                    )
                ),
            ]
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        hosted_tools=[{"type": "web_search"}],
    )

    events = [event async for event in agent.stream("Search the web.")]

    assert [event.type for event in events] == [
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
        "completed",
    ]
    search_events = events[:-1]
    assert [event.hosted_tool_call.item_id for event in search_events if event.hosted_tool_call is not None] == [
        "ws_1",
        "ws_1",
        "ws_1",
    ]
    assert [event.hosted_tool_call.tool_type for event in search_events if event.hosted_tool_call is not None] == [
        "web_search_call",
        "web_search_call",
        "web_search_call",
    ]
    assert [event.hosted_tool_call.status for event in search_events if event.hosted_tool_call is not None] == [
        "in_progress",
        "searching",
        "completed",
    ]
    assert search_events[0].hosted_tool_call is not None
    assert search_events[0].hosted_tool_call.item == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "in_progress",
    }
    assert search_events[1].hosted_tool_call is not None
    assert search_events[1].hosted_tool_call.item is None
    assert search_events[2].hosted_tool_call is not None
    assert search_events[2].hosted_tool_call.item == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "longevity clinic san diego",
        },
    }


@pytest.mark.asyncio
async def test_stream_tool_timeout_raises_after_started_event() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "very_slow_ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                )
            ]
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=0.01),
        tools=[very_slow_ping],
        provider=provider,
    )
    events = []

    with pytest.raises(
        ToolExecutionError,
        match="Tool 'very_slow_ping' timed out after 0.01 seconds.",
    ):
        async for event in agent.stream("Use the slow tool."):
            events.append(event)

    assert [event.type for event in events] == ["tool_call_started"]


def test_stream_sync_preserves_tool_lifecycle_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                )
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Done",
                    )
                )
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    events = list(agent.stream_sync("Use ping."))

    assert [event.type for event in events] == [
        "tool_call_started",
        "tool_call_completed",
        "completed",
    ]
    assert events[1].tool_result is not None
    assert events[1].tool_result.output == "pong: hello"


def test_stream_sync_tool_timeout_raises() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "very_slow_ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                )
            ]
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=0.01),
        tools=[very_slow_ping],
        provider=provider,
    )

    with pytest.raises(
        ToolExecutionError,
        match="Tool 'very_slow_ping' timed out after 0.01 seconds.",
    ):
        list(agent.stream_sync("Use the slow tool."))


def test_stream_sync_after_run_sync_reuses_the_same_agent_cleanly() -> None:
    class CombinedProvider:
        def __init__(self) -> None:
            self.create_calls: list[dict[str, Any]] = []

        async def create_response(self, **kwargs: Any) -> ProviderResponse:
            self.create_calls.append(kwargs)
            return ProviderResponse(
                output_text="hello world",
            )

        async def stream_response(
            self, **kwargs: Any
        ) -> AsyncIterator[ProviderEvent]:
            yield ProviderTextDeltaEvent(delta="Hel")
            yield ProviderTextDeltaEvent(delta="lo")
            yield ProviderCompletedEvent(
                response=ProviderResponse(
                    output_text="Hello",
                )
            )

        async def close(self) -> None:
            return None

    agent = Agent(config=AgentConfig(model="gpt-5"), provider=CombinedProvider())

    result = agent.run_sync("Say hello.")
    events = list(agent.stream_sync("Say hello again."))

    assert result.output_text == "hello world"
    assert [event.delta for event in events if event.type == "text_delta"] == ["Hel", "lo"]
    assert events[-1].type == "completed"


@pytest.mark.asyncio
async def test_stream_parallel_batch_emits_deterministic_lifecycle_events() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "slow_ping",
                                "arguments": {"message": "alpha"},
                                "raw_arguments": '{"message":"alpha"}',
                            },
                            {
                                "call_id": "call_2",
                                "name": "slow_ping",
                                "arguments": {"message": "beta"},
                                "raw_arguments": '{"message":"beta"}',
                            },
                        ],
                    )
                )
            ],
            [
                ProviderTextDeltaEvent(delta="done"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="done",
                    )
                ),
            ],
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping],
        provider=provider,
    )

    events = [event async for event in agent.stream("Use both slow tools.")]

    assert [event.type for event in events] == [
        "tool_call_started",
        "tool_call_started",
        "tool_call_completed",
        "tool_call_completed",
        "text_delta",
        "completed",
    ]
    assert events[0].tool_call is not None and events[0].tool_call.call_id == "call_1"
    assert events[1].tool_call is not None and events[1].tool_call.call_id == "call_2"
    assert events[2].tool_result is not None and events[2].tool_result.call_id == "call_1"
    assert events[3].tool_result is not None and events[3].tool_result.call_id == "call_2"


@pytest.mark.asyncio
async def test_stream_raises_on_provider_failure() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=ExplodingStreamingProvider([]),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        [event async for event in agent.stream("Fail.")]


def test_stream_sync_raises_on_provider_failure() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=ExplodingStreamingProvider([]),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        list(agent.stream_sync("Fail."))


def test_chat_session_stream_sync_preserves_history() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Stored.",
                        output_items=[
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "Stored."}],
                            }
                        ],
                    )
                )
            ],
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="You said Anson.",
                        output_items=[
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "You said Anson."}],
                            }
                        ],
                    )
                )
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    _ = list(chat.stream_sync("My name is Anson."))
    events = list(chat.stream_sync("What name did I say?"))

    assert events[-1].type == "completed"
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
        ChatMessage(role="user", content="What name did I say?"),
        ChatMessage(role="assistant", content="You said Anson."),
    ]


@pytest.mark.asyncio
async def test_snapshot_after_streaming_completion_includes_completed_turn() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="Stored.",
                        output_items=[
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "Stored."}],
                            }
                        ],
                    )
                )
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="You are concise.")

    _ = [event async for event in chat.stream("My name is Anson.")]
    snapshot = chat.snapshot()

    assert snapshot == ChatSnapshot(
        items=[
            {
                "type": "message",
                "role": "user",
                "content": "My name is Anson.",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Stored."}],
            },
        ],
        system_prompt="You are concise.",
    )


@pytest.mark.asyncio
async def test_restored_chat_continues_correctly_after_prior_streamed_turns() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text="You said Anson.",
                        output_items=[
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "You said Anson."}],
                            }
                        ],
                    )
                )
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "My name is Anson.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored."}],
                },
            ],
            "system_prompt": "You are concise.",
        }
    )

    events = [event async for event in restored.stream("What name did I say?")]

    assert events[-1].type == "completed"
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What name did I say?",
        },
    ]


@pytest.mark.asyncio
async def test_stream_returns_structured_output_on_completed_event() -> None:
    summary = Summary(title="Hello", bullets=["one", "two"])
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderTextDeltaEvent(delta="{"),
                ProviderTextDeltaEvent(delta='"title":"Hello"}'),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text='{"title":"Hello","bullets":["one","two"]}',
                        output_data=summary,
                    )
                ),
            ]
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    events = [
        event
        async for event in agent.stream(
            "Summarize this text.",
            response_model=Summary,
        )
    ]

    assert [event.delta for event in events if event.type == "text_delta"] == ["{", '"title":"Hello"}']
    assert events[-1].type == "completed"
    assert events[-1].result is not None
    assert events[-1].result.output_data == summary
    assert provider.calls[0]["response_model"] is Summary


@pytest.mark.asyncio
async def test_stream_returns_structured_output_after_tool_turn() -> None:
    summary = Summary(title="Weather", bullets=["Foggy", "65F"])
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "ping",
                                "arguments": {"message": "hello"},
                                "raw_arguments": '{"message":"hello"}',
                            }
                        ],
                    )
                )
            ],
            [
                ProviderTextDeltaEvent(delta="done"),
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        output_text='{"title":"Weather","bullets":["Foggy","65F"]}',
                        output_data=summary,
                    )
                ),
            ],
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    events = [
        event
        async for event in agent.stream(
            "Use the tool and summarize the result.",
            response_model=Summary,
        )
    ]

    assert [event.type for event in events] == [
        "tool_call_started",
        "tool_call_completed",
        "text_delta",
        "completed",
    ]
    assert events[-1].result is not None
    assert events[-1].result.output_data == summary
    assert provider.calls[0]["response_model"] is Summary
    assert provider.calls[1]["response_model"] is Summary


@pytest.mark.asyncio
async def test_stream_parallel_tool_failure_raises() -> None:
    provider = FakeProvider(stream_sequences=
        [
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "slow_ping",
                                "arguments": {"message": "alpha"},
                                "raw_arguments": '{"message":"alpha"}',
                            },
                            {
                                "call_id": "call_2",
                                "name": "explode",
                                "arguments": {"message": "boom"},
                                "raw_arguments": '{"message":"boom"}',
                            },
                        ],
                    )
                )
            ]
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping, explode],
        provider=provider,
    )

    with pytest.raises(ToolExecutionError, match="boom"):
        [event async for event in agent.stream("Fail with parallel tools.")]
