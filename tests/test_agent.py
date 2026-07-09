from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from conftest import FakeProvider
from simple_agent_base import Agent, AgentConfig, ChatMessage, FilePart, ImagePart, TextPart, UsageMetadata, tool
from simple_agent_base.errors import (
    MaxTurnsExceededError,
    ProviderError,
    ToolExecutionError,
    ToolRegistrationError,
)
from simple_agent_base.providers.base import ProviderResponse
from simple_agent_base.types import ConversationItem


class ClosableFakeProvider(FakeProvider):
    def __init__(self, responses: list[ProviderResponse]) -> None:
        super().__init__(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


@tool
async def uppercase(value: str) -> str:
    """Uppercase a value."""
    return value.upper()


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
def sync_ping(message: str) -> str:
    """Echo a message back synchronously."""
    return f"sync-pong: {message}"


@tool
def very_slow_sync_ping(message: str) -> str:
    """Echo a message back synchronously after a longer delay."""
    time.sleep(0.2)
    return f"sync-pong: {message}"


@tool
def sync_explode(message: str) -> str:
    """Always fail synchronously."""
    raise ValueError(message)


@tool
async def explode(message: str) -> str:
    """Always fail."""
    raise ValueError(message)


class Person(BaseModel):
    name: str
    age: int


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


REASONING_PROMPT = "Think carefully, then answer with exactly reasoning-ok."
REASONING_SUMMARY = "I checked the constraint and kept the answer exact."


def make_reasoning_response(*, reasoning_summary: str | None) -> ProviderResponse:
    return ProviderResponse(
        output_text="reasoning-ok",
        reasoning_summary=reasoning_summary,
    )


@pytest.mark.asyncio
async def test_run_without_tools_returns_plain_text() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello world"}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run("Say hello.")

    assert result.output_text == "hello world"
    assert result.response_id == "resp_1"
    assert result.raw_responses == [{"id": "resp_1"}]
    assert result.output_data is None
    assert result.tool_results == []


@pytest.mark.asyncio
async def test_run_returns_reasoning_summary_when_provider_supplies_it() -> None:
    provider = FakeProvider([make_reasoning_response(reasoning_summary=REASONING_SUMMARY)])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(REASONING_PROMPT)

    assert result.output_text == "reasoning-ok"
    assert result.reasoning_summary == REASONING_SUMMARY


@pytest.mark.asyncio
async def test_run_returns_none_usage_when_provider_omits_usage() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="hello world",
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run("Say hello.")

    assert result.usage is None
    assert result.usage_by_response == []


@pytest.mark.asyncio
async def test_run_aggregates_usage_across_tool_turns() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                usage=UsageMetadata(input_tokens=10, output_tokens=2, total_tokens=12),
            ),
            ProviderResponse(
                output_text="done",
                usage=UsageMetadata(input_tokens=8, output_tokens=4, total_tokens=12),
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run("Use ping.")

    assert result.output_text == "done"
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: hello"]
    assert len(result.usage_by_response) == 2
    assert result.usage is not None
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 24
    assert result.usage.input_tokens_details is None
    assert result.usage.output_tokens_details is None
    assert result.usage.raw is None


@pytest.mark.asyncio
async def test_run_aggregates_partial_usage_fields() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                usage=UsageMetadata(input_tokens=10),
            ),
            ProviderResponse(
                output_text="done",
                usage=UsageMetadata(output_tokens=4),
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run("Say hello.")

    assert len(result.usage_by_response) == 2
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens is None


@pytest.mark.asyncio
async def test_run_returns_none_reasoning_summary_when_provider_supplies_none() -> None:
    provider = FakeProvider([make_reasoning_response(reasoning_summary=None)])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(REASONING_PROMPT)

    assert result.output_text == "reasoning-ok"
    assert result.reasoning_summary is None


def test_run_sync_returns_plain_text() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="hello world",
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = agent.run_sync("Say hello.")

    assert result.output_text == "hello world"


def test_close_wraps_aclose() -> None:
    provider = ClosableFakeProvider([])
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    agent.close()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_provider() -> None:
    provider = ClosableFakeProvider([])

    async with Agent(config=AgentConfig(model="gpt-5"), provider=provider) as agent:
        assert isinstance(agent, Agent)
        assert provider.closed is False

    assert provider.closed is True


def test_sync_context_manager_closes_provider() -> None:
    provider = ClosableFakeProvider([])

    with Agent(config=AgentConfig(model="gpt-5"), provider=provider) as agent:
        assert isinstance(agent, Agent)
        assert provider.closed is False

    assert provider.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exception() -> None:
    provider = ClosableFakeProvider([])

    with pytest.raises(RuntimeError, match="boom"):
        async with Agent(config=AgentConfig(model="gpt-5"), provider=provider):
            raise RuntimeError("boom")

    assert provider.closed is True


@pytest.mark.asyncio
async def test_run_sync_raises_inside_running_event_loop() -> None:
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=FakeProvider([]))

    with pytest.raises(RuntimeError, match="run_sync\\(\\) cannot be used inside a running event loop"):
        agent.run_sync("Say hello.")


def test_chat_session_run_sync_preserves_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
            ),
            ProviderResponse(
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    first = chat.run_sync("My name is Anson.")
    second = chat.run_sync("What name did I say?")

    assert first.output_text == "Stored."
    assert second.output_text == "You said Anson."
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
        ChatMessage(role="user", content="What name did I say?"),
        ChatMessage(role="assistant", content="You said Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_snapshot_returns_full_session_state() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="You are concise.")

    await chat.run("My name is Anson.")
    snapshot = chat.snapshot()

    assert snapshot.version == "v1"
    assert snapshot.system_prompt == "You are concise."
    assert snapshot.items == [
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
    ]


@pytest.mark.asyncio
async def test_chat_from_snapshot_restores_conversation_state() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
            )
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

    result = await restored.run("What name did I say?")

    assert result.output_text == "You said Anson."
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
async def test_restored_chat_uses_snapshot_system_prompt_over_agent_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Done.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Agent default prompt",
    )
    restored = agent.chat_from_snapshot(
        {
            "version": "v1",
            "items": [],
            "system_prompt": "Snapshot prompt",
        }
    )

    await restored.run("Hello.")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Snapshot prompt",
    }


@pytest.mark.asyncio
async def test_snapshot_does_not_persist_convenience_prompt_as_message_item() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="You are concise.")

    await chat.run("My name is Anson.")
    snapshot = chat.snapshot()

    assert all(item.get("role") != "developer" for item in snapshot.items)


@pytest.mark.asyncio
async def test_chat_from_snapshot_preserves_multimodal_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="The image showed a cat.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The image showed a cat."}],
                    }
                ],
            )
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
                    "content": [
                        {"type": "input_text", "text": "Remember this image."},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.png",
                            "detail": "high",
                        },
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored the image."}],
                },
            ],
            "system_prompt": None,
        }
    )

    result = await restored.run("What was in the image?")

    assert result.output_text == "The image showed a cat."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the image."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What was in the image?",
        },
    ]


@pytest.mark.asyncio
async def test_chat_from_snapshot_preserves_file_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="The file mentioned teal.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The file mentioned teal."}],
                    }
                ],
            )
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
                    "content": [
                        {"type": "input_text", "text": "Remember this PDF."},
                        {
                            "type": "input_file",
                            "file_url": "https://example.com/report.pdf",
                        },
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Stored the file."}],
                },
            ],
        }
    )

    result = await restored.run("What did the file mention?")

    assert result.output_text == "The file mentioned teal."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this PDF."},
                {
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the file."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What did the file mention?",
        },
    ]


@pytest.mark.asyncio
async def test_run_prepends_agent_level_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="hello world",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="You are concise.",
    )

    await agent.run("Say hello.")

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Say hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_level_system_prompt_overrides_agent_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="hello world",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Default prompt",
    )

    await agent.run("Say hello.", system_prompt="Override prompt")

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "Override prompt",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Say hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_accepts_multimodal_message() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="That looks like a cat.",
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("What is in this image?"),
                    ImagePart.from_url("https://example.com/cat.png"),
                ],
            )
        ]
    )

    assert result.output_text == "That looks like a cat."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "auto",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_accepts_file_message() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="The file says hello.",
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("What does this file say?"),
                    FilePart.from_url("https://example.com/report.pdf"),
                ],
            )
        ]
    )

    assert result.output_text == "The file says hello."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What does this file say?"},
                {
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_accepts_multiple_messages() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="I remember the earlier messages.",
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="My name is Anson."),
            ChatMessage(role="assistant", content="Noted."),
            ChatMessage(role="user", content="What's my name?"),
        ]
    )

    assert result.output_text == "I remember the earlier messages."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "system",
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
            "content": "Noted.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "What's my name?",
        },
    ]


@pytest.mark.asyncio
async def test_run_preserves_explicit_history_alongside_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="I remember the earlier messages.",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="Default prompt",
    )

    await agent.run(
        [
            ChatMessage(role="system", content="Explicit system."),
            ChatMessage(role="user", content="Hello."),
        ]
    )

    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "developer",
            "content": "Default prompt",
        },
        {
            "type": "message",
            "role": "system",
            "content": "Explicit system.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "Hello.",
        },
    ]


@pytest.mark.asyncio
async def test_run_executes_one_tool_then_returns_final_response() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="",
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
            ),
            ProviderResponse(
                output_text="The tool said pong: hello",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The tool said pong: hello"}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run("Use the tool.")

    assert result.output_text == "The tool said pong: hello"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output == "pong: hello"
    assert provider.calls[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "pong: hello",
    }


@pytest.mark.asyncio
async def test_run_executes_multiple_sequential_tool_calls() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": {"value": "world"},
                        "raw_arguments": '{"value":"world"}',
                    },
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": '{"value":"world"}',
                    },
                ],
            ),
            ProviderResponse(
                output_text="pong: hello / WORLD",
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping, uppercase], provider=provider)

    result = await agent.run("Use both tools.")

    assert [tool_result.output for tool_result in result.tool_results] == ["pong: hello", "WORLD"]
    second_turn_items = provider.calls[1]["input_items"]
    assert second_turn_items[-2:] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "pong: hello"},
        {"type": "function_call_output", "call_id": "call_2", "output": "WORLD"},
    ]


@pytest.mark.asyncio
async def test_parallel_tool_batch_executes_concurrently_when_enabled() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
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
            ),
            ProviderResponse(
                output_text="done",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping],
        provider=provider,
    )

    start = time.perf_counter()
    result = await agent.run("Use the slow tool twice.")
    elapsed = time.perf_counter() - start

    assert result.output_text == "done"
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert provider.calls[1]["input_items"][-2:] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "pong: alpha"},
        {"type": "function_call_output", "call_id": "call_2", "output": "pong: beta"},
    ]
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_parallel_tool_batch_remains_sequential_by_default() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
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
            ),
            ProviderResponse(
                output_text="done",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        tools=[slow_ping],
        provider=provider,
    )

    start = time.perf_counter()
    result = await agent.run("Use the slow tool twice.")
    elapsed = time.perf_counter() - start

    assert result.output_text == "done"
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert elapsed >= 0.09


@pytest.mark.asyncio
async def test_tool_timeout_raises_for_slow_async_tool() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "very_slow_ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            )
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
        await agent.run("Use the slow tool.")


@pytest.mark.asyncio
async def test_no_tool_timeout_preserves_slow_async_tool_behavior() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "slow_ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            ),
            ProviderResponse(
                output_text="done",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=None),
        tools=[slow_ping],
        provider=provider,
    )

    result = await agent.run("Use the slow tool.")

    assert result.output_text == "done"
    assert result.tool_results[0].output == "pong: hello"


@pytest.mark.asyncio
async def test_tool_timeout_raises_for_slow_sync_tool() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "very_slow_sync_ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=0.01),
        tools=[very_slow_sync_ping],
        provider=provider,
    )

    with pytest.raises(
        ToolExecutionError,
        match="Tool 'very_slow_sync_ping' timed out after 0.01 seconds.",
    ):
        await agent.run("Use the slow sync tool.")


@pytest.mark.asyncio
async def test_parallel_tool_batch_timeout_fails_run() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "very_slow_ping",
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
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True, tool_timeout=0.01),
        tools=[very_slow_ping, slow_ping],
        provider=provider,
    )

    with pytest.raises(ToolExecutionError, match="Tool 'very_slow_ping' timed out"):
        await agent.run("Use the slow tools.")


@pytest.mark.asyncio
async def test_max_turns_exceeded_raises_error() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "again"},
                        "raw_arguments": '{"message":"again"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", max_turns=1),
        tools=[ping],
        provider=provider,
    )

    with pytest.raises(MaxTurnsExceededError):
        await agent.run("Loop forever.")


@pytest.mark.asyncio
async def test_tool_errors_surface_as_tool_execution_errors() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[explode], provider=provider)

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail.")


@pytest.mark.asyncio
async def test_sync_tool_definition_executes_successfully() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "sync_ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            ),
            ProviderResponse(
                output_text="done",
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[sync_ping], provider=provider)

    result = await agent.run("Use the sync tool.")

    assert [tool_result.output for tool_result in result.tool_results] == ["sync-pong: hello"]


@pytest.mark.asyncio
async def test_sync_tool_errors_surface_as_tool_execution_errors() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "sync_explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[sync_explode], provider=provider)

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail with sync tool.")


@pytest.mark.asyncio
async def test_run_returns_structured_output_without_tools() -> None:
    person = Person(name="Sarah", age=29)
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text='{"name":"Sarah","age":29}',
                output_data=person,
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        "Extract the person from: Sarah is 29 years old.",
        response_model=Person,
    )

    assert result.output_data == person
    assert result.output_text == '{"name":"Sarah","age":29}'
    assert provider.calls[0]["response_model"] is Person


@pytest.mark.asyncio
async def test_run_supports_structured_output_with_image_input() -> None:
    person = Person(name="Sarah", age=29)
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text='{"name":"Sarah","age":29}',
                output_data=person,
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("Extract the person in this image."),
                    ImagePart.from_url("https://example.com/person.png"),
                ],
            )
        ],
        response_model=Person,
    )

    assert result.output_data == person
    assert provider.calls[0]["response_model"] is Person
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract the person in this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/person.png",
                    "detail": "auto",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_returns_structured_output_after_tool_call() -> None:
    weather = WeatherAnswer(city="San Francisco", temperature_f=65, summary="Foggy")
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "weather"},
                        "raw_arguments": '{"message":"weather"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"weather"}',
                    }
                ],
            ),
            ProviderResponse(
                output_text='{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
                output_data=weather,
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run(
        "Use the tool and return a structured weather answer.",
        response_model=WeatherAnswer,
    )

    assert result.output_data == weather
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: weather"]
    assert provider.calls[0]["response_model"] is WeatherAnswer
    assert provider.calls[1]["response_model"] is WeatherAnswer


@pytest.mark.asyncio
async def test_parallel_tool_batch_supports_structured_output_after_tools() -> None:
    weather = WeatherAnswer(city="San Francisco", temperature_f=65, summary="Foggy")
    provider = FakeProvider(
        [
            ProviderResponse(
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
            ),
            ProviderResponse(
                output_text='{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
                output_data=weather,
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping],
        provider=provider,
    )

    result = await agent.run(
        "Use the tools and return a structured weather answer.",
        response_model=WeatherAnswer,
    )

    assert result.output_data == weather
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: alpha", "pong: beta"]
    assert provider.calls[0]["response_model"] is WeatherAnswer
    assert provider.calls[1]["response_model"] is WeatherAnswer


@pytest.mark.asyncio
async def test_parallel_tool_batch_failure_fails_run() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
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
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", parallel_tool_calls=True),
        tools=[slow_ping, explode],
        provider=provider,
    )

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail with parallel tools.")


@pytest.mark.asyncio
async def test_run_supports_tool_calls_with_system_prompt() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="",
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
            ),
            ProviderResponse(
                output_text="The tool said pong: hello",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        tools=[ping],
        provider=provider,
        system_prompt="Use tools when helpful.",
    )

    result = await agent.run("Use the tool.")

    assert result.output_text == "The tool said pong: hello"
    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Use tools when helpful.",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Use tools when helpful.",
    }


@pytest.mark.asyncio
async def test_run_surfaces_structured_provider_failures() -> None:
    class FailingStructuredProvider(FakeProvider):
        async def create_response(
            self,
            *,
            input_items: Sequence[ConversationItem],
            tools: Sequence[dict[str, Any]],
            response_model: type[BaseModel] | None = None,
        ) -> ProviderResponse:
            raise ProviderError("structured parse failed")

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=FailingStructuredProvider([]),
    )

    with pytest.raises(ProviderError, match="structured parse failed"):
        await agent.run("Return structured data.", response_model=Person)


@pytest.mark.asyncio
async def test_chat_session_preserves_conversation_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Your name is Anson."}],
                    }
                ],
            ),
            ProviderResponse(
                output_text="You told me your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You told me your name is Anson."}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    first = await chat.run("My name is Anson.")
    second = await chat.run("What name did I give you?")

    assert first.output_text == "Your name is Anson."
    assert second.output_text == "You told me your name is Anson."
    assert provider.calls[1]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Your name is Anson."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What name did I give you?",
        },
    ]
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Your name is Anson."),
        ChatMessage(role="user", content="What name did I give you?"),
        ChatMessage(role="assistant", content="You told me your name is Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_session_uses_default_system_prompt_without_leaking_into_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Stored.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored."}],
                    }
                ],
            ),
            ProviderResponse(
                output_text="You said Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You said Anson."}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        system_prompt="You are concise.",
    )
    chat = agent.chat()

    await chat.run("My name is Anson.")
    await chat.run("What name did I say?")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "You are concise.",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "You are concise.",
    }
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
        ChatMessage(role="user", content="What name did I say?"),
        ChatMessage(role="assistant", content="You said Anson."),
    ]


@pytest.mark.asyncio
async def test_chat_session_run_level_system_prompt_overrides_for_one_call_only() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="First",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First"}],
                    }
                ],
            ),
            ProviderResponse(
                output_text="Second",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second"}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat(system_prompt="Session prompt")

    await chat.run("Hello.", system_prompt="Override prompt")
    await chat.run("Hello again.")

    assert provider.calls[0]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Override prompt",
    }
    assert provider.calls[1]["input_items"][0] == {
        "type": "message",
        "role": "developer",
        "content": "Session prompt",
    }


@pytest.mark.asyncio
async def test_chat_session_preserves_multimodal_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Stored the image.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Stored the image."}],
                    }
                ],
            ),
            ProviderResponse(
                output_text="The image showed a cat.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The image showed a cat."}],
                    }
                ],
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    await chat.run(
        [
            ChatMessage(
                role="user",
                content=[
                    TextPart("Remember this image."),
                    ImagePart.from_url("https://example.com/cat.png", detail="high"),
                ],
            )
        ]
    )
    second = await chat.run("What was in the image?")

    assert second.output_text == "The image showed a cat."
    assert provider.calls[1]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Remember this image."},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high",
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stored the image."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What was in the image?",
        },
    ]
    assert chat.history == [
        ChatMessage(
            role="user",
            content=[
                TextPart("Remember this image."),
                ImagePart.from_url("https://example.com/cat.png", detail="high"),
            ],
        ),
        ChatMessage(role="assistant", content="Stored the image."),
        ChatMessage(role="user", content="What was in the image?"),
        ChatMessage(role="assistant", content="The image showed a cat."),
    ]


def test_image_part_from_file_creates_data_url(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")

    part = ImagePart.from_file(str(image_path), detail="high")

    assert part.detail == "high"
    assert part.image_url.startswith("data:image/png;base64,")


def test_file_part_from_file_creates_data_url(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world")

    part = FilePart.from_file(str(file_path))

    assert part.filename == "sample.txt"
    assert part.file_data is not None
    assert part.file_data.startswith("data:text/plain;base64,")


def test_file_part_from_file_rejects_missing_paths() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        FilePart.from_file("missing.pdf")


def test_file_part_from_file_rejects_unsupported_types(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"zip-data")

    with pytest.raises(ValueError, match="Unsupported file type"):
        FilePart.from_file(str(file_path))


def test_image_part_from_file_rejects_missing_paths() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ImagePart.from_file("missing.png")


def test_image_part_from_file_rejects_unsupported_types(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image file type"):
        ImagePart.from_file(str(file_path))


@pytest.mark.asyncio
async def test_hosted_tools_are_passed_through_to_provider() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="search done",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "search done"}],
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        hosted_tools=[{"type": "web_search"}],
    )

    result = await agent.run("Search the web.")

    assert result.output_text == "search done"
    assert provider.calls[0]["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_hosted_tools_coexist_with_local_tools() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="ok",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        tools=[ping],
        provider=provider,
        hosted_tools=[{"type": "web_search"}, {"type": "code_interpreter"}],
    )

    await agent.run("Hi.")

    sent_tools = provider.calls[0]["tools"]
    assert len(sent_tools) == 3
    # Local function tool comes first, hosted tools come last (per _build_tool_params order).
    assert sent_tools[0]["name"] == "ping"
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[1] == {"type": "web_search"}
    assert sent_tools[2] == {"type": "code_interpreter"}


def test_hosted_tools_reject_non_dict_entries() -> None:
    with pytest.raises(ToolRegistrationError, match="must be a dict"):
        Agent(
            config=AgentConfig(model="gpt-5"),
            provider=FakeProvider([]),
            hosted_tools=["web_search"],  # type: ignore[list-item]
        )


def test_hosted_tools_reject_entries_missing_type() -> None:
    with pytest.raises(ToolRegistrationError, match="non-empty string 'type'"):
        Agent(
            config=AgentConfig(model="gpt-5"),
            provider=FakeProvider([]),
            hosted_tools=[{"foo": "bar"}],
        )


def test_hosted_tools_reject_entries_with_empty_type() -> None:
    with pytest.raises(ToolRegistrationError, match="non-empty string 'type'"):
        Agent(
            config=AgentConfig(model="gpt-5"),
            provider=FakeProvider([]),
            hosted_tools=[{"type": ""}],
        )


def test_hosted_tools_default_is_empty_list() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=FakeProvider([]),
    )

    assert agent.hosted_tools == []


@pytest.mark.asyncio
async def test_hosted_tool_only_response_terminates_loop_cleanly() -> None:
    """A response containing only hosted tool output items (no function_call)
    should end the loop after one turn and return the final text."""
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="Python 3.13 added free-threaded mode.",
                output_items=[
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Python 3.13 added free-threaded mode.",
                            }
                        ],
                    },
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        hosted_tools=[{"type": "web_search"}],
    )

    result = await agent.run("What is new in Python 3.13?")

    assert result.output_text == "Python 3.13 added free-threaded mode."
    assert result.tool_results == []
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_hosted_tools_dicts_are_copied_not_shared() -> None:
    """Mutating the original dict after construction must not change what gets sent."""
    original = {"type": "web_search"}
    provider = FakeProvider(
        [
            ProviderResponse(
                output_text="ok",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        hosted_tools=[original],
    )

    original["type"] = "file_search"
    await agent.run("Hi.")

    assert provider.calls[0]["tools"] == [{"type": "web_search"}]
