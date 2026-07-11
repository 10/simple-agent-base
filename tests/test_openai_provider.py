from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from simple_agent_base import AgentConfig
from simple_agent_base.config import ReasoningEffort
from simple_agent_base.providers.openai import OpenAIResponsesProvider


class FakeSummaryPart(BaseModel):
    text: str
    type: str = "summary_text"


class FakeReasoningItem(BaseModel):
    id: str = "rs_1"
    type: str = "reasoning"
    summary: list[FakeSummaryPart]


class FakeOutputTextItem(BaseModel):
    type: str = "message"
    role: str = "assistant"
    content: list[dict[str, str]]


class FakeResponse(BaseModel):
    id: str = "resp_1"
    output_text: str = ""
    output: list[BaseModel]
    output_parsed: BaseModel | None = None


class FakeUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    input_tokens_details: dict[str, object] | None = None
    output_tokens: int | None = None
    output_tokens_details: dict[str, object] | None = None
    total_tokens: int | None = None


class FakeResponseWithUsage(FakeResponse):
    usage: FakeUsage | dict[str, object] | None = None


class FakeStream:
    def __init__(self, events: list[object], final_response: BaseModel) -> None:
        self._events = list(events)
        self._final_response = final_response

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event

    async def get_final_response(self) -> BaseModel:
        return self._final_response


def fake_client(stream: FakeStream | None = None) -> SimpleNamespace:
    def stream_response(**_: Any) -> FakeStream:
        if stream is None:
            raise AssertionError("No fake stream configured.")
        return stream

    async def close() -> None:
        return None

    return SimpleNamespace(
        responses=SimpleNamespace(stream=stream_response),
        close=close,
    )


REASONING_SUMMARY = "Checked the constraint."


def make_provider(*, reasoning_effort: ReasoningEffort | None = None) -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(
        AgentConfig(
            model="gpt-5.6-sol",
            api_key="test-key",
            reasoning_effort=reasoning_effort,
        )
    )


def make_reasoning_response(*, summary_text: str = REASONING_SUMMARY) -> FakeResponse:
    return FakeResponse(
        output_text="reasoning-ok",
        output=[
            FakeReasoningItem(summary=[FakeSummaryPart(text=summary_text)]),
            FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}]),
        ],
    )


def test_request_kwargs_include_reasoning_when_reasoning_effort_is_set() -> None:
    provider = make_provider(reasoning_effort="high")

    kwargs = provider._request_kwargs([], [])

    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}


def test_request_kwargs_omit_reasoning_when_reasoning_effort_is_none() -> None:
    provider = make_provider()

    kwargs = provider._request_kwargs([], [])

    assert "reasoning" not in kwargs


def test_convert_response_extracts_reasoning_summary_from_reasoning_item() -> None:
    provider = make_provider()
    response = FakeResponse(
        output_text="reasoning-ok",
        output=[
            FakeReasoningItem(summary=[FakeSummaryPart(text="First part. "), FakeSummaryPart(text="Second part.")]),
            FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}]),
        ],
    )

    converted = provider._convert_response(response)

    assert converted.output_text == "reasoning-ok"
    assert converted.reasoning_summary == "First part.Second part."


def test_convert_response_returns_none_when_no_reasoning_item_exists() -> None:
    provider = make_provider()
    response = FakeResponse(
        output_text="reasoning-ok",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}])],
    )

    converted = provider._convert_response(response)

    assert converted.reasoning_summary is None


def test_convert_response_extracts_usage_metadata() -> None:
    provider = make_provider()
    response = FakeResponseWithUsage(
        output_text="hello",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "hello"}])],
        usage=FakeUsage(
            input_tokens=10,
            input_tokens_details={"cached_tokens": 3},
            output_tokens=5,
            output_tokens_details={"reasoning_tokens": 2},
            total_tokens=15,
        ),
    )

    converted = provider._convert_response(response)

    assert converted.usage is not None
    assert converted.usage.input_tokens == 10
    assert converted.usage.input_tokens_details == {"cached_tokens": 3}
    assert converted.usage.output_tokens == 5
    assert converted.usage.output_tokens_details == {"reasoning_tokens": 2}
    assert converted.usage.total_tokens == 15
    assert converted.usage.raw == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 15,
    }


def test_convert_response_allows_missing_usage() -> None:
    provider = make_provider()
    response = FakeResponse(
        output_text="hello",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "hello"}])],
    )

    converted = provider._convert_response(response)

    assert converted.usage is None


def test_convert_response_tolerates_partial_or_provider_specific_usage() -> None:
    provider = make_provider()
    response = FakeResponseWithUsage(
        output_text="hello",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "hello"}])],
        usage={"input_tokens": 12, "provider_units": 4, "total_tokens": "not-an-int"},
    )

    converted = provider._convert_response(response)

    assert converted.usage is not None
    assert converted.usage.input_tokens == 12
    assert converted.usage.output_tokens is None
    assert converted.usage.total_tokens is None
    assert converted.usage.raw == {
        "input_tokens": 12,
        "provider_units": 4,
        "total_tokens": "not-an-int",
    }


@pytest.mark.asyncio
async def test_stream_response_emits_reasoning_delta_and_completed_summary() -> None:
    provider = make_provider(reasoning_effort="high")
    final_response = make_reasoning_response()
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Checked ", item_id="rs_1", summary_index=0),
                SimpleNamespace(type="response.reasoning_summary_text.delta", delta="the constraint.", item_id="rs_1", summary_index=0),
                SimpleNamespace(type="response.output_text.delta", delta="reasoning-ok"),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]

    assert [event.delta for event in events if event.type == "reasoning_delta"] == [
        "Checked ",
        "the constraint.",
    ]
    assert [event.delta for event in events if event.type == "text_delta"] == ["reasoning-ok"]
    assert events[-1].type == "completed"
    assert events[-1].response.reasoning_summary == REASONING_SUMMARY


@pytest.mark.asyncio
async def test_stream_response_emits_tool_arguments_delta_events() -> None:
    provider = make_provider()
    final_response = FakeResponse(
        output=[
            FakeOutputTextItem(content=[{"type": "output_text", "text": ""}]),
        ],
    )
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        id="fc_1",
                        type="function_call",
                        call_id="call_1",
                        name="search",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="fc_1",
                    delta='{"query":"san ',
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="fc_1",
                    delta='francisco"}',
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]
    argument_events = [event for event in events if event.type == "tool_arguments_delta"]

    assert [event.item_id for event in argument_events] == ["fc_1", "fc_1"]
    assert [event.call_id for event in argument_events] == ["call_1", "call_1"]
    assert [event.name for event in argument_events] == ["search", "search"]
    assert "".join(event.delta for event in argument_events) == '{"query":"san francisco"}'


@pytest.mark.asyncio
async def test_stream_response_emits_web_search_call_events() -> None:
    provider = make_provider()
    final_response = FakeResponse(
        output=[
            FakeOutputTextItem(content=[{"type": "output_text", "text": "done"}]),
        ],
    )
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.output_item.added",
                    output_index=0,
                    sequence_number=1,
                    item=SimpleNamespace(
                        id="ws_1",
                        type="web_search_call",
                        status="in_progress",
                    ),
                ),
                SimpleNamespace(
                    type="response.web_search_call.in_progress",
                    item_id="ws_1",
                    output_index=0,
                    sequence_number=2,
                ),
                SimpleNamespace(
                    type="response.web_search_call.searching",
                    item_id="ws_1",
                    output_index=0,
                    sequence_number=3,
                ),
                SimpleNamespace(
                    type="response.web_search_call.completed",
                    item_id="ws_1",
                    output_index=0,
                    sequence_number=4,
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    output_index=0,
                    sequence_number=5,
                    item=SimpleNamespace(
                        id="ws_1",
                        type="web_search_call",
                        status="completed",
                        action={
                            "type": "search",
                            "query": "longevity clinic san diego",
                        },
                    ),
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]

    assert [event.type for event in events] == [
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
        "completed",
    ]
    search_events = [event for event in events if event.type != "completed"]
    assert [event.item_id for event in search_events] == ["ws_1", "ws_1", "ws_1"]
    assert [event.tool_type for event in search_events] == ["web_search_call", "web_search_call", "web_search_call"]
    assert [event.status for event in search_events] == ["in_progress", "searching", "completed"]
    assert search_events[0].item == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "in_progress",
    }
    assert search_events[1].item is None
    assert search_events[2].item == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "longevity clinic san diego",
        },
    }


@pytest.mark.asyncio
async def test_stream_response_emits_generic_hosted_tool_events_for_file_search() -> None:
    provider = make_provider()
    final_response = FakeResponse(
        output=[
            FakeOutputTextItem(content=[{"type": "output_text", "text": "done"}]),
        ],
    )
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.output_item.added",
                    output_index=0,
                    sequence_number=1,
                    item=SimpleNamespace(
                        id="fs_1",
                        type="file_search_call",
                        status="in_progress",
                    ),
                ),
                SimpleNamespace(
                    type="response.file_search_call.searching",
                    item_id="fs_1",
                    output_index=0,
                    sequence_number=2,
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    output_index=0,
                    sequence_number=3,
                    item=SimpleNamespace(
                        id="fs_1",
                        type="file_search_call",
                        status="completed",
                    ),
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]
    hosted_events = [event for event in events if event.type != "completed"]

    assert [event.type for event in hosted_events] == [
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
    ]
    assert [event.tool_type for event in hosted_events] == [
        "file_search_call",
        "file_search_call",
        "file_search_call",
    ]
    assert [event.status for event in hosted_events] == [
        "in_progress",
        "searching",
        "completed",
    ]
    assert hosted_events[0].item == {
        "id": "fs_1",
        "type": "file_search_call",
        "status": "in_progress",
    }
    assert hosted_events[1].item is None
    assert hosted_events[2].item == {
        "id": "fs_1",
        "type": "file_search_call",
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_stream_response_emits_generic_hosted_tool_events_for_code_interpreter() -> None:
    provider = make_provider()
    final_response = FakeResponse(
        output=[
            FakeOutputTextItem(content=[{"type": "output_text", "text": "done"}]),
        ],
    )
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.output_item.added",
                    output_index=0,
                    sequence_number=1,
                    item=SimpleNamespace(
                        id="ci_1",
                        type="code_interpreter_call",
                        status="in_progress",
                    ),
                ),
                SimpleNamespace(
                    type="response.code_interpreter_call.interpreting",
                    item_id="ci_1",
                    output_index=0,
                    sequence_number=2,
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    output_index=0,
                    sequence_number=3,
                    item=SimpleNamespace(
                        id="ci_1",
                        type="code_interpreter_call",
                        status="completed",
                    ),
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]
    hosted_events = [event for event in events if event.type != "completed"]

    assert [event.type for event in hosted_events] == [
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
    ]
    assert [event.tool_type for event in hosted_events] == [
        "code_interpreter_call",
        "code_interpreter_call",
        "code_interpreter_call",
    ]
    assert [event.status for event in hosted_events] == [
        "in_progress",
        "interpreting",
        "completed",
    ]
    assert hosted_events[0].item == {
        "id": "ci_1",
        "type": "code_interpreter_call",
        "status": "in_progress",
    }
    assert hosted_events[1].item is None
    assert hosted_events[2].item == {
        "id": "ci_1",
        "type": "code_interpreter_call",
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_stream_response_uses_done_fallback_when_no_reasoning_delta_is_seen() -> None:
    provider = make_provider(reasoning_effort="high")
    final_response = FakeResponse(
        output_text="reasoning-ok",
        output=[FakeOutputTextItem(content=[{"type": "output_text", "text": "reasoning-ok"}])],
    )
    provider._client = fake_client(
        stream=FakeStream(
            events=[
                SimpleNamespace(
                    type="response.reasoning_summary_text.done",
                    text=REASONING_SUMMARY,
                    item_id="rs_1",
                    summary_index=0,
                ),
            ],
            final_response=final_response,
        )
    )

    events = [event async for event in provider.stream_response(input_items=[], tools=[])]

    assert [event.delta for event in events if event.type == "reasoning_delta"] == []
    assert events[-1].type == "completed"
    assert events[-1].response.reasoning_summary == REASONING_SUMMARY
