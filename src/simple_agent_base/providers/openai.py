from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import cast

from pydantic import BaseModel
from openai import AsyncOpenAI, DefaultAioHttpClient

from simple_agent_base.config import AgentConfig
from simple_agent_base.errors import ProviderError
from simple_agent_base.types import ConversationItem, JSONObject, ToolCallRequest, UsageMetadata

from .base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderHostedToolCallEvent,
    ProviderReasoningDeltaEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
    ProviderToolArgumentsDeltaEvent,
)


class OpenAIResponsesProvider:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            http_client=DefaultAioHttpClient(),
        )

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        try:
            kwargs = self._request_kwargs(input_items, tools, response_model=response_model)
            if response_model is None:
                response = await self._client.responses.create(**kwargs)
            else:
                response = await self._client.responses.parse(**kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI response request failed: {exc}") from exc

        return self._convert_response(response)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        reasoning_parts: list[str] = []
        seen_reasoning_keys: set[tuple[str | None, int | None]] = set()
        function_call_meta: dict[str, tuple[str | None, str | None]] = {}
        hosted_tool_statuses: dict[str, str] = {}

        try:
            async with self._client.responses.stream(
                **self._request_kwargs(input_items, tools, response_model=response_model)
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield ProviderTextDeltaEvent(delta=event.delta)
                    elif event.type == "response.reasoning_summary_text.delta":
                        seen_reasoning_keys.add(
                            (getattr(event, "item_id", None), getattr(event, "summary_index", None))
                        )
                        delta = getattr(event, "delta", "")
                        if delta:
                            reasoning_parts.append(delta)
                        yield ProviderReasoningDeltaEvent(delta=delta)
                    elif event.type == "response.reasoning_summary_text.done":
                        key = (getattr(event, "item_id", None), getattr(event, "summary_index", None))
                        text = getattr(event, "text", "")
                        if key not in seen_reasoning_keys and text:
                            reasoning_parts.append(text)
                    elif event.type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", None) == "function_call":
                            function_call_meta[item.id] = (
                                getattr(item, "call_id", None),
                                getattr(item, "name", None),
                            )
                        elif (hosted_event := self._hosted_tool_event_from_output_item(
                            item,
                            event_type="hosted_tool_call_started",
                            default_status="in_progress",
                            skip_duplicate=True,
                            output_index=getattr(event, "output_index", None),
                            sequence_number=getattr(event, "sequence_number", None),
                            hosted_tool_statuses=hosted_tool_statuses,
                        )) is not None:
                            yield hosted_event
                    elif event.type == "response.function_call_arguments.delta":
                        call_id, name = function_call_meta.get(event.item_id, (None, None))
                        yield ProviderToolArgumentsDeltaEvent(
                            item_id=event.item_id,
                            call_id=call_id,
                            name=name,
                            delta=event.delta,
                        )
                    elif event.type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if (hosted_event := self._hosted_tool_event_from_output_item(
                            item,
                            event_type="hosted_tool_call_completed",
                            default_status="completed",
                            skip_duplicate=False,
                            output_index=getattr(event, "output_index", None),
                            sequence_number=getattr(event, "sequence_number", None),
                            hosted_tool_statuses=hosted_tool_statuses,
                        )) is not None:
                            yield hosted_event
                    elif (hosted_event := self._hosted_tool_event_from_stream_event(
                        event,
                        hosted_tool_statuses=hosted_tool_statuses,
                    )) is not None:
                        yield hosted_event

                final_response = await stream.get_final_response()
        except Exception as exc:
            raise ProviderError(f"OpenAI streaming response request failed: {exc}") from exc

        response = self._convert_response(final_response)
        if response.reasoning_summary is None:
            summary = "".join(reasoning_parts).strip()
            response.reasoning_summary = summary or None
        yield ProviderCompletedEvent(response=response)

    async def close(self) -> None:
        await self._client.close()

    def _request_kwargs(
        self,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> JSONObject:
        kwargs: JSONObject = {
            "model": self._config.model,
            "input": list(input_items),
            "parallel_tool_calls": self._config.parallel_tool_calls,
        }

        if tools:
            kwargs["tools"] = list(tools)

        if self._config.reasoning_effort is not None:
            kwargs["reasoning"] = {
                "effort": self._config.reasoning_effort,
                "summary": "auto",
            }

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        if response_model is not None:
            kwargs["text_format"] = response_model

        return kwargs

    def _convert_response(self, response: BaseModel) -> ProviderResponse:
        output_items = [self._to_dict(item) for item in getattr(response, "output", [])]
        tool_calls: list[ToolCallRequest] = []

        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "function_call":
                continue

            raw_arguments = getattr(item, "arguments", "{}")

            try:
                parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Model returned invalid JSON arguments for tool '{item.name}': {raw_arguments}"
                ) from exc

            if not isinstance(parsed_arguments, dict):
                raise ProviderError(
                    f"Model returned non-object JSON arguments for tool '{item.name}': {raw_arguments}"
                )

            tool_calls.append(
                ToolCallRequest(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=cast(JSONObject, parsed_arguments),
                    raw_arguments=raw_arguments,
                )
            )

        return ProviderResponse(
            response_id=getattr(response, "id", None),
            output_text=getattr(response, "output_text", ""),
            reasoning_summary=self._extract_reasoning_summary(response),
            output_data=cast(BaseModel | None, getattr(response, "output_parsed", None)),
            tool_calls=tool_calls,
            output_items=output_items,
            usage=self._extract_usage(response),
            raw_response=self._to_dict(response),
        )

    def _extract_usage(self, response: BaseModel) -> UsageMetadata | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        raw_usage = self._to_dict(usage)
        input_tokens = raw_usage.get("input_tokens")
        output_tokens = raw_usage.get("output_tokens")
        total_tokens = raw_usage.get("total_tokens")
        input_tokens_details = raw_usage.get("input_tokens_details")
        output_tokens_details = raw_usage.get("output_tokens_details")
        return UsageMetadata(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            input_tokens_details=cast(JSONObject, input_tokens_details)
            if isinstance(input_tokens_details, dict)
            else None,
            output_tokens_details=cast(JSONObject, output_tokens_details)
            if isinstance(output_tokens_details, dict)
            else None,
            raw=raw_usage,
        )

    def _extract_reasoning_summary(self, response: BaseModel) -> str | None:
        reasoning_summaries = [
            summary
            for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "reasoning"
            if (summary := self._join_non_empty_texts(getattr(item, "summary", []) or [])) is not None
        ]
        return self._join_non_empty_texts(reasoning_summaries, separator="\n\n")

    @staticmethod
    def _join_non_empty_texts(parts: Sequence[object], *, separator: str = "") -> str | None:
        cleaned_parts = [
            stripped
            for part in parts
            if isinstance((text := getattr(part, "text", part)), str)
            if (stripped := text.strip())
        ]
        if not cleaned_parts:
            return None
        return separator.join(cleaned_parts)

    @staticmethod
    def _to_dict(value: object) -> JSONObject:
        if isinstance(value, BaseModel):
            return cast(JSONObject, value.model_dump(mode="json", warnings="none"))
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return cast(JSONObject, to_dict())
        if isinstance(value, dict):
            return cast(JSONObject, value)
        try:
            return cast(JSONObject, dict(vars(value)))
        except TypeError:
            pass
        raise ProviderError(f"Unsupported response payload type: {type(value)!r}")

    @staticmethod
    def _is_hosted_tool_output_type(item_type: object) -> bool:
        return (
            isinstance(item_type, str)
            and "call" in item_type
            and item_type not in {"function_call", "function_call_output", "mcp_call"}
            and not item_type.startswith("mcp_")
        )

    @classmethod
    def _hosted_tool_event_from_output_item(
        cls,
        item: object,
        *,
        event_type: str,
        default_status: str,
        skip_duplicate: bool,
        output_index: object,
        sequence_number: object,
        hosted_tool_statuses: dict[str, str],
    ) -> ProviderHostedToolCallEvent | None:
        item_type = getattr(item, "type", None)
        item_id = getattr(item, "id", None)
        if not cls._is_hosted_tool_output_type(item_type) or not isinstance(item_id, str):
            return None

        status = getattr(item, "status", default_status)
        if not isinstance(status, str):
            status = default_status
        if skip_duplicate and hosted_tool_statuses.get(item_id) == status:
            return None

        hosted_tool_statuses[item_id] = status
        return ProviderHostedToolCallEvent(
            type=event_type,
            item_id=item_id,
            tool_type=item_type,
            status=status,
            output_index=output_index if isinstance(output_index, int) else None,
            sequence_number=sequence_number if isinstance(sequence_number, int) else None,
            item=cls._to_dict(item),
        )

    @classmethod
    def _hosted_tool_event_from_stream_event(
        cls,
        event: object,
        *,
        hosted_tool_statuses: dict[str, str],
    ) -> ProviderHostedToolCallEvent | None:
        event_type = getattr(event, "type", None)
        if not isinstance(event_type, str) or not event_type.startswith("response."):
            return None

        tool_type, status = cls._parse_hosted_tool_stream_event(event_type)
        item_id = getattr(event, "item_id", None)
        if tool_type is None or status is None or not isinstance(item_id, str):
            return None

        if status == "completed":
            hosted_tool_statuses[item_id] = status
            return None

        provider_event_type = "hosted_tool_call_started" if status == "in_progress" else "hosted_tool_call_updated"
        if hosted_tool_statuses.get(item_id) == status:
            return None

        hosted_tool_statuses[item_id] = status
        return ProviderHostedToolCallEvent(
            type=provider_event_type,
            item_id=item_id,
            tool_type=tool_type,
            status=status,
            output_index=getattr(event, "output_index", None)
            if isinstance(getattr(event, "output_index", None), int)
            else None,
            sequence_number=getattr(event, "sequence_number", None)
            if isinstance(getattr(event, "sequence_number", None), int)
            else None,
            item=None,
        )

    @staticmethod
    def _parse_hosted_tool_stream_event(event_type: str) -> tuple[str | None, str | None]:
        tool_name, _, suffix = event_type.removeprefix("response.").rpartition(".")
        tool_types = {
            "web_search_call",
            "file_search_call",
            "image_generation_call",
            "code_interpreter_call",
        }
        allowed_statuses = {"in_progress", "searching", "generating", "interpreting", "completed"}
        if tool_name in tool_types and suffix in allowed_statuses:
            return (tool_name, suffix)
        return (None, None)
