from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from simple_agent_base.chat import ChatSession
from simple_agent_base.config import AgentConfig
from simple_agent_base.errors import (
    MaxTurnsExceededError,
    MCPApprovalRequiredError,
    ToolExecutionError,
    ToolRegistrationError,
)
from simple_agent_base.mcp import (
    ApprovalHandler,
    MCPApprovalRequest,
    MCPBridgeManager,
    MCPCallRecord,
    MCPToolDefinition,
    MCPServer,
    build_mcp_approval_request,
    mcp_result_payload,
    normalize_mcp_tool_result,
)
from simple_agent_base.providers.base import Provider, ProviderResponse
from simple_agent_base.providers.openai import OpenAIResponsesProvider
from simple_agent_base.sync_utils import SyncRuntime, ensure_sync_allowed, run_sync_awaitable
from simple_agent_base.tools import ToolRegistry
from simple_agent_base.transcript import (
    build_transcript,
    clean_system_prompt,
    normalize_input,
    tool_output_item,
)
from simple_agent_base.types import (
    AgentEvent,
    AgentRunResult,
    ChatSnapshot,
    ConversationItem,
    MessageInput,
    JSONObject,
    ToolCallRequest,
    ToolExecutionResult,
    HostedToolCallUpdate,
    UsageMetadata,
)

T = TypeVar("T")


@dataclass(slots=True)
class _ExecutedCall:
    tool_result: ToolExecutionResult
    mcp_call: MCPCallRecord | None = None
    emit_mcp_events: bool = True


@dataclass(slots=True)
class _PreparedToolCall:
    events: list[AgentEvent]
    mcp_tool: MCPToolDefinition | None = None


@dataclass(slots=True)
class _StreamToolExecution:
    events: list[AgentEvent]
    denied_call: _ExecutedCall | None = None
    skip_mcp_approval: bool = False


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: list[Callable[..., object]] | ToolRegistry | None = None,
        provider: Provider | None = None,
        system_prompt: str | None = None,
        mcp_servers: Sequence[MCPServer] | None = None,
        hosted_tools: Sequence[JSONObject] | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.config = config
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.provider = provider or OpenAIResponsesProvider(config)
        self.system_prompt = clean_system_prompt(system_prompt)
        self.mcp_servers: list[MCPServer] = list(mcp_servers or [])
        self.hosted_tools: list[JSONObject] = self._validate_hosted_tools(hosted_tools)
        self.approval_handler = approval_handler
        self._mcp_manager = MCPBridgeManager(self.mcp_servers)
        self._sync_runtime: SyncRuntime | None = None

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        transcript = build_transcript(
            input_data,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )
        return await self._run_transcript(transcript, response_model=response_model)

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        transcript = build_transcript(
            input_data,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )
        async for event in self._stream_transcript(transcript, response_model=response_model):
            yield event

    def chat(
        self,
        messages: str | Sequence[MessageInput] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> ChatSession:
        initial_items: list[ConversationItem] = []
        if messages is not None:
            initial_items = normalize_input(messages)
        return ChatSession(
            self,
            items=initial_items,
            system_prompt=self._resolve_system_prompt(system_prompt),
        )

    def chat_from_snapshot(
        self,
        snapshot: ChatSnapshot | JSONObject,
    ) -> ChatSession:
        validated = snapshot if isinstance(snapshot, ChatSnapshot) else ChatSnapshot.model_validate(snapshot)
        return ChatSession(
            self,
            items=validated.items,
            system_prompt=clean_system_prompt(validated.system_prompt),
        )

    async def aclose(self) -> None:
        try:
            await self._mcp_manager.close()
        finally:
            await self.provider.close()

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def run_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        return self._run_sync_call(
            lambda: self.run(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="run_sync()",
            async_hint="await agent.run(...)",
        )

    def stream_sync(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[AgentEvent]:
        return self._stream_sync_call(
            lambda: self.stream(
                input_data,
                response_model=response_model,
                system_prompt=system_prompt,
            ),
            api_name="stream_sync()",
            async_hint="async for event in agent.stream(...)",
        )

    def close(self) -> None:
        ensure_sync_allowed("close()", "await agent.aclose()")
        if self._sync_runtime is not None:
            try:
                self._sync_runtime.run(lambda: self.aclose())
            finally:
                self._sync_runtime.close()
                self._sync_runtime = None
            return

        run_sync_awaitable(self.aclose())

    async def _run_transcript(
        self,
        transcript: list[ConversationItem],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        await self._ensure_mcp_ready()
        tool_results: list[ToolExecutionResult] = []
        mcp_calls: list[MCPCallRecord] = []
        usage_by_response: list[UsageMetadata] = []
        raw_responses: list[JSONObject] = []

        for _ in range(self.config.max_turns):
            response = await self.provider.create_response(
                input_items=transcript,
                tools=self._build_tool_params(),
                response_model=response_model,
            )
            raw_responses.append(response.raw_response or {})
            if response.usage is not None:
                usage_by_response.append(response.usage)
            transcript.extend(response.output_items)

            if not response.tool_calls:
                return self._build_run_result(
                    response=response,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    usage_by_response=usage_by_response,
                    raw_responses=raw_responses,
                )

            for executed in await self._execute_tool_batch(response.tool_calls):
                tool_results.append(executed.tool_result)
                if executed.mcp_call is not None:
                    mcp_calls.append(executed.mcp_call)
                transcript.append(tool_output_item(executed.tool_result))

        raise MaxTurnsExceededError(
            f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
        )

    async def _stream_transcript(
        self,
        transcript: list[ConversationItem],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        tool_results: list[ToolExecutionResult] = []
        mcp_calls: list[MCPCallRecord] = []
        usage_by_response: list[UsageMetadata] = []
        raw_responses: list[JSONObject] = []

        await self._ensure_mcp_ready()
        for _ in range(self.config.max_turns):
            final_response = None

            async for event in self.provider.stream_response(
                input_items=transcript,
                tools=self._build_tool_params(),
                response_model=response_model,
            ):
                if event.type in {"text_delta", "reasoning_delta"}:
                    yield AgentEvent(type=event.type, delta=event.delta)
                elif event.type == "tool_arguments_delta":
                    yield AgentEvent(
                        type="tool_arguments_delta",
                        delta=event.delta,
                        tool_item_id=event.item_id,
                        tool_name=event.name,
                    )
                elif event.type in {
                    "hosted_tool_call_started",
                    "hosted_tool_call_updated",
                    "hosted_tool_call_completed",
                }:
                    yield AgentEvent(
                        type=event.type,
                        hosted_tool_call=HostedToolCallUpdate(
                            item_id=event.item_id,
                            tool_type=event.tool_type,
                            status=event.status,
                            output_index=event.output_index,
                            sequence_number=event.sequence_number,
                            item=event.item,
                        ),
                    )
                elif event.type == "completed":
                    final_response = event.response

            if final_response is None:
                raise MaxTurnsExceededError("Provider stream completed without a final response.")

            raw_responses.append(final_response.raw_response or {})
            if final_response.usage is not None:
                usage_by_response.append(final_response.usage)
            transcript.extend(final_response.output_items)

            if not final_response.tool_calls:
                result = self._build_run_result(
                    response=final_response,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    usage_by_response=usage_by_response,
                    raw_responses=raw_responses,
                )
                yield AgentEvent(type="completed", result=result)
                return

            for call in final_response.tool_calls:
                yield AgentEvent(type="tool_call_started", tool_call=call)

            async for event in self._execute_tool_batch_stream(
                final_response.tool_calls,
                tool_results=tool_results,
                mcp_calls=mcp_calls,
                transcript=transcript,
            ):
                yield event

        raise MaxTurnsExceededError(
            f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
        )

    async def _ensure_mcp_ready(self) -> None:
        await self._mcp_manager.ensure_initialized()
        local_names = {definition.name for definition in self.registry.list_definitions()}
        duplicate_names = sorted(local_names & self._mcp_manager.tool_names())
        if duplicate_names:
            duplicate_list = ", ".join(duplicate_names)
            raise ToolRegistrationError(f"MCP tool names conflict with local tools: {duplicate_list}")

    async def _execute_tool(
        self,
        call: ToolCallRequest,
        *,
        skip_mcp_approval: bool = False,
    ) -> _ExecutedCall:
        if self._mcp_manager.has_tool(call.name):
            return await self._execute_mcp_tool(call, skip_approval=skip_mcp_approval)
        return _ExecutedCall(
            tool_result=await self._with_tool_timeout(
                self.registry.execute(call),
                timeout_message=f"Tool '{call.name}' timed out after {self.config.tool_timeout} seconds.",
            )
        )

    async def _execute_tool_batch(
        self,
        calls: Sequence[ToolCallRequest],
    ) -> list[_ExecutedCall]:
        if not self.config.parallel_tool_calls:
            results: list[_ExecutedCall] = []
            for call in calls:
                results.append(await self._execute_tool(call))
            return results

        return list(await asyncio.gather(*(self._execute_tool(call) for call in calls)))

    async def _execute_tool_batch_stream(
        self,
        calls: Sequence[ToolCallRequest],
        *,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        transcript: list[ConversationItem],
    ) -> AsyncIterator[AgentEvent]:
        if self.config.parallel_tool_calls:
            pending: list[asyncio.Task[_ExecutedCall]] = []
            for call in calls:
                prepared = self._prepare_tool_call_for_stream(call)
                for event in prepared.events:
                    yield event
                execution = await self._resolve_prepared_tool_call_for_stream(call, prepared)
                for event in execution.events:
                    yield event
                if execution.denied_call is not None:
                    for event in self._append_executed_call_events(
                        execution.denied_call,
                        tool_results=tool_results,
                        mcp_calls=mcp_calls,
                        transcript=transcript,
                    ):
                        yield event
                    continue
                pending.append(
                    asyncio.create_task(
                        self._execute_tool(call, skip_mcp_approval=execution.skip_mcp_approval)
                    )
                )

            if not pending:
                return

            executed_calls = await asyncio.gather(*pending)
            for executed in executed_calls:
                for event in self._append_executed_call_events(
                    executed,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    transcript=transcript,
                ):
                    yield event
            return

        for call in calls:
            prepared = self._prepare_tool_call_for_stream(call)
            for event in prepared.events:
                yield event
            execution = await self._resolve_prepared_tool_call_for_stream(call, prepared)
            for event in execution.events:
                yield event
            if execution.denied_call is not None:
                for event in self._append_executed_call_events(
                    execution.denied_call,
                    tool_results=tool_results,
                    mcp_calls=mcp_calls,
                    transcript=transcript,
                ):
                    yield event
                continue
            executed = await self._execute_tool(
                call,
                skip_mcp_approval=execution.skip_mcp_approval,
            )
            for event in self._append_executed_call_events(
                executed,
                tool_results=tool_results,
                mcp_calls=mcp_calls,
                transcript=transcript,
            ):
                yield event

    def _build_tool_params(self) -> list[JSONObject]:
        tools = list(self.registry.to_openai_tools())
        tools.extend(self._mcp_manager.to_openai_tools())
        tools.extend(self.hosted_tools)
        return tools

    @staticmethod
    def _validate_hosted_tools(
        hosted_tools: Sequence[JSONObject] | None,
    ) -> list[JSONObject]:
        if not hosted_tools:
            return []

        validated: list[JSONObject] = []
        for index, entry in enumerate(hosted_tools):
            if not isinstance(entry, dict):
                raise ToolRegistrationError(
                    f"hosted_tools[{index}] must be a dict, got {type(entry).__name__}."
                )
            tool_type = entry.get("type")
            if not isinstance(tool_type, str) or not tool_type:
                raise ToolRegistrationError(
                    f"hosted_tools[{index}] must include a non-empty string 'type' field."
                )
            validated.append(dict(entry))
        return validated

    @staticmethod
    def _build_run_result(
        *,
        response: ProviderResponse,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        usage_by_response: list[UsageMetadata],
        raw_responses: list[JSONObject],
    ) -> AgentRunResult:
        return AgentRunResult(
            output_text=response.output_text,
            reasoning_summary=response.reasoning_summary,
            output_data=response.output_data,
            response_id=response.response_id,
            tool_results=tool_results,
            mcp_calls=mcp_calls,
            usage=Agent._aggregate_usage(usage_by_response),
            usage_by_response=list(usage_by_response),
            raw_responses=raw_responses,
        )

    @staticmethod
    def _aggregate_usage(usages: Sequence[UsageMetadata]) -> UsageMetadata | None:
        input_total = Agent._sum_optional_usage_field(usages, "input_tokens")
        output_total = Agent._sum_optional_usage_field(usages, "output_tokens")
        total = Agent._sum_optional_usage_field(usages, "total_tokens")

        if input_total is None and output_total is None and total is None:
            return None

        return UsageMetadata(
            input_tokens=input_total,
            output_tokens=output_total,
            total_tokens=total,
        )

    @staticmethod
    def _sum_optional_usage_field(usages: Sequence[UsageMetadata], field_name: str) -> int | None:
        values = [getattr(usage, field_name) for usage in usages]
        present = [value for value in values if isinstance(value, int)]
        return sum(present) if present else None

    async def _execute_mcp_tool(
        self,
        call: ToolCallRequest,
        *,
        skip_approval: bool = False,
    ) -> _ExecutedCall:
        tool = self._mcp_manager.get_tool(call.name)
        approved = True

        if tool.require_approval and not skip_approval:
            approved = await self._approve_mcp_call(self._build_mcp_approval_request(tool, call))

        if not approved:
            return self._build_denied_mcp_call(tool, call)

        async def call_mcp() -> tuple[MCPToolDefinition, object, str]:
            called_tool, raw_result = await self._mcp_manager.call_tool(
                namespaced_name=call.name,
                arguments=call.arguments,
            )
            output = normalize_mcp_tool_result(raw_result)
            return called_tool, raw_result, output

        try:
            tool, raw_result, output = await self._with_tool_timeout(
                call_mcp(),
                timeout_message=(
                    f"MCP tool '{tool.tool_name}' timed out after "
                    f"{self.config.tool_timeout} seconds."
                ),
            )
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"MCP tool '{tool.tool_name}' failed: {exc}") from exc

        if raw_result.isError:
            raise ToolExecutionError(f"MCP tool '{tool.tool_name}' failed: {output}")

        return _ExecutedCall(
            tool_result=ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                output=output,
                raw_output=mcp_result_payload(raw_result),
            ),
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
                output=output,
            ),
        )

    async def _with_tool_timeout(
        self,
        awaitable: Awaitable[T],
        *,
        timeout_message: str,
    ) -> T:
        if self.config.tool_timeout is None:
            return await awaitable

        try:
            return await asyncio.wait_for(awaitable, timeout=self.config.tool_timeout)
        except TimeoutError as exc:
            raise ToolExecutionError(timeout_message) from exc

    def _prepare_tool_call_for_stream(
        self,
        call: ToolCallRequest,
    ) -> _PreparedToolCall:
        if not self._mcp_manager.has_tool(call.name):
            return _PreparedToolCall(events=[])

        tool = self._mcp_manager.get_tool(call.name)
        events: list[AgentEvent] = []
        if tool.require_approval:
            approval_event = self._build_mcp_approval_event(call)
            if approval_event is not None:
                events.append(approval_event)
            return _PreparedToolCall(events=events, mcp_tool=tool)

        events.append(self._build_mcp_started_event(call))
        return _PreparedToolCall(events=events)

    async def _resolve_prepared_tool_call_for_stream(
        self,
        call: ToolCallRequest,
        prepared: _PreparedToolCall,
    ) -> _StreamToolExecution:
        if prepared.mcp_tool is None:
            return _StreamToolExecution(events=[])

        approved = await self._approve_mcp_call(
            self._build_mcp_approval_request(prepared.mcp_tool, call)
        )
        if not approved:
            return _StreamToolExecution(
                events=[],
                denied_call=self._build_denied_mcp_call(prepared.mcp_tool, call),
            )

        return _StreamToolExecution(
            events=[self._build_mcp_started_event(call)],
            skip_mcp_approval=True,
        )

    def _build_denied_mcp_call(self, tool: MCPToolDefinition, call: ToolCallRequest) -> _ExecutedCall:
        message = "MCP tool call denied by approval handler."
        return _ExecutedCall(
            tool_result=ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                output=message,
            ),
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
                error=message,
            ),
            emit_mcp_events=False,
        )

    def _append_executed_call_events(
        self,
        executed: _ExecutedCall,
        *,
        tool_results: list[ToolExecutionResult],
        mcp_calls: list[MCPCallRecord],
        transcript: list[ConversationItem],
    ) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if executed.mcp_call is not None:
            mcp_calls.append(executed.mcp_call)
            if executed.emit_mcp_events:
                events.append(AgentEvent(type="mcp_call_completed", mcp_call=executed.mcp_call))

        tool_results.append(executed.tool_result)
        transcript.append(tool_output_item(executed.tool_result))
        events.append(AgentEvent(type="tool_call_completed", tool_result=executed.tool_result))
        return events

    def _build_mcp_approval_request(
        self,
        tool: MCPToolDefinition,
        call: ToolCallRequest,
    ) -> MCPApprovalRequest:
        return build_mcp_approval_request(
            request_id=f"mcp-approval-{call.call_id}",
            server_name=tool.server_name,
            tool_name=tool.tool_name,
            arguments=call.arguments,
        )

    def _build_mcp_approval_event(self, call: ToolCallRequest) -> AgentEvent | None:
        tool = self._mcp_manager.get_tool(call.name)
        if not tool.require_approval:
            return None
        return AgentEvent(
            type="mcp_approval_requested",
            mcp_approval=self._build_mcp_approval_request(tool, call),
        )

    def _build_mcp_started_event(self, call: ToolCallRequest) -> AgentEvent:
        tool = self._mcp_manager.get_tool(call.name)
        return AgentEvent(
            type="mcp_call_started",
            mcp_call=MCPCallRecord(
                id=call.call_id,
                server_name=tool.server_name,
                name=tool.tool_name,
                arguments=call.arguments,
            ),
        )

    async def _approve_mcp_call(self, approval: MCPApprovalRequest) -> bool:
        if self.approval_handler is None:
            raise MCPApprovalRequiredError(
                "An MCP tool requires approval but no approval_handler was provided. "
                "Set require_approval=False on the MCPServer or pass approval_handler=... to Agent(...)."
            )

        result = self.approval_handler(approval)
        if inspect.isawaitable(result):
            return bool(await result)
        return bool(result)

    def _run_sync_call(
        self,
        awaitable_factory: Callable[[], Awaitable[T]],
        *,
        api_name: str,
        async_hint: str,
    ) -> T:
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().run(awaitable_factory)

    def _stream_sync_call(
        self,
        async_iterable_factory: Callable[[], AsyncIterator[T]],
        *,
        api_name: str,
        async_hint: str,
    ) -> Iterator[T]:
        ensure_sync_allowed(api_name, async_hint)
        return self._get_sync_runtime().iterate(async_iterable_factory)

    def _get_sync_runtime(self) -> SyncRuntime:
        if self._sync_runtime is None:
            self._sync_runtime = SyncRuntime()
        return self._sync_runtime

    def _resolve_system_prompt(self, system_prompt: str | None) -> str | None:
        return self._resolve_system_prompt_with_default(system_prompt, self.system_prompt)

    def _resolve_system_prompt_with_default(
        self,
        system_prompt: str | None,
        default: str | None,
    ) -> str | None:
        cleaned = clean_system_prompt(system_prompt)
        if cleaned is not None:
            return cleaned
        return default
