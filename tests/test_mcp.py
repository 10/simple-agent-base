from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp import types as mcp_types
from pydantic import ValidationError

from conftest import FakeProvider
from simple_agent_base import Agent, AgentConfig, MCPServer, tool
from simple_agent_base.errors import MCPApprovalRequiredError, ToolExecutionError, ToolRegistrationError
from simple_agent_base.mcp import normalize_mcp_tool_result
from simple_agent_base.providers.base import ProviderCompletedEvent, ProviderResponse
import simple_agent_base.mcp as mcp_module

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "mcp_demo_server.py"


@pytest.fixture
def stdio_server() -> MCPServer:
    return MCPServer.stdio(
        name="demo",
        command=sys.executable,
        args=[str(FIXTURE_SERVER), "stdio"],
    )


@pytest.fixture
def http_server() -> str:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE_SERVER), "http", str(port)],
        cwd=str(FIXTURE_SERVER.parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        _wait_for_http_server(url)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_mcp_server_stdio_constructor() -> None:
    server = MCPServer.stdio(
        name="demo",
        command="python",
        args=["server.py"],
        env={"TOKEN": "abc"},
        cwd="C:/tmp",
        allowed_tools=["echo"],
        require_approval=True,
    )

    assert server.transport == "stdio"
    assert server.command == "python"
    assert server.args == ["server.py"]
    assert server.env == {"TOKEN": "abc"}
    assert server.cwd == "C:/tmp"
    assert server.allowed_tools == ["echo"]
    assert server.require_approval is True


def test_mcp_server_http_constructor() -> None:
    server = MCPServer.http(
        name="demo-http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer 123"},
        allowed_tools=["echo"],
    )

    assert server.transport == "streamable_http"
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"Authorization": "Bearer 123"}
    assert server.allowed_tools == ["echo"]
    assert server.require_approval is False


def test_mcp_server_rejects_invalid_transport_fields() -> None:
    with pytest.raises(ValidationError):
        MCPServer(name="bad", transport="stdio")

    with pytest.raises(ValidationError):
        MCPServer(
            name="bad-http",
            transport="streamable_http",
            url="https://example.com/mcp",
            command="python",
        )


def test_normalize_mcp_tool_result_prefers_text_blocks() -> None:
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="one"),
            mcp_types.TextContent(type="text", text="two"),
        ],
        isError=False,
    )

    assert normalize_mcp_tool_result(result) == "one\ntwo"


def test_normalize_mcp_tool_result_uses_structured_content() -> None:
    result = mcp_types.CallToolResult(
        content=[],
        structuredContent={"name": "Ada"},
        isError=False,
    )

    assert normalize_mcp_tool_result(result) == '{"name": "Ada"}'


def test_normalize_mcp_tool_result_falls_back_to_full_payload() -> None:
    result = mcp_types.CallToolResult(content=[], isError=False)

    output = normalize_mcp_tool_result(result)

    assert '"isError": false' in output


@pytest.mark.asyncio
async def test_agent_discovers_and_executes_stdio_mcp_tool(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
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
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server],
    )

    try:
        result = await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()

    assert [tool["name"] for tool in provider.calls[0]["tools"]] == [
        "demo__echo",
        "demo__add",
        "demo__structured_value",
        "demo__fail",
    ]
    assert result.output_text == "done"
    assert result.tool_results[0].output == "echo:hello"
    assert result.mcp_calls[0].server_name == "demo"
    assert result.mcp_calls[0].name == "echo"
    assert result.mcp_calls[0].output == "echo:hello"


@pytest.mark.asyncio
async def test_agent_filters_allowed_mcp_tools(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                output_text="done",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[
            stdio_server.model_copy(update={"allowed_tools": ["echo"]})
        ],
    )

    try:
        await agent.run("List tools.")
    finally:
        await agent.aclose()

    assert [tool["name"] for tool in provider.calls[0]["tools"]] == ["demo__echo"]


@pytest.mark.asyncio
async def test_agent_with_empty_mcp_allowlist_exposes_no_tools(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                output_text="done",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"allowed_tools": []})],
    )

    try:
        await agent.run("List tools.")
    finally:
        await agent.aclose()

    assert provider.calls[0]["tools"] == []


def test_agent_rejects_duplicate_mcp_server_names(stdio_server: MCPServer) -> None:
    duplicate = stdio_server.model_copy(update={"args": [str(FIXTURE_SERVER), "stdio", "--unused"]})

    with pytest.raises(ToolRegistrationError, match="MCP server 'demo' is already registered"):
        Agent(
            config=AgentConfig(model="gpt-5"),
            provider=FakeProvider(),
            mcp_servers=[stdio_server, duplicate],
        )


@pytest.mark.asyncio
async def test_mcp_manager_rejects_duplicate_namespaced_tool_names() -> None:
    class FakeBridge:
        def __init__(self, tool: mcp_module.MCPToolDefinition) -> None:
            self._tool = tool

        async def list_tools(self) -> list[mcp_module.MCPToolDefinition]:
            return [self._tool]

    manager = mcp_module.MCPBridgeManager()
    manager._bridges = {
        "alpha": FakeBridge(
            mcp_module.MCPToolDefinition(
                server_name="alpha",
                tool_name="beta__gamma",
                namespaced_name="alpha__beta__gamma",
                description="First tool",
                parameters={"type": "object"},
                require_approval=False,
            )
        ),
        "alpha__beta": FakeBridge(
            mcp_module.MCPToolDefinition(
                server_name="alpha__beta",
                tool_name="gamma",
                namespaced_name="alpha__beta__gamma",
                description="Second tool",
                parameters={"type": "object"},
                require_approval=False,
            )
        ),
    }

    with pytest.raises(ToolRegistrationError, match="MCP tool 'alpha__beta__gamma' is already registered"):
        await manager.ensure_initialized()


@pytest.mark.asyncio
async def test_failed_mcp_initialization_closes_opened_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"transport_closed": False, "session_closed": False}
    server = MCPServer.stdio(
        name="demo",
        command=sys.executable,
        args=[str(FIXTURE_SERVER), "stdio"],
    )

    @asynccontextmanager
    async def fake_stdio_client(_params: Any) -> AsyncIterator[tuple[str, str]]:
        try:
            yield ("read-stream", "write-stream")
        finally:
            state["transport_closed"] = True

    class FakeSession:
        def __init__(self, read_stream: str, write_stream: str) -> None:
            self.read_stream = read_stream
            self.write_stream = write_stream

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            state["session_closed"] = True

        async def initialize(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(mcp_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp_module, "ClientSession", FakeSession)

    bridge = mcp_module.MCPClientBridge(server)

    with pytest.raises(RuntimeError, match="boom"):
        await bridge.list_tools()

    assert state == {"transport_closed": True, "session_closed": True}
    assert bridge._stack is None
    assert bridge._session is None


@pytest.mark.asyncio
async def test_agent_approval_handler_can_deny_mcp_call(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            ),
            ProviderResponse(
                output_text="handled",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
        approval_handler=lambda request: False,
    )

    try:
        result = await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()

    assert result.output_text == "handled"
    assert result.tool_results[0].output == "MCP tool call denied by approval handler."
    assert result.mcp_calls[0].error == "MCP tool call denied by approval handler."


@pytest.mark.asyncio
async def test_agent_requires_approval_handler_for_gated_mcp_call(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
    )

    try:
        with pytest.raises(MCPApprovalRequiredError):
            await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_surfaces_mcp_error_results_as_tool_failures(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__fail",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server],
    )

    try:
        with pytest.raises(ToolExecutionError, match="boom"):
            await agent.run("Use the failing MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_stream_emits_local_mcp_events(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "demo__echo",
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
                        output_text="done",
                    )
                )
            ],
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[stdio_server.model_copy(update={"require_approval": True})],
        approval_handler=lambda request: True,
    )

    try:
        events = [event async for event in agent.stream("Use the MCP tool.")]
    finally:
        await agent.aclose()

    assert [event.type for event in events] == [
        "tool_call_started",
        "mcp_approval_requested",
        "mcp_call_started",
        "mcp_call_completed",
        "tool_call_completed",
        "completed",
    ]
    approval = events[1].mcp_approval
    assert approval is not None
    assert approval.server_name == "demo"
    assert approval.name == "echo"
    mcp_completed = events[3].mcp_call
    assert mcp_completed is not None
    assert mcp_completed.server_name == "demo"
    assert mcp_completed.output == "echo:hello"


@pytest.mark.asyncio
async def test_stream_yields_mcp_approval_request_before_waiting_for_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "demo__echo",
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
                        output_text="done",
                    )
                )
            ],
        ]
    )

    async def approve(_request: Any) -> bool:
        await asyncio.sleep(0.1)
        return True

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        approval_handler=approve,
    )
    fake_tool = SimpleNamespace(server_name="demo", tool_name="echo", require_approval=True)

    async def fake_call_tool(*, namespaced_name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="echo:hello")],
            isError=False,
        )

    monkeypatch.setattr(agent, "_ensure_mcp_ready", AsyncMock())
    monkeypatch.setattr(agent._mcp_manager, "has_tool", lambda name: name == "demo__echo")
    monkeypatch.setattr(agent._mcp_manager, "get_tool", lambda name: fake_tool)
    monkeypatch.setattr(agent._mcp_manager, "call_tool", fake_call_tool)

    stream = agent.stream("Use the MCP tool.")
    try:
        first = await asyncio.wait_for(anext(stream), timeout=0.05)
        second = await asyncio.wait_for(anext(stream), timeout=0.05)
        remaining = [event async for event in stream]
    finally:
        await agent.aclose()

    assert first.type == "tool_call_started"
    assert second.type == "mcp_approval_requested"
    assert [event.type for event in remaining] == [
        "mcp_call_started",
        "mcp_call_completed",
        "tool_call_completed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_stream_does_not_emit_mcp_started_when_approval_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        stream_sequences=[
            [
                ProviderCompletedEvent(
                    response=ProviderResponse(
                        tool_calls=[
                            {
                                "call_id": "call_1",
                                "name": "demo__echo",
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
                        output_text="handled",
                    )
                )
            ],
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        approval_handler=lambda _request: False,
    )
    fake_tool = SimpleNamespace(server_name="demo", tool_name="echo", require_approval=True)

    monkeypatch.setattr(agent, "_ensure_mcp_ready", AsyncMock())
    monkeypatch.setattr(agent._mcp_manager, "has_tool", lambda name: name == "demo__echo")
    monkeypatch.setattr(agent._mcp_manager, "get_tool", lambda name: fake_tool)

    try:
        events = [event async for event in agent.stream("Use the MCP tool.")]
    finally:
        await agent.aclose()

    assert [event.type for event in events] == [
        "tool_call_started",
        "mcp_approval_requested",
        "tool_call_completed",
        "completed",
    ]


@tool(name="demo__echo")
async def conflicting_local_tool(message: str) -> str:
    """Conflicts with the MCP echo tool."""
    return message


@pytest.mark.asyncio
async def test_agent_rejects_conflicting_local_and_mcp_tool_names(stdio_server: MCPServer) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                output_text="done",
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        tools=[conflicting_local_tool],
        mcp_servers=[stdio_server],
    )

    try:
        with pytest.raises(ToolRegistrationError, match="demo__echo"):
            await agent.run("Hello.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_executes_streamable_http_mcp_tool(http_server: str) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demohttp__add",
                        "arguments": {"a": 2, "b": 3},
                        "raw_arguments": '{"a":2,"b":3}',
                    }
                ],
            ),
            ProviderResponse(
                output_text="done",
            ),
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
        mcp_servers=[MCPServer.http(name="demohttp", url=http_server)],
    )

    try:
        result = await agent.run("Use the HTTP MCP tool.")
    finally:
        await agent.aclose()

    assert result.tool_results[0].output == "5"
    assert result.mcp_calls[0].server_name == "demohttp"
    assert result.mcp_calls[0].name == "add"


@pytest.mark.asyncio
async def test_agent_wraps_mcp_transport_errors_as_tool_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=provider,
    )
    fake_tool = SimpleNamespace(server_name="demo", tool_name="echo", require_approval=False)

    async def broken_call_tool(*, namespaced_name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("transport dropped")

    monkeypatch.setattr(agent, "_ensure_mcp_ready", AsyncMock())
    monkeypatch.setattr(agent._mcp_manager, "has_tool", lambda name: name == "demo__echo")
    monkeypatch.setattr(agent._mcp_manager, "get_tool", lambda name: fake_tool)
    monkeypatch.setattr(agent._mcp_manager, "call_tool", broken_call_tool)

    try:
        with pytest.raises(ToolExecutionError, match="transport dropped"):
            await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_times_out_slow_mcp_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=0.01),
        provider=provider,
    )
    fake_tool = SimpleNamespace(server_name="demo", tool_name="echo", require_approval=False)

    async def slow_call_tool(*, namespaced_name: str, arguments: dict[str, Any]) -> Any:
        await asyncio.sleep(0.2)
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="echo:hello")],
            isError=False,
        )

    monkeypatch.setattr(agent, "_ensure_mcp_ready", AsyncMock())
    monkeypatch.setattr(agent._mcp_manager, "has_tool", lambda name: name == "demo__echo")
    monkeypatch.setattr(agent._mcp_manager, "get_tool", lambda name: fake_tool)
    monkeypatch.setattr(agent._mcp_manager, "call_tool", slow_call_tool)

    try:
        with pytest.raises(
            ToolExecutionError,
            match="MCP tool 'echo' timed out after 0.01 seconds.",
        ):
            await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_mcp_approval_time_is_excluded_from_tool_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        responses=[
            ProviderResponse(
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "demo__echo",
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
    fake_tool = SimpleNamespace(server_name="demo", tool_name="echo", require_approval=True)

    async def approve(_request: Any) -> bool:
        await asyncio.sleep(0.05)
        return True

    agent = Agent(
        config=AgentConfig(model="gpt-5", tool_timeout=0.01),
        provider=provider,
        approval_handler=approve,
    )

    async def quick_call_tool(*, namespaced_name: str, arguments: dict[str, Any]) -> Any:
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="echo:hello")],
            isError=False,
        )

    monkeypatch.setattr(agent, "_ensure_mcp_ready", AsyncMock())
    monkeypatch.setattr(agent._mcp_manager, "has_tool", lambda name: name == "demo__echo")
    monkeypatch.setattr(agent._mcp_manager, "get_tool", lambda name: fake_tool)
    monkeypatch.setattr(agent._mcp_manager, "call_tool", quick_call_tool)

    try:
        result = await agent.run("Use the MCP tool.")
    finally:
        await agent.aclose()

    assert result.output_text == "done"
    assert result.tool_results[0].output == "echo:hello"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_server(url: str) -> None:
    host = "127.0.0.1"
    port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except Exception as exc:  # pragma: no cover - best-effort startup polling
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"HTTP MCP fixture did not start: {last_error}")
