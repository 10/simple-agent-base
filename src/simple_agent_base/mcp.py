from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, model_validator

from simple_agent_base.errors import ToolRegistrationError
from simple_agent_base.json_types import JSONObject


class MCPServer(BaseModel):
    """Configuration for a client-side MCP server connection."""

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "streamable_http"]
    allowed_tools: list[str] | None = None
    require_approval: bool = False
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServer:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("MCPServer transport='stdio' requires command.")
            if self.url is not None or self.headers is not None:
                raise ValueError("MCPServer transport='stdio' does not accept url or headers.")
        elif self.transport == "streamable_http":
            if not self.url:
                raise ValueError("MCPServer transport='streamable_http' requires url.")
            if self.command is not None or self.args or self.env is not None or self.cwd is not None:
                raise ValueError(
                    "MCPServer transport='streamable_http' does not accept command, args, env, or cwd."
                )
        return self

    @classmethod
    def stdio(
        cls,
        *,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
    ) -> MCPServer:
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=list(args or []),
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            allowed_tools=list(allowed_tools) if allowed_tools is not None else None,
            require_approval=require_approval,
        )

    @classmethod
    def http(
        cls,
        *,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
    ) -> MCPServer:
        return cls(
            name=name,
            transport="streamable_http",
            url=url,
            headers=dict(headers) if headers is not None else None,
            allowed_tools=list(allowed_tools) if allowed_tools is not None else None,
            require_approval=require_approval,
        )


class MCPApprovalRequest(BaseModel):
    """A local approval request before invoking a bridged MCP tool."""

    model_config = ConfigDict(extra="forbid")

    id: str
    server_name: str
    name: str
    arguments: JSONObject = Field(default_factory=dict)


class MCPCallRecord(BaseModel):
    """A record of a client-side MCP tool invocation."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    server_name: str
    name: str
    arguments: JSONObject = Field(default_factory=dict)
    output: str | None = None
    error: str | None = None


ApprovalHandler = Callable[[MCPApprovalRequest], bool | Awaitable[bool]]


@dataclass(slots=True)
class MCPToolDefinition:
    server_name: str
    tool_name: str
    namespaced_name: str
    description: str
    parameters: JSONObject
    require_approval: bool

    def to_openai_tool(self) -> JSONObject:
        return {
            "type": "function",
            "name": self.namespaced_name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


def build_mcp_approval_request(
    *,
    request_id: str | None = None,
    server_name: str,
    tool_name: str,
    arguments: JSONObject,
) -> MCPApprovalRequest:
    return MCPApprovalRequest(
        id=request_id or f"mcp-approval-{uuid.uuid4().hex}",
        server_name=server_name,
        name=tool_name,
        arguments=dict(arguments),
    )


def normalize_mcp_tool_result(result: mcp_types.CallToolResult) -> str:
    text_blocks: list[str] = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                text_blocks.append(text)

    if text_blocks:
        return "\n".join(text_blocks)

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=str)

    return json.dumps(_mcp_result_payload(result), ensure_ascii=False, default=str)


def mcp_result_payload(result: mcp_types.CallToolResult) -> object:
    return _mcp_result_payload(result)


class MCPClientBridge:
    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools: dict[str, MCPToolDefinition] | None = None

    async def list_tools(self) -> list[MCPToolDefinition]:
        await self._ensure_initialized()
        if self._tools is None:
            tools: dict[str, MCPToolDefinition] = {}
            cursor: str | None = None
            allowed = None if self.server.allowed_tools is None else set(self.server.allowed_tools)

            while True:
                response = await self._session.list_tools(cursor=cursor)
                for tool in response.tools:
                    if allowed is not None and tool.name not in allowed:
                        continue
                    definition = self._build_tool_definition(tool)
                    if definition.namespaced_name in tools:
                        raise ToolRegistrationError(
                            f"MCP tool '{definition.namespaced_name}' is already registered."
                        )
                    tools[definition.namespaced_name] = definition
                cursor = response.nextCursor
                if cursor is None:
                    break

            self._tools = tools

        return list(self._tools.values())

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: JSONObject,
    ) -> mcp_types.CallToolResult:
        await self._ensure_initialized()
        return await self._session.call_tool(tool_name, arguments=arguments)

    def get_tool(self, namespaced_name: str) -> MCPToolDefinition:
        if self._tools is None:
            raise ToolRegistrationError("MCP tools have not been initialized.")
        try:
            return self._tools[namespaced_name]
        except KeyError as exc:
            raise ToolRegistrationError(f"MCP tool '{namespaced_name}' is not registered.") from exc

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
                self._session = None
                self._tools = None

    async def _ensure_initialized(self) -> None:
        if self._session is not None:
            return

        stack = AsyncExitStack()
        try:
            if self.server.transport == "stdio":
                params = StdioServerParameters(
                    command=self.server.command,
                    args=list(self.server.args),
                    env=dict(self.server.env) if self.server.env is not None else None,
                    cwd=self.server.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            else:
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(headers=dict(self.server.headers or {}))
                )
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(self.server.url, http_client=http_client)
                )

            session = ClientSession(read_stream, write_stream)
            await stack.enter_async_context(session)
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    def _build_tool_definition(self, tool: mcp_types.Tool) -> MCPToolDefinition:
        description = (tool.description or "").strip() or f"Run the {tool.name} tool from {self.server.name}."
        parameters = dict(tool.inputSchema or _empty_parameters_schema())
        return MCPToolDefinition(
            server_name=self.server.name,
            tool_name=tool.name,
            namespaced_name=f"{self.server.name}__{tool.name}",
            description=description,
            parameters=parameters,
            require_approval=self.server.require_approval,
        )


class MCPBridgeManager:
    def __init__(self, servers: list[MCPServer] | None = None) -> None:
        self._bridges: dict[str, MCPClientBridge] = {}
        for server in servers or []:
            if server.name in self._bridges:
                raise ToolRegistrationError(f"MCP server '{server.name}' is already registered.")
            self._bridges[server.name] = MCPClientBridge(server)
        self._tools_by_name: dict[str, tuple[MCPClientBridge, MCPToolDefinition]] = {}
        self._initialized = False

    async def ensure_initialized(self) -> None:
        if self._initialized:
            return

        tools_by_name: dict[str, tuple[MCPClientBridge, MCPToolDefinition]] = {}
        for bridge in self._bridges.values():
            for tool in await bridge.list_tools():
                if tool.namespaced_name in tools_by_name:
                    raise ToolRegistrationError(f"Tool '{tool.namespaced_name}' is already registered.")
                tools_by_name[tool.namespaced_name] = (bridge, tool)

        self._tools_by_name = tools_by_name
        self._initialized = True

    def has_tool(self, name: str) -> bool:
        return name in self._tools_by_name

    def get_tool(self, name: str) -> MCPToolDefinition:
        try:
            return self._tools_by_name[name][1]
        except KeyError as exc:
            raise ToolRegistrationError(f"MCP tool '{name}' is not registered.") from exc

    def to_openai_tools(self) -> list[JSONObject]:
        return [tool.to_openai_tool() for _, tool in self._tools_by_name.values()]

    def tool_names(self) -> set[str]:
        return set(self._tools_by_name)

    async def call_tool(
        self,
        *,
        namespaced_name: str,
        arguments: JSONObject,
    ) -> tuple[MCPToolDefinition, mcp_types.CallToolResult]:
        bridge, tool = self._tools_by_name[namespaced_name]
        result = await bridge.call_tool(tool_name=tool.tool_name, arguments=arguments)
        return tool, result

    async def close(self) -> None:
        for bridge in self._bridges.values():
            await bridge.close()
        self._tools_by_name = {}
        self._initialized = False


def _empty_parameters_schema() -> JSONObject:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def _mcp_result_payload(result: mcp_types.CallToolResult) -> object:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", warnings="none")
    return result
