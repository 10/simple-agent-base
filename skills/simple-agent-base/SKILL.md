---
name: simple-agent-base
description: "Use this skill whenever the user wants to build, debug, review, or modify Python code that uses `simple_agent_base` or the `simple-agent-base` package. This skill is the shortcut context for 10/simple-agent-base: Agent, AgentConfig, tools, hosted tools and hosted-tool streaming events, MCP servers, streaming, structured output, usage metadata, chat snapshots, multimodal inputs, sync wrappers, custom providers, tests, and common mistakes."
---

# Simple Agent Base

Use this skill to write correct code against `simple_agent_base` without re-discovering the library. It is based on the `10/simple-agent-base` GitHub project and the published `simple-agent-base` package.

If you need exact edge-case behavior, read `references/api-notes.md`. If you need ready-to-adapt snippets, read `references/examples.md`.

## Core Mental Model

`simple_agent_base` is a small async-first Python harness for the OpenAI Responses API. It gives you a request loop, local Python tools, hosted tool passthrough, client-side MCP tool bridging, streaming events, structured output, in-memory chat state, multimodal message parts, sync wrappers, and a minimal provider interface.

It is not a full agent framework. Do not assume it has planning, durable memory, retrieval, workflow orchestration, retries, dependency-aware tool scheduling, or multi-agent primitives unless the caller asks you to build those on top.

## First Choice Rules

- Use `Agent` plus `AgentConfig`; do not invent a wrapper unless the surrounding project already has one.
- Set `system_prompt` when the agent needs durable role, behavior, style, safety, or tool-use instructions. Do not bury those instructions only in the user prompt.
- Prefer async APIs: `await agent.run(...)` and `async for event in agent.stream(...)`.
- Use `async with Agent(...) as agent:` for cleanup when possible.
- Use `@tool` for local Python functions rather than hand-writing tool schemas.
- Use `response_model=MyPydanticModel` when the caller needs typed application data.
- Use `agent.chat()` for multi-turn state; use `chat.export()` and `agent.chat_from_snapshot(...)` for persistence.
- Use `ChatMessage` plus `TextPart`, `ImagePart`, and `FilePart` for multimodal input.
- Use `hosted_tools=[{"type": "..."}]` only for provider-side tools.
- Use `MCPServer.stdio(...)` or `MCPServer.http(...)` for client-side MCP tools.
- Use sync wrappers only in non-async programs, and close with `agent.close()`.

## Public Imports

```python
from simple_agent_base import (
    Agent,
    AgentConfig,
    AgentEvent,
    AgentRunResult,
    ApprovalHandler,
    ChatMessage,
    ChatSession,
    ChatSnapshot,
    ConversationItem,
    FilePart,
    HostedToolCallUpdate,
    ImagePart,
    MCPApprovalRequest,
    MCPApprovalRequiredError,
    MCPCallRecord,
    MCPServer,
    TextPart,
    ToolCallRequest,
    ToolExecutionResult,
    ToolRegistry,
    UsageMetadata,
    tool,
)
```

Error classes are in `simple_agent_base.errors`:

```python
from simple_agent_base.errors import (
    AgentHarnessError,
    MaxTurnsExceededError,
    MCPApprovalRequiredError,
    ProviderError,
    ToolDefinitionError,
    ToolExecutionError,
    ToolRegistrationError,
)
```

## Choose The API

- One final answer: `await agent.run("...")`
- Incremental text or lifecycle events: `async for event in agent.stream("...")`
- Conversation state: `chat = agent.chat(system_prompt="...")`
- Resume saved conversation: `chat = agent.chat_from_snapshot(snapshot_or_dict)`
- Typed output: `await agent.run("...", response_model=MyModel)`
- Plain text input: pass a string.
- Message history or multimodal input: pass a list of `ChatMessage`, dicts, or strings.
- Synchronous script: `agent.run_sync(...)` or `agent.stream_sync(...)`

## Minimal Async Pattern

```python
import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="gpt-6-astra"),
        tools=[ping],
        system_prompt="You are concise.",
    ) as agent:
        result = await agent.run("Call ping with hello and tell me the result.")
        print(result.output_text)
        print(result.tool_results[0].output)


asyncio.run(main())
```

## Configuration

```python
AgentConfig(
    model="gpt-6-astra",
    api_key=None,
    base_url=None,
    max_turns=8,
    parallel_tool_calls=False,
    reasoning_effort=None,
    temperature=None,
    timeout=None,
    tool_timeout=None,
)
```

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`
- `OPENAI_REASONING_EFFORT`

Use `tool_timeout` to limit each local or MCP tool call. Use `timeout` for provider requests.

## System Prompts

Use `system_prompt` for stable agent instructions such as role, tone, tool-use policy, output style, or constraints that should apply across runs.

Agent-level default:

```python
async with Agent(
    config=AgentConfig(model="gpt-6-astra"),
    tools=[ping],
    system_prompt=(
        "You are a concise support agent. Use tools for account lookups, "
        "do not guess account data, and explain failures plainly."
    ),
) as agent:
    result = await agent.run("Check user 123.")
```

Per-call override:

```python
result = await agent.run(
    "Explain this error.",
    system_prompt="You are a patient Python tutor. Keep the answer beginner-friendly.",
)
```

Chat-level default:

```python
chat = agent.chat(system_prompt="You are concise and remember user preferences within this chat.")
```

The library sends this convenience prompt as a prepended `developer` message. It is not stored as a normal chat history message, so use snapshots to preserve a chat-level prompt.

## Local Tools

Use plain typed Python callables.

```python
from simple_agent_base import tool


@tool(name="lookup_user", description="Fetch a user record.")
def get_user(user_id: int) -> dict[str, object]:
    return {"id": user_id, "name": "Ada"}
```

Rules:

- Every parameter needs a type annotation.
- `*args` and `**kwargs` are rejected.
- Defaults become optional schema fields.
- The first docstring line becomes the default description.
- Sync tools run via `asyncio.to_thread(...)`.
- Strings are returned as-is; Pydantic models are dumped as JSON; other outputs are JSON serialized.
- Tool failures abort the run with `ToolExecutionError`; they are not converted into model-visible recovery output.
- `parallel_tool_calls=True` runs same-turn tool calls concurrently with `asyncio.gather(...)`; enable it only for independent tools.

## Structured Output

Pass a Pydantic model as `response_model`.

```python
from pydantic import BaseModel


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


result = await agent.run(
    "Use the weather tool for San Francisco and return structured data.",
    response_model=WeatherAnswer,
)

answer = result.output_data
```

`output_text` still exists. `output_data` is the parsed Pydantic object. Structured output works with normal runs, streaming, chat sessions, tools, and multimodal input.

## Streaming

Streaming yields `AgentEvent` objects.

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "tool_call_started":
        print(f"\ncalling {event.tool_call.name}")
    elif event.type == "completed" and event.result is not None:
        print(event.result.output_text)
```

Current event types:

- `text_delta`
- `reasoning_delta`
- `tool_arguments_delta`
- `hosted_tool_call_started`
- `hosted_tool_call_updated`
- `hosted_tool_call_completed`
- `tool_call_started`
- `tool_call_completed`
- `mcp_approval_requested`
- `mcp_call_started`
- `mcp_call_completed`
- `completed`

Important: streaming failures raise exceptions while the stream is consumed. There is no `error` event in the current source.

For hosted tools, streaming events carry `event.hosted_tool_call` as a `HostedToolCallUpdate` with `item_id`, `tool_type`, `status`, optional ordering fields, and sometimes the raw hosted tool item. These are provider-side lifecycle updates, not local tool executions.

## Chat Sessions

Use `ChatSession` for in-memory conversation state.

```python
chat = agent.chat(system_prompt="You are concise.")
await chat.run("My name is Taylor.")
result = await chat.run("What name did I give you?")

snapshot = chat.export()
restored = agent.chat_from_snapshot(snapshot)
```

Chat snapshots store conversation items and the chat-level `system_prompt`. They do not store `AgentConfig`, provider settings, tools, or MCP servers.

The convenience `system_prompt` is sent as a prepended `developer` message. It does not appear in `chat.history`.

## Multimodal Input

```python
from simple_agent_base import ChatMessage, FilePart, ImagePart, TextPart

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Summarize this screenshot and report."),
                ImagePart.from_file("screen.png", detail="high"),
                FilePart.from_file("report.pdf"),
            ],
        )
    ]
)
```

`ImagePart.from_file(...)` supports PNG, JPEG, WEBP, and GIF. `FilePart.from_file(...)` supports common document and text formats and sends Base64 data URLs. It does not upload through the OpenAI Files API.

## Hosted Tools

Hosted tools are provider-side tools. The library only validates and passes their dicts through.

```python
agent = Agent(
    config=AgentConfig(model="gpt-6-astra"),
    hosted_tools=[{"type": "web_search"}],
)
```

Hosted tools can be mixed with local tools and MCP servers. They do not create local `tool_call_started` or `tool_call_completed` events, and they do not appear in `result.tool_results`. During streaming, supported OpenAI hosted calls such as web search, file search, code interpreter, and image generation can emit `hosted_tool_call_started`, `hosted_tool_call_updated`, and `hosted_tool_call_completed` events. Inspect `event.hosted_tool_call` while streaming or `result.raw_responses` after completion if you need hosted tool call payloads.

Provider compatibility is backend-specific. Real OpenAI supports more hosted tools than most OpenAI-compatible proxies.

## Client-Side MCP

Use `MCPServer` for MCP servers that this library connects to locally.

```python
import sys
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPApprovalRequest, MCPServer


def approve(request: MCPApprovalRequest) -> bool:
    return request.name in {"echo", "add"}


server_path = Path("tests/fixtures/mcp_demo_server.py").resolve()

async with Agent(
    config=AgentConfig(model="gpt-6-astra"),
    mcp_servers=[
        MCPServer.stdio(
            name="demo",
            command=sys.executable,
            args=[str(server_path), "stdio"],
            allowed_tools=["echo", "add"],
            require_approval=True,
        )
    ],
    approval_handler=approve,
) as agent:
    result = await agent.run("Use demo echo with hello.")
```

MCP tools are exposed as `server__tool`, for example `demo__echo`. Name conflicts with local tools raise `ToolRegistrationError`. If `require_approval=True` and no `approval_handler` is supplied, calls raise `MCPApprovalRequiredError`.

## Sync Usage

```python
from simple_agent_base import Agent, AgentConfig


with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
    result = agent.run_sync("Say hello.")
    print(result.output_text)
```

Do not call `run_sync()` or `stream_sync()` from inside a running event loop. In async code, use `await agent.run(...)`.

## When Modifying The Library

Use the source and tests as the contract. Some docs in the repo can lag the current implementation.

Inspect these first:

- `src/simple_agent_base/agent.py`: run loop, streaming loop, hosted tools, MCP integration, cleanup, input normalization.
- `src/simple_agent_base/types.py`: public models, result fields, event types, file/image helpers.
- `src/simple_agent_base/tools/base.py`: tool signature rules, schema generation, output serialization.
- `src/simple_agent_base/tools/registry.py`: registration and execution.
- `src/simple_agent_base/mcp.py`: MCP server config, approvals, result normalization.
- `src/simple_agent_base/providers/openai.py`: Responses API request kwargs and response conversion.
- `tests/test_agent.py`, `tests/test_streaming.py`, `tests/test_tools.py`, `tests/test_mcp.py`: behavioral expectations.

Run focused checks with:

```bash
uv run pytest tests/test_agent.py tests/test_streaming.py tests/test_tools.py tests/test_mcp.py
```

## Common Mistakes To Avoid

- Do not document or implement streaming `error` events; current streams raise exceptions.
- Do not manually persist a fake developer message in chat snapshots.
- Do not use sync wrappers inside FastAPI, notebooks, or other running event loops.
- Do not enable parallel tools for shared mutable state or ordered workflows.
- Do not assume hosted tools are executed locally.
- Do not assume MCP hosted tool declarations like `{"type": "mcp"}` are implemented by this package; use `MCPServer`.
- Do not forget cleanup for agents that own providers or MCP connections.
- Do not assume `FilePart.from_file(...)` uses the OpenAI Files API.
- Do not add broad framework features unless the request clearly asks for them.

## Deeper References

- `references/api-notes.md`: source-grounded API notes, edge cases, and test contract.
- `references/examples.md`: copyable examples for common use cases.
