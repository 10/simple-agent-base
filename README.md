# Simple Agent Base

`simple-agent-base` is a small async-first Python package for building OpenAI Responses API agents.

It gives you the pieces most small agent projects need: a request/tool loop, local Python tools, structured outputs, streaming events, chat history, image and file input, MCP tool bridging, and sync wrappers.

It is intentionally not a full agent framework. It does not include planning, retrieval, memory systems, workflow orchestration, or multi-agent primitives.

## Quick Start

Requirements:

- Python `3.12+`
- An OpenAI API key
- `uv` or `pip`

### 1. Install

From PyPI:

```bash
python -m pip install simple-agent-base
```

With `uv`:

```bash
uv add simple-agent-base
```

From GitHub:

```bash
python -m pip install "git+https://github.com/10/simple-agent-base.git"
```

From a local checkout:

```bash
uv sync
```

### 2. Configure

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.6-sol"
```

You can also pass `model` directly through `AgentConfig`.

### 3. Run an agent

```python
import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="gpt-5.6-sol"),
        tools=[ping],
        system_prompt="You are concise.",
    ) as agent:
        result = await agent.run("Call ping with hello and tell me the result.")
        print(result.output_text)


asyncio.run(main())
```

`Agent` supports `async with` (and `with` for sync code) so cleanup happens automatically. If you prefer explicit lifecycle management, `await agent.aclose()` and `agent.close()` still work.

## Core API

Most projects use these exports:

- `Agent`
- `AgentConfig`
- `ChatSession`
- `ChatMessage`
- `TextPart`, `ImagePart`, `FilePart`
- `ToolRegistry`
- `tool`
- `MCPServer`

Common calls:

```python
result = await agent.run("Say hello.")

async for event in agent.stream("Explain async IO."):
    ...

chat = agent.chat(system_prompt="Be brief.")
await chat.run("My name is Anson.")
await chat.run("What is my name?")
```

`AgentRunResult` includes:

- `output_text`
- `output_data` for structured output
- `tool_results`
- `mcp_calls`
- `reasoning_summary`
- `response_id`
- `usage`
- `usage_by_response`
- `raw_responses`

## How It Works

1. `Agent.run(...)` or `Agent.stream(...)` receives a string or message list.
2. The input is converted to Responses API items.
3. A convenience `system_prompt` is sent as a `developer` message.
4. The OpenAI provider sends the request.
5. If the model returns tool calls, local or MCP tools run and their outputs are appended.
6. The loop repeats until the model returns a final response or `max_turns` is reached.

## Tools

Use `@tool` on async or sync Python functions:

```python
from simple_agent_base import tool


@tool
def lookup_user(user_id: int) -> str:
    """Fetch a user record."""
    return '{"id": 1, "name": "Ada"}'
```

Tool parameters must have type annotations. `*args` and `**kwargs` are rejected. The first docstring line becomes the tool description unless you override it:

```python
@tool(name="lookup_user", description="Fetch a user record.")
def get_user(user_id: int) -> str:
    return '{"id": 1, "name": "Ada"}'
```

Parallel same-turn tool execution is opt-in:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol", parallel_tool_calls=True),
    tools=[get_weather, get_news],
)
```

Only enable it for independent tools.

Set `tool_timeout` when each local or MCP tool call should have a maximum runtime:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol", tool_timeout=30.0),
    tools=[lookup_user],
)
```

Timeouts raise `ToolExecutionError`. For sync tools, the timeout stops waiting for the result, but Python cannot forcibly stop the worker thread.

### Hosted Tools

Some providers (notably OpenAI) execute tools server-side and return the result directly in the response. These do not have a Python implementation — you just declare them and the provider handles execution.

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    hosted_tools=[{"type": "web_search"}],
)

result = await agent.run("What's new in Python 3.13?")
print(result.output_text)
```

Hosted tool entries are passed through to the provider unchanged. Common types on the OpenAI Responses API include `web_search`, `file_search`, `code_interpreter`, `image_generation`, and `computer_use`.

Support depends on the provider. Real OpenAI supports the full set; OpenAI-compatible proxies and self-hosted servers usually support a subset or none. If your provider rejects a tool type, the error surfaces from the provider, not from this library.

Hosted tools do not appear in `result.tool_results`, but streaming can emit `hosted_tool_call_started`, `hosted_tool_call_updated`, and `hosted_tool_call_completed` events for supported provider-side calls.

## Streaming

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "completed":
        print(event.result.output_text)
```

Event types include:

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

## Structured Output

Pass a Pydantic model as `response_model`:

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)

print(result.output_data)
```

Structured output works with normal runs, streaming, and tool calls.

## Chat Sessions

`ChatSession` keeps in-memory history:

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
result = await chat.run("What is my name?")

print(result.output_text)
print(chat.history)
```

Snapshots can be stored and restored:

```python
payload = chat.export()
restored = agent.chat_from_snapshot(payload)
```

Snapshots include conversation items and the chat-level `system_prompt`. They do not include model config, tools, or provider settings.

## Images and Files

Use content parts when a message needs more than plain text:

```python
from simple_agent_base import ChatMessage, ImagePart, TextPart


result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Describe this image."),
                ImagePart.from_file("cat.png"),
            ],
        )
    ]
)
```

Use `FilePart.from_file(...)` for local documents or `from_url(...)` for hosted files. Local helpers convert files to Base64 data URLs; they do not use the OpenAI Files API.

## MCP Tools

Client-side MCP servers can be exposed to the model as function tools:

```python
import sys
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPServer


server_path = Path("tests/fixtures/mcp_demo_server.py").resolve()

agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    mcp_servers=[
        MCPServer.stdio(
            name="demo",
            command=sys.executable,
            args=[str(server_path), "stdio"],
            require_approval=False,
        )
    ],
)
```

Discovered MCP tools are namespaced as `server__tool`. Use `allowed_tools` to expose only specific tools, and set `require_approval=True` with an `approval_handler` when calls need local confirmation.

Supported transports:

- `MCPServer.stdio(...)`
- `MCPServer.http(...)`

## Configuration

```python
AgentConfig(
    model="gpt-5.6-sol",
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

## Sync Usage

The package is async-first, but synchronous programs can use:

```python
with Agent(config=AgentConfig(model="gpt-5.6-sol")) as agent:
    result = agent.run_sync("Say hello.")
    print(result.output_text)
```

`agent.close()` is also available if you'd rather manage the lifecycle yourself.

Do not call `run_sync()` or `stream_sync()` from inside an existing event loop.

## Examples and Docs

Start with:

- [examples/basic_agent.py](examples/basic_agent.py)
- [examples/structured_output.py](examples/structured_output.py)
- [examples/streaming.py](examples/streaming.py)
- [examples/chat_session.py](examples/chat_session.py)
- [examples/mcp_server.py](examples/mcp_server.py)

More details:

- [docs/usage.md](docs/usage.md)
- [docs/tools.md](docs/tools.md)
- [docs/structured-output.md](docs/structured-output.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/development.md](docs/development.md)

## Development

Install development dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run the live provider check:

```bash
uv run python scripts/live_e2e_test.py
```

Unit tests do not require an API key. The live script does.
