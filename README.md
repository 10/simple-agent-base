<div align="center">

# simple-agent-base

*Async-first Python base for OpenAI agents, without the framework.*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-simple--agent--base-006dad?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/simple-agent-base/)
[![API](https://img.shields.io/badge/API-OpenAI%20Responses-412991?style=flat&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference/responses)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

</div>

---

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


asyncio.run(main())
```

The request/tool loop, local tools, structured output, streaming, chat history,
images and files, MCP bridging, and sync wrappers. No planning, retrieval,
memory, orchestration, or multi-agent primitives.

## Features

- **Tools are plain functions** — `@tool` on any async or sync function, with types read from the annotations.
- **Streaming with real events** — text and reasoning deltas, tool call lifecycle, MCP approvals.
- **Structured output** — hand it a Pydantic model, get `output_data` back.
- **MCP over stdio or HTTP** — discovered tools appear to the model as normal functions.
- **Async-first, sync when you need it** — `run_sync()` and `stream_sync()` for scripts.

## Install

```bash
uv add simple-agent-base           # or: pip install simple-agent-base
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-6-astra"
```

Python 3.12+. From a checkout, `uv sync`.

## Agent

```python
result = await agent.run("Say hello.")

async for event in agent.stream("Explain async IO."):
    ...

chat = agent.chat(system_prompt="Be brief.")
await chat.run("My name is Anson.")
await chat.run("What is my name?")
```

`Agent` supports `async with` (and `with` in sync code); `aclose()` and
`close()` are there if you want to manage it yourself.

Each run converts input to Responses API items, sends `system_prompt` as a
`developer` message, then loops: if the model returns tool calls, local or MCP
tools run and their outputs are appended. It repeats until a final response or
`max_turns`.

`AgentRunResult` carries `output_text`, `output_data`, `tool_results`,
`mcp_calls`, `reasoning_summary`, `response_id`, `usage`, `usage_by_response`,
and `raw_responses`.

Main exports: `Agent`, `AgentConfig`, `ChatSession`, `ChatMessage`, `TextPart`,
`ImagePart`, `FilePart`, `ToolRegistry`, `tool`, `MCPServer`.

## Tools

```python
@tool
def lookup_user(user_id: int) -> str:
    """Fetch a user record."""
    return '{"id": 1, "name": "Ada"}'
```

Parameters need type annotations. `*args` and `**kwargs` are rejected. The first
docstring line becomes the description unless you pass
`@tool(name=..., description=...)`.

Two `AgentConfig` knobs: `parallel_tool_calls=True` runs same-turn calls
together — only for independent tools — and `tool_timeout=30.0` caps each call,
raising `ToolExecutionError`. A sync tool's timeout stops the wait, but Python
cannot kill the worker thread.

### Hosted tools

Provider-side tools you declare but do not implement:

```python
agent = Agent(
    config=AgentConfig(model="gpt-6-astra"),
    hosted_tools=[{"type": "web_search"}],
)
```

Entries pass through unchanged. OpenAI supports `web_search`, `file_search`,
`code_interpreter`, `image_generation`, and `computer_use`; proxies and
self-hosted servers usually support fewer, and rejections surface as provider
errors. Hosted calls skip `result.tool_results` but do emit
`hosted_tool_call_*` streaming events.

## Streaming

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "completed":
        print(event.result.output_text)
```

```
text_delta                 tool_call_started
reasoning_delta            tool_call_completed
tool_arguments_delta       mcp_approval_requested
hosted_tool_call_started   mcp_call_started
hosted_tool_call_updated   mcp_call_completed
hosted_tool_call_completed completed
```

## Structured Output

```python
class Person(BaseModel):
    name: str
    age: int


result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)
print(result.output_data)
```

Works with normal runs, streaming, and tool calls.

## Chat Sessions

```python
chat = agent.chat(system_prompt="You are concise.")
await chat.run("My name is Anson.")
result = await chat.run("What is my name?")

payload = chat.export()
restored = agent.chat_from_snapshot(payload)
```

History is in memory. Snapshots hold conversation items and the chat-level
`system_prompt` — not model config, tools, or provider settings.

## Images and Files

```python
result = await agent.run([
    ChatMessage(role="user", content=[
        TextPart("Describe this image."),
        ImagePart.from_file("cat.png"),
    ])
])
```

`FilePart.from_file(...)` for local documents, `from_url(...)` for hosted ones.
Local helpers inline files as Base64 data URLs rather than using the Files API.

## MCP

```python
agent = Agent(
    config=AgentConfig(model="gpt-6-astra"),
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

`MCPServer.stdio(...)` and `MCPServer.http(...)`. Discovered tools are
namespaced `server__tool`. Narrow them with `allowed_tools`, or set
`require_approval=True` with an `approval_handler` to confirm calls locally.

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

Read from the environment: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`,
`OPENAI_REASONING_EFFORT`.

The examples use GPT-6 Astra. Leave `temperature=None` and set
`reasoning_effort="low"` for lighter reasoning.

For synchronous programs:

```python
with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
    print(agent.run_sync("Say hello.").output_text)
```

Do not call `run_sync()` or `stream_sync()` from inside a running event loop.

## Layout

```
simple-agent-base/
├── src/simple_agent_base/
│   ├── agent.py              # run/stream loop and turn handling
│   ├── chat.py               # ChatSession and snapshots
│   ├── config.py             # AgentConfig and environment
│   ├── mcp.py                # MCPServer, stdio and http transports
│   ├── tools/                # @tool decorator, registry, schemas
│   ├── providers/            # provider interface and the OpenAI one
│   ├── transcript.py         # Responses item conversion
│   ├── types.py              # messages, parts, results, events
│   └── sync_utils.py         # run_sync / stream_sync wrappers
├── examples/                 # 18 runnable scripts
├── skills/simple-agent-base/ # agent skill
└── docs/                     # usage, tools, architecture
```

## Docs

Start with [basic_agent.py](examples/basic_agent.py),
[structured_output.py](examples/structured_output.py),
[streaming.py](examples/streaming.py),
[chat_session.py](examples/chat_session.py), and
[mcp_server.py](examples/mcp_server.py).

Then [docs/usage.md](docs/usage.md), [docs/tools.md](docs/tools.md),
[docs/structured-output.md](docs/structured-output.md),
[docs/architecture.md](docs/architecture.md), and
[docs/development.md](docs/development.md).

## Development

```bash
uv sync --dev
uv run pytest                             # no API key needed
uv run python scripts/live_e2e_test.py    # needs one
```
