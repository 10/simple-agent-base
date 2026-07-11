# Simple Agent Base Examples

Copy and adapt these examples instead of reinventing the library patterns.

## Async Agent With Local Tool

```python
import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def get_weather(city: str) -> str:
    """Return weather information for a city as JSON text."""
    data = {
        "san francisco": '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
        "phoenix": '{"city":"Phoenix","temperature_f":92,"summary":"Hot and sunny"}',
    }
    return data.get(city.lower(), f'{{"city":"{city}","temperature_f":70,"summary":"Mild"}}')


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="gpt-5.6-sol", tool_timeout=30.0),
        tools=[get_weather],
        system_prompt="Answer briefly and use tools when current data is needed.",
    ) as agent:
        result = await agent.run("Use the weather tool for San Francisco.")
        print(result.output_text)
        for tool_result in result.tool_results:
            print(tool_result.name, tool_result.arguments, tool_result.output)


asyncio.run(main())
```

## Structured Output With Tool Result

```python
from pydantic import BaseModel
from simple_agent_base import Agent, AgentConfig, tool


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@tool
async def get_weather(city: str) -> str:
    """Return weather information for a city as JSON text."""
    return '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}'


async with Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    tools=[get_weather],
) as agent:
    result = await agent.run(
        "Use the weather tool for San Francisco and return structured data.",
        response_model=WeatherAnswer,
    )

answer = result.output_data
assert isinstance(answer, WeatherAnswer)
```

## Streaming UI Loop

```python
async for event in agent.stream("Use the ping tool, then summarize the result."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "reasoning_delta" and event.delta:
        print(f"[reasoning] {event.delta}")
    elif event.type == "tool_arguments_delta" and event.delta:
        pass
    elif event.type in {
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
    } and event.hosted_tool_call is not None:
        print(f"\n[hosted] {event.hosted_tool_call.tool_type}: {event.hosted_tool_call.status}")
    elif event.type == "tool_call_started" and event.tool_call is not None:
        print(f"\n[tool] starting {event.tool_call.name}")
    elif event.type == "tool_call_completed" and event.tool_result is not None:
        print(f"\n[tool] {event.tool_result.name}: {event.tool_result.output}")
    elif event.type == "completed" and event.result is not None:
        print("\nFinal:", event.result.output_text)
```

Wrap the consuming loop in `try/except` if you need to display provider or tool failures. Current streams raise exceptions; they do not yield an `error` event.

## Chat Session With Snapshot Persistence

```python
async with Agent(config=AgentConfig(model="gpt-5.6-sol")) as agent:
    chat = agent.chat(system_prompt="You are concise.")

    await chat.run("My name is Taylor.")
    second = await chat.run("What name did I give you?")
    print(second.output_text)

    payload = chat.export()

    restored = agent.chat_from_snapshot(payload)
    third = await restored.run("Repeat it once more.")
    print(third.output_text)
```

Persist `payload` in your own storage. It contains chat items and the chat prompt, not API keys, config, tools, hosted tools, or MCP server definitions.

## Multimodal Input

```python
from simple_agent_base import ChatMessage, FilePart, ImagePart, TextPart


message = ChatMessage(
    role="user",
    content=[
        TextPart("Compare this screenshot with the attached checklist."),
        ImagePart.from_file("screen.png", detail="high"),
        FilePart.from_file("checklist.pdf"),
    ],
)

result = await agent.run([message])
print(result.output_text)
```

Use `from_url(...)` for hosted image or file URLs. Local `from_file(...)` helpers embed Base64 data URLs.

## Hosted Web Search

```python
async with Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    hosted_tools=[{"type": "web_search"}],
) as agent:
    result = await agent.run("Find recent Python 3.13 release highlights.")
    print(result.output_text)
```

Hosted tools are provider-side. No local tool function is called, and `result.tool_results` remains empty for hosted-only responses.

When using `agent.stream(...)`, supported hosted calls can emit `hosted_tool_call_started`, `hosted_tool_call_updated`, and `hosted_tool_call_completed` events with details on `event.hosted_tool_call`.

## Client-Side MCP Over Stdio

```python
import sys
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPApprovalRequest, MCPServer


def approve(request: MCPApprovalRequest) -> bool:
    print(f"Approve {request.server_name}.{request.name}: {request.arguments}")
    return request.name in {"echo", "add"}


server_path = Path("tests/fixtures/mcp_demo_server.py").resolve()

async with Agent(
    config=AgentConfig(model="gpt-5.6-sol", tool_timeout=15.0),
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
    result = await agent.run("Use demo add with 2 and 3, then answer briefly.")

for call in result.mcp_calls:
    print(call.server_name, call.name, call.arguments, call.output, call.error)
```

MCP tools are presented to the model as namespaced function tools like `demo__add`.

## Client-Side MCP Over Streamable HTTP

```python
from simple_agent_base import Agent, AgentConfig, MCPServer


async with Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    mcp_servers=[
        MCPServer.http(
            name="docs",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
            allowed_tools=["search"],
            require_approval=False,
        )
    ],
) as agent:
    result = await agent.run("Search docs for the install command.")
```

Use an allowlist when exposing a large MCP server so the model sees only relevant tools.

## Sync Script

```python
from simple_agent_base import Agent, AgentConfig, tool


@tool
def slugify(title: str) -> str:
    """Convert a title into a URL slug."""
    return title.lower().replace(" ", "-")


with Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    tools=[slugify],
) as agent:
    result = agent.run_sync("Slugify 'Simple Agent Base'.")
    print(result.output_text)
```

Do not use this inside async frameworks. Use async APIs there.

## Custom Provider Skeleton

```python
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig
from simple_agent_base.providers.base import (
    ProviderCompletedEvent,
    ProviderEvent,
    ProviderResponse,
    ProviderTextDeltaEvent,
)
from simple_agent_base.types import ConversationItem, JSONObject


class StaticProvider:
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            response_id="static_1",
            output_text="Hello from a custom provider.",
            output_items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello from a custom provider."}],
                }
            ],
            raw_response={"id": "static_1"},
        )

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        response = await self.create_response(
            input_items=input_items,
            tools=tools,
            response_model=response_model,
        )
        yield ProviderTextDeltaEvent(delta=response.output_text)
        yield ProviderCompletedEvent(response=response)

    async def close(self) -> None:
        return None


async with Agent(
    config=AgentConfig(model="static-demo"),
    provider=StaticProvider(),
) as agent:
    result = await agent.run("Say hello.")
```

## Test Pattern With Fake Provider

```python
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig, tool
from simple_agent_base.providers.base import ProviderEvent, ProviderResponse
from simple_agent_base.types import ConversationItem


class FakeProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        self.calls.append({"input_items": list(input_items), "tools": list(tools)})
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


provider = FakeProvider(
    [
        ProviderResponse(
            response_id="resp_1",
            tool_calls=[
                {
                    "call_id": "call_1",
                    "name": "ping",
                    "arguments": {"message": "hello"},
                    "raw_arguments": '{"message":"hello"}',
                }
            ],
            output_items=[],
            raw_response={"id": "resp_1"},
        ),
        ProviderResponse(
            response_id="resp_2",
            output_text="done",
            output_items=[],
            raw_response={"id": "resp_2"},
        ),
    ]
)

agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)
result = await agent.run("Use ping.")
assert result.tool_results[0].output == "pong: hello"
```
