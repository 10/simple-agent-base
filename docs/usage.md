# Usage Guide

This guide covers the public runtime API in detail:

- `Agent`
- `ChatSession`
- message inputs
- multimodal content
- system prompts
- streaming
- snapshots
- sync wrappers

If you want exact tool behavior, read [tools.md](./tools.md). If you want typed outputs, read [structured-output.md](./structured-output.md).

## Mental Model

`simple-agent-base` gives you one small agent runtime with three main modes:

- one request with `agent.run(...)`
- one streamed request with `agent.stream(...)`
- multi-turn state with `agent.chat()`

The package is async-first. Sync wrappers exist for scripts and non-async programs, but the main API is asynchronous.

The package does not hide a large framework behind the API. It keeps a transcript, calls the provider, executes local tools if the model requests them, appends tool outputs back into the transcript, and repeats until the model returns a final answer or `max_turns` is hit.

## Main Objects

### `Agent`

`Agent` is the main entrypoint.

Typical construction:

```python
from simple_agent_base import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    tools=[...],
    system_prompt="...",
)
```

The constructor takes:

- `config`
  Required. `AgentConfig` controls the model, API connection, turn limit, timeout, and tool parallelism.
- `tools`
  Optional. You can pass a list of tool callables or a prebuilt `ToolRegistry`.
- `provider`
  Optional. If omitted, the package uses the built-in OpenAI Responses provider.
- `system_prompt`
  Optional. Convenience prompt applied as a `developer` message unless overridden per call.

Important methods:

- `await agent.run(input_data, response_model=None, system_prompt=None)`
- `async for event in agent.stream(input_data, response_model=None, system_prompt=None)`
- `agent.run_sync(...)`
- `agent.stream_sync(...)`
- `agent.chat(messages=None, system_prompt=None)`
- `agent.chat_from_snapshot(snapshot)`
- `await agent.aclose()`
- `agent.close()`

### `ChatSession`

`ChatSession` is a thin stateful wrapper around `Agent`.

Typical construction:

```python
chat = agent.chat(system_prompt="You are concise.")
```

You can also seed a chat with existing messages:

```python
from simple_agent_base import ChatMessage


chat = agent.chat(
    messages=[
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Stored."),
    ]
)
```

Important methods:

- `await chat.run(...)`
- `async for event in chat.stream(...)`
- `chat.run_sync(...)`
- `chat.stream_sync(...)`
- `chat.snapshot()`
- `chat.export()`
- `chat.reset()`

Important properties:

- `chat.history`
  Returns `ChatMessage` objects reconstructed from stored conversation items.
- `chat.items`
  Returns the raw stored conversation items.

## Choosing The Right Entry Point

Use this guide:

- one request, final answer: `await agent.run(...)`
- one request, incremental text: `async for event in agent.stream(...)`
- multi-turn chat state: `chat = agent.chat()`
- resume saved chat state: `agent.chat_from_snapshot(...)`
- sync script: `agent.run_sync(...)`
- sync streaming script: `agent.stream_sync(...)`

## Input Shapes

`input_data` can be:

- a plain string
- a sequence of message-like values

The sequence form accepts:

- `ChatMessage`
- plain dicts that validate into `ChatMessage`
- bare strings, which the package treats as user messages

### Plain String Input

The smallest call:

```python
result = await agent.run("Say hello.")
print(result.output_text)
```

This becomes one user message internally.

### Explicit Message Input

Use explicit messages when you already have history or need multimodal input:

```python
from simple_agent_base import ChatMessage

messages = [
    ChatMessage(role="user", content="My name is Anson."),
    ChatMessage(role="assistant", content="Noted."),
    ChatMessage(role="user", content="What name did I say?"),
]

result = await agent.run(messages)
```

Supported `ChatMessage.role` values:

- `user`
- `assistant`
- `developer`
- `system`

### Mixed Sequence Input

Bare strings inside a sequence are treated as user messages:

```python
messages = [
    ChatMessage(role="assistant", content="Stored."),
    "What did I say earlier?",
]

result = await agent.run(messages)
```

## Message Content

`ChatMessage.content` can be:

- a plain string
- a list of content parts

Content parts:

- `TextPart`
- `ImagePart`
- `FilePart`

### `TextPart`

```python
from simple_agent_base import TextPart

TextPart("Describe this image.")
```

### `ImagePart`

Remote URL:

```python
from simple_agent_base import ImagePart

ImagePart.from_url("https://example.com/cat.png")
```

Local file:

```python
ImagePart.from_file("cat.png", detail="high")
```

Allowed `detail` values:

- `low`
- `high`
- `auto`
- `original`

Supported local image file types:

- PNG
- JPEG
- WEBP
- GIF

Behavior of `ImagePart.from_file(...)`:

- validates that the path exists
- validates that the path is a file
- infers the MIME type from the filename
- rejects unsupported image types
- reads the file
- converts it to a Base64 data URL

### `FilePart`

Remote URL:

```python
from simple_agent_base import FilePart

FilePart.from_url("https://example.com/report.pdf")
```

Local file:

```python
FilePart.from_file("report.pdf")
```

Manual construction is also supported:

```python
FilePart(file_url="https://example.com/report.pdf")
FilePart(file_data="data:application/pdf;base64,...", filename="report.pdf")
```

Behavior of `FilePart.from_file(...)`:

- validates that the path exists
- validates that the path is a file
- infers a supported document MIME type
- rejects unsupported file types
- reads the file
- converts it to a Base64 data URL
- stores the original filename

Validation rules for `FilePart` itself:

- exactly one of `file_url` or `file_data` must be provided
- if you use `file_data`, you must also provide `filename`

Common supported file types include:

- PDF
- TXT
- CSV and TSV
- JSON
- HTML
- XML
- YAML
- Markdown
- DOC and DOCX
- XLS and XLSX
- PPT and PPTX
- RTF
- ODT

This helper does not use the OpenAI Files API. It keeps the request self-contained and works with OpenAI-compatible base URLs.

### Multimodal Example

```python
from simple_agent_base import ChatMessage, ImagePart, TextPart

result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("Describe this image."),
                ImagePart.from_file("cat.png", detail="high"),
            ],
        )
    ]
)
```

## Return Value Of `run(...)`

`agent.run(...)` and `chat.run(...)` return `AgentRunResult`.

Fields:

- `output_text`
  Final assistant text from the provider.
- `reasoning_summary`
  Final reasoning summary when reasoning is enabled and the backend returns one.
- `output_data`
  Parsed Pydantic object when you use `response_model=...`. Otherwise `None`.
- `response_id`
  Provider response id when available.
- `tool_results`
  Ordered list of `ToolExecutionResult` values for the run.
- `usage`
  Optional aggregate usage metadata across all provider turns in the run.
- `usage_by_response`
  Per-response usage metadata in provider-call order. Responses without usage are skipped.
- `raw_responses`
  Provider payloads collected across all turns.

Usage metadata is optional. APIs and custom `base_url` providers that do not return usage
produce `usage=None` and `usage_by_response=[]`. Provider-specific usage fields are preserved
on each per-response usage record's `raw` payload. Use `raw_responses` when you need the full
provider response payload for debugging.

## Streaming

Use `stream(...)` when you want incremental text or tool lifecycle events.

Example:

```python
async for event in agent.stream("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "reasoning_delta" and event.delta:
        print(f"[reasoning] {event.delta}")
    elif event.type == "completed" and event.result is not None:
        print()
        print(event.result.output_text)
        print(event.result.reasoning_summary)
```

Event types:

- `text_delta`
- `reasoning_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`

### Streaming Event Semantics

`text_delta`

- emitted for incremental text from the provider
- contains `delta`

`reasoning_delta`

- emitted for incremental reasoning summary text from the provider
- contains `delta`

`tool_call_started`

- emitted after the provider returns a final response for the turn
- one event per requested tool call
- contains `tool_call`

`tool_call_completed`

- emitted after a tool finishes
- contains `tool_result`

`completed`

- emitted once, after the final assistant answer is available
- contains the final `AgentRunResult`

### Important Streaming Behavior

- structured output only appears on the final `completed` event
- if a tool batch runs during a streamed request, you will see tool lifecycle events between turns

## Reasoning

You can enable reasoning through `AgentConfig.reasoning_effort`.

```python
agent = Agent(
    config=AgentConfig(
        model="gpt-5.6-sol",
        reasoning_effort="high",
    )
)
```

Behavior:

- when set, the provider sends `reasoning={"effort": ..., "summary": "auto"}`
- streamed reasoning summary text is surfaced as `reasoning_delta`
- the final `AgentRunResult` may include `reasoning_summary`

## System Prompts

You can set a convenience prompt at three levels:

- agent-level default
- per-call override
- chat-session default

Agent-level default:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    system_prompt="You are concise and helpful.",
)
```

Per-call override:

```python
result = await agent.run(
    "Explain async IO.",
    system_prompt="You are a patient teacher.",
)
```

Chat-session default:

```python
chat = agent.chat(system_prompt="You are a terse coding assistant.")
```

### Exact Behavior

The convenience `system_prompt` is not a separate feature in the provider. The package turns it into a `developer` message at the front of the transcript.

Important consequences:

- if you pass `system_prompt`, the package prepends a `developer` message
- if you also pass explicit `developer` or `system` messages, the package sends both
- the convenience prompt does not appear in `chat.history`
- snapshots store the session `system_prompt` separately instead of persisting a fake developer message into history
- a per-call `system_prompt` overrides the agent default for that one call
- a chat-level `system_prompt` applies to that chat unless you override it for one call

## Chat Sessions

Use a chat session when you want in-memory conversation state.

```python
chat = agent.chat()

first = await chat.run("My name is Anson.")
second = await chat.run("What name did I give you?")
```

Behavior:

- the session stores completed conversation items in memory
- each new call sends prior stored items plus the new input
- assistant replies from completed turns become part of the stored history
- tool output items are not exposed through `chat.history`

### `chat.history`

`chat.history` returns reconstructed `ChatMessage` values.

This is useful for:

- display
- logging
- exporting user-visible conversation history

It is not a byte-for-byte dump of every internal transcript item. Tool outputs are intentionally omitted.

### `chat.items`

`chat.items` returns the raw stored conversation items.

This is useful when you want the internal message representation instead of reconstructed `ChatMessage` objects.

### Resetting Chat State

```python
chat.reset()
```

This clears the stored conversation items for the session.

## Snapshot And Restore

Use snapshots when you need exact resumable chat state.

```python
chat = agent.chat(system_prompt="You are concise.")

await chat.run("My name is Anson.")
await chat.run("My favorite color is teal.")

snapshot = chat.snapshot()
payload = chat.export()
restored = agent.chat_from_snapshot(payload)
```

### Snapshot Methods

`chat.snapshot()`

- returns a typed `ChatSnapshot`

`chat.export()`

- returns a JSON-friendly dict version of the snapshot

`agent.chat_from_snapshot(snapshot_or_dict)`

- restores a `ChatSession` from a `ChatSnapshot` or compatible dict

### What A Snapshot Contains

- `version`
  Currently `"v1"`.
- `items`
  Persisted conversation items.
- `system_prompt`
  The chat session convenience prompt, if present.

### Important Snapshot Behavior

- snapshots restore chat history and the session prompt
- snapshots do not restore `AgentConfig`
- snapshots do not restore registered tools
- snapshots do not restore the provider instance
- convenience prompts are stored as `system_prompt` separately, not as a persisted message

### Streaming And Snapshots

For streamed chat sessions:

- the chat state updates only after a successful `completed` event
- if a streamed turn fails and ends with `error`, the incomplete turn is not persisted as completed chat state

## Sync Wrappers

The package is async-first, but it also supports synchronous programs.

### `run_sync(...)`

```python
agent = Agent(config=AgentConfig(model="gpt-5.6-sol"))

try:
    result = agent.run_sync("Say hello.")
    print(result.output_text)
finally:
    agent.close()
```

### `stream_sync(...)`

```python
for event in agent.stream_sync("Explain async IO in one sentence."):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
```

### Exact Sync Behavior

- sync wrappers use a dedicated background event loop thread
- the agent keeps that sync runtime alive across sync calls
- `agent.close()` shuts down the provider and the sync runtime
- if you used sync wrappers, call `agent.close()` when you are done

Restrictions:

- `run_sync()` cannot be used inside an already running event loop
- `stream_sync()` cannot be used inside an already running event loop
- `chat.run_sync()` and `chat.stream_sync()` have the same restriction

If you are already inside async code, use `await agent.run(...)` and `async for event in agent.stream(...)` instead.

## Lifecycle And Cleanup

For async usage:

```python
try:
    result = await agent.run("...")
finally:
    await agent.aclose()
```

For sync usage:

```python
try:
    result = agent.run_sync("...")
finally:
    agent.close()
```

The built-in provider owns an `AsyncOpenAI` client, so explicit cleanup is the normal pattern.

## Configuration

`AgentConfig` fields:

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
)
```

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`
- `OPENAI_REASONING_EFFORT`

Field behavior:

- `model` is required unless `OPENAI_MODEL` is set
- `api_key` defaults from `OPENAI_API_KEY`
- `base_url` points the provider at an OpenAI-compatible endpoint
- `max_turns` limits tool-call loops
- `parallel_tool_calls` enables same-turn parallel local tool execution
- `reasoning_effort` is optional and, when set, requests a reasoning summary
- `temperature` is optional and only sent if set
- `timeout` is optional and only sent if set

## Errors

For `run()` and `run_sync()`, runtime failures raise exceptions such as:

- `ProviderError`
- `ToolExecutionError`
- `MaxTurnsExceededError`

For `stream()` and `stream_sync()`, the same runtime failures raise exceptions.

## Examples

Use the repo examples for runnable reference:

- [basic_agent.py](../examples/basic_agent.py)
- [streaming.py](../examples/streaming.py)
- [chat_session.py](../examples/chat_session.py)
- [chat_persistence.py](../examples/chat_persistence.py)
- [image_input.py](../examples/image_input.py)
- [file_input.py](../examples/file_input.py)
- [chat_with_images.py](../examples/chat_with_images.py)
- [sync_agent.py](../examples/sync_agent.py)
- [sync_streaming.py](../examples/sync_streaming.py)
