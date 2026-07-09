# Architecture Guide

This guide explains how `simple-agent-base` works internally.

It is for developers who want to:

- understand the runtime model
- modify the package safely
- add a new provider
- debug transcript or tool behavior

## Design Goals

The package has a narrow goal:

- wrap the OpenAI Responses API with a small, readable runtime

It tries to provide:

- one-shot runs
- streaming
- local tool execution
- structured outputs
- in-memory chat state
- multimodal input
- sync wrappers

It does not try to provide:

- orchestration
- planning
- retrieval
- memory systems
- client-side execution of hosted tools
- workflow engines

The code is deliberately small enough that one developer can understand it quickly.

## Source Map

Core files:

- [`src/simple_agent_base/__init__.py`](../src/simple_agent_base/__init__.py)
  Public exports.
- [`src/simple_agent_base/agent.py`](../src/simple_agent_base/agent.py)
  Main run and stream loops.
- [`src/simple_agent_base/chat.py`](../src/simple_agent_base/chat.py)
  Stateful chat sessions and snapshots.
- [`src/simple_agent_base/config.py`](../src/simple_agent_base/config.py)
  `AgentConfig` and environment-backed settings.
- [`src/simple_agent_base/transcript.py`](../src/simple_agent_base/transcript.py)
  Internal transcript normalization, system prompt prepending, and chat-history reconstruction.
- [`src/simple_agent_base/types.py`](../src/simple_agent_base/types.py)
  Pydantic models, multimodal parts, and tool result types.
- [`src/simple_agent_base/errors.py`](../src/simple_agent_base/errors.py)
  Public exception classes.
- [`src/simple_agent_base/sync_utils.py`](../src/simple_agent_base/sync_utils.py)
  Sync wrappers implemented on top of a background event loop thread.

Provider layer:

- [`src/simple_agent_base/providers/base.py`](../src/simple_agent_base/providers/base.py)
  Provider protocol and normalized provider result types.
- [`src/simple_agent_base/providers/openai.py`](../src/simple_agent_base/providers/openai.py)
  OpenAI Responses implementation.

Tool layer:

- [`src/simple_agent_base/tools/base.py`](../src/simple_agent_base/tools/base.py)
  Tool schema helpers and serialization.
- [`src/simple_agent_base/tools/decorators.py`](../src/simple_agent_base/tools/decorators.py)
  `@tool` decorator.
- [`src/simple_agent_base/tools/registry.py`](../src/simple_agent_base/tools/registry.py)
  Tool registry, validation, and execution.

Behavioral tests:

- [`tests/test_agent.py`](../tests/test_agent.py)
- [`tests/test_streaming.py`](../tests/test_streaming.py)
- [`tests/test_tools.py`](../tests/test_tools.py)

## Runtime Overview

The runtime centers on `Agent`.

High-level loop:

1. normalize input into conversation items
2. prepend convenience system prompt if configured
3. call the provider
4. append provider output items to the transcript
5. if the provider returned tool calls, execute them and append `function_call_output` items
6. repeat until the provider returns a final answer or `max_turns` is exceeded

That loop exists in two forms:

- `_run_transcript(...)`
  Final-result API
- `_stream_transcript(...)`
  Streaming API

## Transcript Model

The provider layer uses `ConversationItem = dict[str, Any]`.

The package does not expose a full typed transcript object graph for provider items. Instead, it keeps the internal transcript as normalized dicts that match Responses-style items closely.

Important item types:

- `message`
- `function_call_output`

Example user message item:

```python
{
    "type": "message",
    "role": "user",
    "content": "Say hello.",
}
```

Example tool output item:

```python
{
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "pong: hello",
}
```

## Input Normalization

Input normalization lives in `transcript.normalize_input(...)`.

Rules:

- a plain string becomes one user message
- a sequence is processed item by item
- bare strings inside a sequence become user messages
- non-string items validate through `ChatMessage`
- rich content parts are converted into Responses input items

Content conversion lives in:

- `message_to_item(...)`
- `content_part_to_item(...)`

Reconstruction back into `ChatMessage` values lives in:

- `messages_from_items(...)`

That reconstruction is used by `ChatSession.history`.

## Convenience System Prompt Handling

Convenience system prompts are not a separate provider feature. The package implements them by prepending a `developer` message to the transcript.

Relevant methods:

- `transcript.clean_system_prompt(...)`
- `Agent._resolve_system_prompt(...)`
- `transcript.build_transcript(...)`
- `transcript.persist_chat_items(...)`

Important consequence:

- the convenience prompt is part of the provider input transcript
- it is not stored as a normal visible history item in snapshots or `chat.history`

## `Agent.run(...)`

`run(...)` does this:

1. normalize input
2. resolve the effective system prompt
3. prepend the convenience prompt if present
4. call `_run_transcript(...)`

`_run_transcript(...)`:

- loops up to `config.max_turns`
- calls `provider.create_response(...)`
- stores `raw_response`
- extends the transcript with `response.output_items`
- returns immediately if no tool calls were requested
- otherwise executes the tool batch and appends tool outputs
- raises `MaxTurnsExceededError` if the loop never reaches a final answer

## `Agent.stream(...)`

`stream(...)` mirrors `run(...)` but yields `AgentEvent` values.

`_stream_transcript(...)`:

- loops up to `config.max_turns`
- calls `provider.stream_response(...)`
- yields `text_delta` events as they arrive
- yields `reasoning_delta`, `tool_arguments_delta`, and hosted-tool lifecycle events when the provider supplies them
- waits for the provider's final streamed response
- appends provider output items to the transcript
- if tool calls were returned, yields `tool_call_started`, executes them, then yields `tool_call_completed`
- when the final assistant answer arrives, yields one `completed` event with the full `AgentRunResult`
- raises provider, tool, MCP, and max-turn failures while the stream is being consumed

This split is deliberate:

- final-result APIs raise runtime errors
- streaming APIs emit lifecycle events for normal progress and still raise runtime errors

For example, if a tool times out during streaming, the consumer may already have received
`tool_call_started`, and then iteration raises `ToolExecutionError`.

## Provider Abstraction

The provider contract is intentionally small.

Defined in [`providers/base.py`](../src/simple_agent_base/providers/base.py):

- `create_response(...) -> ProviderResponse`
- `stream_response(...) -> AsyncIterator[ProviderEvent]`
- `close()`

Normalized provider result types:

- `ProviderResponse`
- `ProviderTextDeltaEvent`
- `ProviderReasoningDeltaEvent`
- `ProviderToolArgumentsDeltaEvent`
- `ProviderHostedToolCallEvent`
- `ProviderCompletedEvent`

This means the rest of the runtime does not need to know the exact OpenAI SDK object types.

## OpenAI Provider

The built-in provider is `OpenAIResponsesProvider`.

Responsibilities:

- own the `AsyncOpenAI` client
- build request kwargs from `AgentConfig`
- call `responses.create(...)`, `responses.parse(...)`, or `responses.stream(...)`
- convert OpenAI SDK response objects into `ProviderResponse`
- extract function calls from response output
- normalize optional usage metadata when the backend returns it
- parse tool arguments from JSON strings
- convert provider failures into `ProviderError`

The agent aggregates normalized usage across provider turns in a run. Per-response usage is
also preserved so tool-loop runs can be inspected turn by turn. Usage remains optional because
custom `base_url` providers may omit it or return provider-specific fields.

### Request Construction

The provider sends:

- `model`
- `input`
- `parallel_tool_calls`
- `tools`, when any are registered
- `temperature`, when set
- `text_format`, when a `response_model` is provided

### Response Conversion

The OpenAI provider converts:

- output items into plain dicts
- function call items into `ToolCallRequest`
- `output_text` into the normalized final text field
- parsed structured output into `output_data`

Invalid JSON tool arguments raise `ProviderError`.

## Tool System

The tool system has three layers:

1. tool definition
2. tool registration
3. tool execution

### Tool Definition

`tools/base.py` builds a `ToolDefinition` from a Python callable by:

- inspecting its signature
- rejecting unsupported parameter kinds
- building a generated Pydantic arguments model
- deriving name and description
- producing JSON schema

### Decorator

`@tool` stores tool metadata and a prebuilt definition on the function object.

Relevant attributes:

- `__simple_agent_base_tool_metadata__`
- `__simple_agent_base_tool_definition__`

### Registry

`ToolRegistry` stores tool definitions by name, enforces uniqueness, exposes OpenAI-compatible tool payloads, validates call arguments, and executes tools.

Execution path:

1. look up tool by name
2. validate arguments through the generated arguments model
3. execute async tools directly
4. execute sync tools through `asyncio.to_thread(...)`
5. serialize the return value into a string

## Parallel Tool Calls

`Agent._execute_tool_batch(...)` handles same-turn tool execution.

Behavior:

- if `parallel_tool_calls` is `False`, tools run sequentially
- if `parallel_tool_calls` is `True`, tools run with `asyncio.gather(...)`
- final result order still follows the model's original call order

The package does not infer dependency safety. It trusts the caller to enable parallel execution only for independent tools.

## Chat Session Internals

`ChatSession` is a wrapper around `Agent` plus stored items.

It stores:

- `_agent`
- `_items`
- `_system_prompt`

Important behavior:

- `history` reconstructs `ChatMessage` values from stored items
- `snapshot()` captures `items` plus `system_prompt`
- `export()` serializes the snapshot to a JSON-friendly dict
- `run(...)` and `stream(...)` create a transcript from stored items plus new input
- after a successful turn, the chat stores only persistable message items

Current rule:

- only items with `type == "message"` are stored in chat state

That means:

- tool output items are not preserved in chat history
- assistant messages produced after tool turns are preserved

### Streaming Chat Updates

For `ChatSession.stream(...)`:

- the session updates its stored items only after a `completed` event
- if the streamed turn ends in `error`, the incomplete turn is not stored as a completed state update

## Sync Runtime

Sync wrappers are implemented in [`sync_utils.py`](../src/simple_agent_base/sync_utils.py).

Core components:

- `ensure_sync_allowed(...)`
- `SyncRuntime`

`SyncRuntime` owns:

- a background thread
- an event loop running in that thread

It provides:

- `run(...)`
  Submit one awaitable and wait for the result
- `iterate(...)`
  Bridge an async iterator into a sync iterator using a queue

Why this exists:

- `asyncio.run(...)` is fine for one call
- it is not enough for a reusable sync API that also needs streaming
- a persistent background loop gives the package a reusable sync bridge

## Errors

Public errors live in [`errors.py`](../src/simple_agent_base/errors.py):

- `AgentHarnessError`
- `ToolDefinitionError`
- `ToolRegistrationError`
- `ToolExecutionError`
- `MaxTurnsExceededError`
- `ProviderError`

General rule:

- configuration and synchronous call misuse raise immediately
- final-result APIs raise runtime errors
- streaming APIs raise runtime errors while the stream is being consumed

There is no `error` event type in `AgentEvent`. Streaming events describe normal lifecycle
progress; failures are surfaced as exceptions.

## Tests As Behavioral Spec

The test suite is the best reference for exact behavior.

Use it when changing semantics around:

- system prompt precedence
- chat snapshot persistence
- multimodal reconstruction
- streaming event order
- tool failure behavior
- sync wrapper constraints

Key files:

- [`tests/test_agent.py`](../tests/test_agent.py)
- [`tests/test_streaming.py`](../tests/test_streaming.py)
- [`tests/test_tools.py`](../tests/test_tools.py)

## Extending The Package

If you add a feature, prefer these rules:

- keep the public API small
- preserve the normalized provider boundary
- add tests before or alongside behavior changes
- update both the README and `docs/` when behavior changes
- avoid adding framework-level abstractions unless the package clearly needs them
