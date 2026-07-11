# Simple Agent Base API Notes

These notes summarize the current behavior of the `10/simple-agent-base` GitHub project. Prefer source and tests over older prose docs when they conflict.

## Package Shape

- Python package: `simple_agent_base`
- Project name: `simple-agent-base`
- Python: `>=3.12`
- Main dependencies: `openai[aiohttp]`, `pydantic`, `pydantic-settings`, `mcp`, `httpx`
- Install from GitHub: `uv add "git+https://github.com/10/simple-agent-base.git"`
- Local development: `uv sync --dev`
- Tests: `uv run pytest`

## Public Exports

`simple_agent_base.__init__` exports:

- `Agent`, `AgentConfig`
- `AgentEvent`, `AgentRunResult`, `UsageMetadata`
- `ChatSession`, `ChatMessage`, `ChatSnapshot`, `ConversationItem`
- `TextPart`, `ImagePart`, `FilePart`
- `HostedToolCallUpdate`
- `ToolRegistry`, `tool`, `ToolCallRequest`, `ToolExecutionResult`
- `MCPServer`, `MCPApprovalRequest`, `MCPApprovalRequiredError`, `MCPCallRecord`, `ApprovalHandler`

Errors in `simple_agent_base.errors`:

- `AgentHarnessError`
- `ToolDefinitionError`
- `ToolRegistrationError`
- `ToolExecutionError`
- `MaxTurnsExceededError`
- `ProviderError`
- `MCPApprovalRequiredError`

## Agent Constructor

```python
Agent(
    config: AgentConfig,
    tools: list[Callable[..., object]] | ToolRegistry | None = None,
    provider: Provider | None = None,
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServer] | None = None,
    hosted_tools: Sequence[JSONObject] | None = None,
    approval_handler: ApprovalHandler | None = None,
)
```

If `provider` is omitted, `OpenAIResponsesProvider(config)` is used.

`tools` may be a list of callables or a ready `ToolRegistry`.

`hosted_tools` must be dicts with a non-empty string `type`. Entries are copied at construction so later caller mutations do not change the agent.

## AgentConfig

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

Validation:

- `model` is required unless `OPENAI_MODEL` is set.
- `max_turns >= 1`
- `temperature` is `0.0 <= value <= 2.0`
- `timeout > 0`
- `tool_timeout > 0`
- `reasoning_effort`: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`

Environment aliases:

- `OPENAI_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_REASONING_EFFORT`

## Run Loop

`await agent.run(input_data, response_model=None, system_prompt=None)`:

1. Normalize input to Responses-style conversation items.
2. Resolve and prepend the effective convenience prompt as a `developer` message.
3. Ensure MCP tools are initialized and names do not conflict with local tools.
4. Call `provider.create_response(...)`.
5. Append provider output items to the transcript.
6. If there are no local or MCP function calls, return `AgentRunResult`.
7. Execute requested local or MCP tools, append `function_call_output`, and repeat.
8. Raise `MaxTurnsExceededError` if the loop exceeds `config.max_turns`.

## Streaming Loop

`async for event in agent.stream(...)` mirrors the run loop but yields lifecycle events.

Current event types:

- `text_delta`: incremental assistant text.
- `reasoning_delta`: incremental reasoning summary text.
- `tool_arguments_delta`: incremental JSON argument text from function-call streaming.
- `hosted_tool_call_started`: provider-side hosted tool call became visible.
- `hosted_tool_call_updated`: provider-side hosted tool call changed status.
- `hosted_tool_call_completed`: provider-side hosted tool call completed and may include the raw item.
- `tool_call_started`: emitted once for each model-requested local or MCP function call.
- `tool_call_completed`: emitted after local or MCP call output is appended.
- `mcp_approval_requested`: emitted before awaiting approval for gated MCP calls.
- `mcp_call_started`: emitted when an approved MCP call starts.
- `mcp_call_completed`: emitted when an MCP call finishes.
- `completed`: emitted once with final `AgentRunResult`.

Failures raise while iterating. There is no `error` event type in current `AgentEvent`.

For parallel streaming tool batches, start events are emitted in model order, execution uses `asyncio.gather`, and completion events are appended in the original call order.

Hosted tool events carry `AgentEvent.hosted_tool_call` as a `HostedToolCallUpdate`. They are normal stream lifecycle events for provider-side tools, but they are not local function calls and do not produce `ToolExecutionResult` entries.

## Input Normalization

Accepted `input_data`:

- `str`: becomes one user message.
- Sequence of `ChatMessage`, plain dicts that validate as `ChatMessage`, or bare strings.

Message roles:

- `user`
- `assistant`
- `developer`
- `system`

`ChatMessage.content` can be a string or a list of content parts.

Content parts convert to provider items:

- `TextPart("...")` -> `{"type": "input_text", "text": "..."}`
- `ImagePart` -> `{"type": "input_image", "image_url": "...", "detail": "..."}`
- `FilePart` -> `{"type": "input_file", ...}`

## System Prompt Semantics

The convenience `system_prompt` is implemented as a prepended `developer` message.

Precedence:

- Per-call `system_prompt` overrides the agent default for that call.
- `agent.chat(system_prompt=...)` creates a chat-level default.
- A chat call's `system_prompt` overrides the chat default for that one call.
- `agent.chat_from_snapshot(...)` uses the snapshot's `system_prompt`, not the agent default.

Chat snapshots store `system_prompt` separately. They do not include a fake developer message.

## AgentRunResult

Fields:

- `output_text`: final assistant text.
- `reasoning_summary`: summary text when backend returns one.
- `output_data`: parsed Pydantic object when `response_model` is used.
- `response_id`: provider response id when available.
- `tool_results`: local and MCP function tool outputs in call order.
- `mcp_calls`: MCP-specific records.
- `usage`: aggregate normalized usage across provider turns, or `None`.
- `usage_by_response`: per-turn normalized usage list.
- `raw_responses`: provider responses as JSON-like dicts.

Usage aggregation sums whichever token fields are present. Provider-specific usage payloads are preserved in per-response `raw`.

## Chat Sessions

`agent.chat(messages=None, system_prompt=None)` returns `ChatSession`.

Methods:

- `await chat.run(...)`
- `async for event in chat.stream(...)`
- `chat.run_sync(...)`
- `chat.stream_sync(...)`
- `chat.snapshot()`
- `chat.export()`
- `chat.reset()`

Properties:

- `chat.history`: reconstructed `ChatMessage` list for display/logging.
- `chat.items`: raw stored conversation items.

Persistence behavior:

- Chat stores only items whose `type` is `"message"`.
- Tool output items are not exposed in `chat.history`.
- Assistant messages after tool turns are preserved.
- Streamed chat state updates only after `completed`.

Snapshots contain:

- `version`, currently `"v1"`
- `items`
- `system_prompt`

Snapshots do not restore config, provider, tools, hosted tools, MCP servers, or API keys.

## Multimodal Parts

`ImagePart.from_url(url, detail="auto")`

- Rejects blank URLs.
- `detail` is `"low"`, `"high"`, `"auto"`, or `"original"`.

`ImagePart.from_file(path, detail="auto")`

- Checks path exists and is a file.
- Supports PNG, JPEG, WEBP, GIF.
- Reads the file and creates a Base64 data URL.

`FilePart.from_url(url)`

- Rejects blank URLs.

`FilePart.from_file(path)`

- Checks path exists and is a file.
- Supports common text/document types such as PDF, TXT, CSV, TSV, JSON, HTML, XML, YAML, Markdown, DOC/DOCX, XLS/XLSX, PPT/PPTX, RTF, ODT.
- Infers MIME type from filename with fallback extension mapping.
- Reads the file and creates a Base64 data URL with `filename`.
- Does not use the OpenAI Files API.

Manual `FilePart` validation:

- Exactly one of `file_url` or `file_data` is required.
- `filename` is required when using `file_data`.

## Local Tool System

`@tool` stores prebuilt metadata and a `ToolDefinition` on the function object.

Tool definition rules:

- All parameters require type annotations.
- `*args` and `**kwargs` are invalid.
- Missing annotations raise `ToolDefinitionError`.
- Defaults become optional fields.
- Generated Pydantic arguments models forbid unknown fields.
- Default tool name is the function name.
- Default description is the first docstring line or `Run the <name> tool.`

Registration:

- Pass tools directly to `Agent` or create a `ToolRegistry`.
- Duplicate names raise `ToolRegistrationError`.
- Local names must not conflict with MCP namespaced names.

Execution:

- Model arguments are validated before execution.
- Async functions are awaited directly.
- Sync functions run in `asyncio.to_thread`.
- Tool exceptions are wrapped in `ToolExecutionError`.
- Unknown tool names raise `ToolRegistrationError`.
- `tool_timeout` wraps local and MCP execution.
- Sync tool timeout stops waiting, but cannot forcibly kill the worker thread.

Serialization:

- `str` output is sent unchanged.
- Pydantic `BaseModel` output is dumped to JSON.
- Other values are JSON serialized with `default=str`.

## Hosted Tools

`hosted_tools` are raw provider-side declarations.

```python
Agent(
    config=AgentConfig(model="gpt-5.6-sol"),
    hosted_tools=[{"type": "web_search"}],
)
```

Behavior:

- Entries are validated only for dict plus non-empty string `type`.
- Entries are passed to the provider after local and MCP function tools.
- Hosted tool output items are provider output items, not local tool calls.
- The run loop terminates if no local or MCP function calls are returned.
- `result.tool_results` remains empty for hosted-only responses.
- Streaming can emit `hosted_tool_call_started`, `hosted_tool_call_updated`, and `hosted_tool_call_completed` events for supported OpenAI hosted call item types such as `web_search_call`, `file_search_call`, `code_interpreter_call`, and `image_generation_call`.
- Inspect `result.raw_responses` for provider-specific hosted tool payloads.

Do not model OpenAI hosted MCP as `{"type": "mcp"}` for this library. Client-side MCP uses `MCPServer`.

## MCP

`MCPServer.stdio(...)`:

```python
MCPServer.stdio(
    name="demo",
    command=sys.executable,
    args=["server.py", "stdio"],
    env={"TOKEN": "abc"},
    cwd="C:/work",
    allowed_tools=["echo"],
    require_approval=True,
)
```

`MCPServer.http(...)`:

```python
MCPServer.http(
    name="deepwiki",
    url="https://mcp.deepwiki.com/mcp",
    headers={"Authorization": "Bearer ..."},
    allowed_tools=["search"],
    require_approval=False,
)
```

Behavior:

- Stdio transport requires `command`; it rejects `url` and `headers`.
- HTTP transport requires `url`; it rejects `command`, `args`, `env`, and `cwd`.
- Server names must be unique.
- Discovered tools are namespaced as `server__tool`.
- `allowed_tools=None` exposes all discovered tools.
- `allowed_tools=[]` exposes none.
- MCP result normalization prefers text blocks, then `structuredContent`, then full payload JSON.
- MCP error results raise `ToolExecutionError`.
- Transport failures are wrapped as `ToolExecutionError`.
- Approval handlers can be sync or async.
- If approval is denied, a normal tool output says `MCP tool call denied by approval handler.` and the agent continues.
- Approval waiting time is not included in `tool_timeout`.

Streaming gated MCP event order for an approved call:

1. `tool_call_started`
2. `mcp_approval_requested`
3. `mcp_call_started`
4. `mcp_call_completed`
5. `tool_call_completed`

For denied calls, there is no `mcp_call_started` or `mcp_call_completed`.

## OpenAI Provider

The built-in provider owns an `AsyncOpenAI` client using `DefaultAioHttpClient`.

Request kwargs include:

- `model`
- `input`
- `parallel_tool_calls`
- `tools`, only when non-empty
- `reasoning={"effort": ..., "summary": "auto"}`, when `reasoning_effort` is set
- `temperature`, when set
- `text_format=response_model`, when structured output is requested

Non-streaming:

- Uses `responses.create(...)` normally.
- Uses `responses.parse(...)` when `response_model` is set.

Streaming:

- Uses `responses.stream(...)`.
- Converts `response.output_text.delta` to `text_delta`.
- Converts `response.reasoning_summary_text.delta` to `reasoning_delta`.
- Converts `response.function_call_arguments.delta` to `tool_arguments_delta`.
- Gets the final response from `stream.get_final_response()`.

Provider conversion:

- Extracts `output_text`.
- Extracts reasoning summaries from reasoning output items when available.
- Parses function call arguments from JSON strings.
- Invalid or non-object tool arguments raise `ProviderError`.
- Preserves raw responses as JSON-like dicts.

## Custom Provider Contract

A custom provider implements:

```python
class Provider:
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse: ...

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...

    async def close(self) -> None: ...
```

Use normalized provider event classes from `simple_agent_base.providers.base`.

## Sync Wrappers

`run_sync` and `stream_sync` use a background event loop thread.

Rules:

- Do not call sync wrappers from inside a running event loop.
- Sync wrappers preserve reusable agent state across calls.
- `agent.close()` closes provider, MCP resources, and the sync runtime.
- `with Agent(...) as agent:` calls `close()`.

## Source Map

- `src/simple_agent_base/__init__.py`: public exports.
- `src/simple_agent_base/agent.py`: `Agent`, run/stream loops, input conversion, hosted tools, MCP execution, usage aggregation, cleanup, sync bridge calls.
- `src/simple_agent_base/chat.py`: `ChatSession`, history, snapshots, stream persistence.
- `src/simple_agent_base/config.py`: `AgentConfig`.
- `src/simple_agent_base/types.py`: public Pydantic models, content parts, result fields, event types.
- `src/simple_agent_base/errors.py`: public exceptions.
- `src/simple_agent_base/tools/base.py`: tool schema generation and output serialization.
- `src/simple_agent_base/tools/decorators.py`: `@tool`.
- `src/simple_agent_base/tools/registry.py`: registration, provider tool payloads, execution.
- `src/simple_agent_base/mcp.py`: client-side MCP bridge.
- `src/simple_agent_base/providers/base.py`: provider protocol and normalized event/result models.
- `src/simple_agent_base/providers/openai.py`: OpenAI Responses provider.
- `src/simple_agent_base/sync_utils.py`: sync runtime.

## Test Contract

Run targeted tests after behavior changes:

```bash
uv run pytest tests/test_config.py tests/test_tools.py tests/test_agent.py tests/test_streaming.py tests/test_mcp.py tests/test_openai_provider.py
```

Use tests to verify:

- system prompt precedence and no leakage into history
- chat snapshot persistence and restore
- multimodal reconstruction
- hosted tool validation and passthrough
- hosted tool streaming lifecycle updates
- usage aggregation across tool turns
- tool argument validation and failure wrapping
- streaming lifecycle event order
- sync wrapper restrictions
- MCP approvals, allowlists, namespacing, and timeouts

## Known Drift To Watch

The repo's current source supports hosted tools and MCP. Older wording in some docs says the package does not provide hosted tools or approval flows. Treat that as shorthand for "not a full framework" rather than an exact feature list.

The current source and tests say streaming raises exceptions on failure. Do not describe an `error` stream event unless the code changes.
