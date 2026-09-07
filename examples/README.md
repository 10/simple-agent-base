# Examples

These examples show the intended ways to use the harness.

Run any example from the repo root:

```bash
uv run python examples/basic_agent.py
```

All examples assume:

- you already ran `uv sync`
- you are running from the repo root

Most examples assume `OPENAI_API_KEY` is set. `custom_provider.py` uses a local
static provider and does not require an API key.

The OpenAI examples use `gpt-6-astra`. The MCP examples also accept an
`OPENAI_MODEL` override; `AgentConfig()` without an explicit model requires it.

For Astra, leave `temperature` unset. Use `reasoning_effort="low"` for lighter
reasoning; the reasoning and streaming examples retain `"high"`. See the
[Astra model guide](https://developers.openai.com/api/docs/guides/latest-model).

## Start Here

If you are new to the repo, read these in this order:

1. `basic_agent.py`
2. `structured_output.py`
3. `streaming.py`
4. `reasoning.py`
5. `chat_session.py`
6. `parallel_tools.py`

## Example Guide

### `basic_agent.py`

Use this first.

Shows:

- the smallest useful `Agent`
- a tool defined with `@tool`
- a normal `await agent.run(...)` call
- how to read `output_text` and `tool_results`

### `custom_provider.py`

Use this when you want to see how provider injection works.

Shows:

- a minimal provider object
- `agent.run(...)` without an OpenAI API key
- `agent.stream(...)` with provider events
- where to adapt test doubles, demos, or OpenAI-compatible providers

### `sync_agent.py`

Use this when your program is fully synchronous.

Shows:

- `agent.run_sync(...)`
- `agent.close()`

### `sync_streaming.py`

Use this when you want streaming output from sync code.

Shows:

- `agent.stream_sync(...)`
- handling `text_delta` and `completed` events

### `reasoning.py`

Use this when you want to request reasoning and inspect the streamed summary.

Shows:

- `AgentConfig(reasoning_effort="high")`
- handling `reasoning_delta`
- reading `result.reasoning_summary`

### `chat_session.py`

Use this when you want the harness to keep conversation history for you.

Shows:

- `chat = agent.chat()`
- follow-up calls with `chat.run(...)`
- `chat.history`

### `chat_persistence.py`

Use this when you want to save and resume chat state.

Shows:

- `chat.snapshot()`
- `chat.export()`
- `agent.chat_from_snapshot(...)`

### `system_prompt.py`

Use this when you want behavior steering without manually building message lists.

Shows:

- agent-level `system_prompt`
- per-run `system_prompt`
- chat-session `system_prompt`

### `parallel_tools.py`

Use this only when your tools are independent and safe to run together.

Shows:

- `AgentConfig(parallel_tool_calls=True)`
- concurrent same-turn tool execution

### `image_input.py`

Use this for a one-turn image prompt.

Shows:

- `ChatMessage`
- `TextPart`
- `ImagePart.from_file(...)`

### `file_input.py`

Use this for a one-turn document prompt.

Shows:

- `ChatMessage`
- `TextPart`
- `FilePart.from_file(...)`

### `chat_with_images.py`

Use this when the first turn contains an image and later turns refer back to it.

Shows:

- multimodal chat history
- follow-up questions after an image turn

### `structured_output.py`

Use this when you want a typed final result.

Shows:

- `response_model=...`
- reading `result.output_data`

### `structured_with_tools.py`

Use this when the model must call a tool and still return typed output.

Shows:

- tools plus structured output in one run

### `mcp_server.py`

Use this when you want to bridge a client-side MCP server into the agent.

Shows:

- `MCPServer.stdio(...)`
- `Agent(mcp_servers=[...])`
- reading `result.mcp_calls`
- an `approval_handler` with `require_approval=True`

### `mcp_http_server.py`

Use this when you want the same client-side MCP bridge flow over streamable HTTP.

Shows:

- `MCPServer.http(...)`
- starting a local demo HTTP MCP server
- reading `result.mcp_calls`
- an `approval_handler` with `require_approval=True`

### `streaming.py`

Use this when you want progressive output or tool lifecycle events.

Shows:

- `agent.stream(...)`
- `text_delta`
- `reasoning_delta`
- `tool_call_started`
- `tool_call_completed`
- `completed`
