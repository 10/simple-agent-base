# Development Guide

This guide covers local development for `simple-agent-base`.

It is for contributors who want to:

- set up the repo
- run tests
- verify behavior against the real OpenAI provider
- understand where to make changes
- keep docs and examples aligned with the code

## Requirements

- Python 3.12 or newer
- `uv` recommended for dependency management

The project also works with `pip`, but the repo examples and local workflow use `uv`.

## Local Setup

Install dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --dev
```

If you prefer `pip`:

```bash
python -m pip install -e ".[dev]"
```

## Environment Variables

The package reads these values from the environment:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`

For local development against the real provider, set at least:

```bash
$env:OPENAI_API_KEY="your_key_here"
```

Optionally:

```bash
$env:OPENAI_MODEL="gpt-6-astra"
```

The repo includes [`.env.example`](../.env.example) as a template.

## Project Layout

```text
simple-agent-base/
|-- src/
|   `-- simple_agent_base/
|       |-- __init__.py
|       |-- agent.py
|       |-- chat.py
|       |-- config.py
|       |-- errors.py
|       |-- sync_utils.py
|       |-- types.py
|       |-- providers/
|       |   |-- base.py
|       |   `-- openai.py
|       `-- tools/
|           |-- base.py
|           |-- decorators.py
|           `-- registry.py
|-- examples/
|-- scripts/
|-- tests/
|-- docs/
|-- pyproject.toml
`-- README.md
```

## Where To Make Changes

Public API exports:

- [`src/simple_agent_base/__init__.py`](../src/simple_agent_base/__init__.py)

Core runtime behavior:

- [`src/simple_agent_base/agent.py`](../src/simple_agent_base/agent.py)

Chat sessions and persistence:

- [`src/simple_agent_base/chat.py`](../src/simple_agent_base/chat.py)

Settings and environment-backed config:

- [`src/simple_agent_base/config.py`](../src/simple_agent_base/config.py)

Public types and multimodal helpers:

- [`src/simple_agent_base/types.py`](../src/simple_agent_base/types.py)

Provider abstraction:

- [`src/simple_agent_base/providers/base.py`](../src/simple_agent_base/providers/base.py)

OpenAI implementation:

- [`src/simple_agent_base/providers/openai.py`](../src/simple_agent_base/providers/openai.py)

Tool schema and execution:

- [`src/simple_agent_base/tools/base.py`](../src/simple_agent_base/tools/base.py)
- [`src/simple_agent_base/tools/decorators.py`](../src/simple_agent_base/tools/decorators.py)
- [`src/simple_agent_base/tools/registry.py`](../src/simple_agent_base/tools/registry.py)

Sync bridge:

- [`src/simple_agent_base/sync_utils.py`](../src/simple_agent_base/sync_utils.py)

## Tests

Run the unit test suite:

```bash
uv run pytest
```

The unit tests:

- use fake providers
- do not require a real API key
- cover the public behavioral contract

Current test files:

- [`tests/test_agent.py`](../tests/test_agent.py)
- [`tests/test_streaming.py`](../tests/test_streaming.py)
- [`tests/test_tools.py`](../tests/test_tools.py)

## What The Tests Cover

The tests verify:

- basic runs
- sync wrappers
- cleanup behavior
- system prompt precedence
- chat history persistence
- snapshot and restore
- multimodal input normalization
- file and image helpers
- tool registration and schema generation
- sequential and parallel tool execution
- structured output
- streaming event order
- streaming error behavior

When you change runtime behavior, update tests in the same commit.

## Live End-To-End Verification

Use the live verification script when you want to test the real OpenAI path:

```bash
uv run python scripts/live_e2e_test.py
```

To run only the client-side MCP checks:

```bash
uv run python scripts/live_e2e_test.py --mcp-only
```

This script exercises:

- plain text requests
- structured output
- system prompts
- sync usage
- tool calls
- parallel tool calls
- chat history
- chat persistence
- file input
- streaming
- image structured output
- image follow-up in chat

Unlike the unit tests, the live script:

- requires a real API key
- talks to the configured provider
- is slower and more environment-dependent
- uses the repo's local demo MCP server for MCP coverage

Use it when:

- changing provider behavior
- changing multimodal handling
- changing sync wrappers
- verifying behavior against a real model

### Client-Side MCP Note

The MCP checks use a local stdio server from `tests/fixtures/mcp_demo_server.py`.
That means they only require normal function-tool support from the configured
provider. OpenAI-compatible backends such as OpenRouter can run these checks as
long as they support standard Responses API function calling.

## Running Examples

Run examples from the repo root:

```bash
uv run python examples/basic_agent.py
```

Examples are meant to be:

- copy-paste friendly
- small
- focused on one concept

If you add a feature that changes how users should adopt the package, add or update an example.

Example index:

- [`examples/README.md`](../examples/README.md)

## Contributor Rules Of Thumb

### Keep The Surface Area Small

This package is valuable because it stays narrow and readable.

Prefer:

- small composable helpers
- explicit behavior
- clear public types

Avoid:

- broad abstractions that hide behavior
- provider-generalized layers unless they solve a real problem
- framework features that would turn the package into a full agent platform

### Treat Tests As Contract

Behavior around these areas is user-facing and easy to break:

- system prompt scoping
- snapshot persistence
- streaming event order
- tool result ordering
- sync runtime restrictions

If you change one of those behaviors, update the tests and docs together.

### Update Docs With Behavior Changes

When changing public behavior:

- update [`README.md`](../README.md)
- update the relevant file in [`docs/`](./README.md)
- update or add examples when needed

### Add Public APIs Carefully

If you add a new public export:

1. add it to `src/simple_agent_base/__init__.py`
2. test it
3. document it
4. consider whether it belongs in the narrow public API at all

## Packaging

Project metadata lives in [`pyproject.toml`](../pyproject.toml).

Build the package:

```bash
uv build
```

Publish the package:

```bash
uv publish
```

The wheel target is configured to package `src/simple_agent_base`.

## Debugging Tips

If something looks wrong:

- inspect `AgentRunResult.raw_responses`
- inspect `ChatSession.items`
- inspect generated tool schemas through `ToolRegistry.to_openai_tools()`
- check whether a convenience `system_prompt` became a prepended `developer` message
- confirm whether you are using `run()` or `stream()`, since they surface failures differently

If sync code behaves oddly:

- confirm you are not inside an existing event loop
- confirm you call `agent.close()` after sync usage

If tool behavior looks wrong:

- confirm every parameter has a type annotation
- confirm tool names are unique
- confirm the model returned valid JSON arguments
- confirm parallel tool calls are safe for your tools
