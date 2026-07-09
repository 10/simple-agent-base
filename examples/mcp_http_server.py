"""Use a streamable HTTP MCP server with an Agent.

Run:
    uv run python examples/mcp_http_server.py

Requires a valid OPENAI_API_KEY. This example starts the repo's local demo MCP
server over HTTP, connects with `MCPServer.http(...)`, and runs the chosen tool
calls locally through the agent's MCP bridge.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPApprovalRequest, MCPServer

FIXTURE_SERVER = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mcp_demo_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(host: str, port: int) -> None:
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)

    raise RuntimeError(f"HTTP MCP fixture did not start: {last_error}")


@contextmanager
def demo_http_server() -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE_SERVER), "http", str(port)],
        cwd=str(FIXTURE_SERVER.parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server("127.0.0.1", port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def basic_http_mcp_run() -> None:
    with demo_http_server() as server_url:
        async with Agent(
            config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-5")),
            mcp_servers=[MCPServer.http(name="demohttp", url=server_url)],
        ) as agent:
            result = await agent.run(
                "Use the demohttp MCP add tool with 2 and 3, "
                "then reply with one short sentence that includes the result."
            )

    print("=== Server URL ===")
    print(server_url)
    print()
    print("=== Model output ===")
    print(result.output_text)
    print()
    print("=== MCP tool calls ===")
    for call in result.mcp_calls:
        print(f"- {call.server_name}.{call.name}({call.arguments}) -> {call.output}")


async def http_mcp_run_with_approvals() -> None:
    def approve(request: MCPApprovalRequest) -> bool:
        print(f"[approval] {request.server_name}.{request.name}({request.arguments})")
        return True

    with demo_http_server() as server_url:
        async with Agent(
            config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-5")),
            mcp_servers=[MCPServer.http(name="demohttp", url=server_url, require_approval=True)],
            approval_handler=approve,
        ) as agent:
            result = await agent.run(
                "Use the demohttp MCP echo tool with the message 'hello over HTTP', "
                "then reply with one short sentence that includes the tool result."
            )

    print("=== With-approval output ===")
    print(result.output_text)


async def main() -> None:
    await basic_http_mcp_run()
    print()
    await http_mcp_run_with_approvals()


if __name__ == "__main__":
    asyncio.run(main())
