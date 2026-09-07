"""Use a client-side MCP server with an Agent.

Run:
    uv run python examples/mcp_server.py

Requires a valid OPENAI_API_KEY. This example starts the repo's local demo MCP
server over stdio, exposes its tools as normal function tools to the model, and
runs the tool calls locally inside this Python process.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from simple_agent_base import Agent, AgentConfig, MCPApprovalRequest, MCPServer

FIXTURE_SERVER = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mcp_demo_server.py"


def build_demo_server(*, require_approval: bool) -> MCPServer:
    return MCPServer.stdio(
        name="demo",
        command=sys.executable,
        args=[str(FIXTURE_SERVER), "stdio"],
        require_approval=require_approval,
    )


async def basic_mcp_run() -> None:
    async with Agent(
        config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-6-astra")),
        mcp_servers=[build_demo_server(require_approval=False)],
    ) as agent:
        result = await agent.run(
            "Use the demo MCP echo tool with the message 'hello from MCP' "
            "and reply with one short sentence that includes the tool result."
        )

    print("=== Model output ===")
    print(result.output_text)
    print()
    print("=== MCP tool calls ===")
    for call in result.mcp_calls:
        print(f"- {call.server_name}.{call.name}({call.arguments}) -> {call.output}")


async def mcp_run_with_approvals() -> None:
    def approve(request: MCPApprovalRequest) -> bool:
        print(f"[approval] {request.server_name}.{request.name}({request.arguments})")
        return True

    async with Agent(
        config=AgentConfig(model=os.environ.get("OPENAI_MODEL", "gpt-6-astra")),
        mcp_servers=[build_demo_server(require_approval=True)],
        approval_handler=approve,
    ) as agent:
        result = await agent.run(
            "Use the demo MCP add tool with 2 and 3, then reply with one short sentence."
        )

    print("=== With-approval output ===")
    print(result.output_text)


async def main() -> None:
    await basic_mcp_run()
    print()
    await mcp_run_with_approvals()


if __name__ == "__main__":
    asyncio.run(main())
