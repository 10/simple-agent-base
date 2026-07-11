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
    ) as agent:
        result = await agent.run("Call ping with hello and tell me the result.")
        print("Final text:")
        print(result.output_text)
        print()
        print("Tool results:")
        for tool_result in result.tool_results:
            print(f"- {tool_result.name}: {tool_result.output}")


if __name__ == "__main__":
    asyncio.run(main())
