import asyncio

from simple_agent_base import Agent, AgentConfig, tool


@tool
async def get_weather(city: str) -> str:
    """Return a fake weather summary for a city."""
    await asyncio.sleep(0.5)
    return f"{city}: 72F and sunny"


@tool
async def get_news(city: str) -> str:
    """Return a fake news headline for a city."""
    await asyncio.sleep(0.5)
    return f"{city}: local tech conference opens today"


async def main() -> None:
    async with Agent(
        config=AgentConfig(
            model="gpt-5.6-sol",
            parallel_tool_calls=True,
        ),
        tools=[get_weather, get_news],
    ) as agent:
        result = await agent.run(
            "Call both tools for San Francisco, then summarize the results in one short paragraph."
        )
        print("Final text:")
        print(result.output_text)
        print()
        print("Tool results:")
        for tool_result in result.tool_results:
            print(f"- {tool_result.name}: {tool_result.output}")


if __name__ == "__main__":
    asyncio.run(main())
