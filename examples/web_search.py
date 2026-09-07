import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="gpt-6-astra"),
        hosted_tools=[{"type": "web_search"}],
    ) as agent:
        result = await agent.run("What's the latest news about Python 3.13?")
        print("Final text:")
        print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
