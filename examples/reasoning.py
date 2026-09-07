import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    async with Agent(
        config=AgentConfig(
            model="gpt-6-astra",
            reasoning_effort="high",
        )
    ) as agent:
        async for event in agent.stream(
            "Think carefully, then explain in one short sentence why async I/O helps agent workloads."
        ):
            if event.type == "reasoning_delta" and event.delta is not None:
                print(f"[reasoning] {event.delta}")
            elif event.type == "text_delta" and event.delta is not None:
                print(event.delta, end="")
            elif event.type == "completed" and event.result is not None:
                print()
                print()
                print("Final text:")
                print(event.result.output_text)
                print()
                print("Reasoning summary:")
                print(event.result.reasoning_summary)


if __name__ == "__main__":
    asyncio.run(main())
