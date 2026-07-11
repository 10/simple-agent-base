import asyncio

from pydantic import BaseModel

from simple_agent_base import Agent, AgentConfig


class Summary(BaseModel):
    title: str
    bullets: list[str]


async def main() -> None:
    async with Agent(
        config=AgentConfig(
            model="gpt-5.6-sol",
            reasoning_effort="high",
        )
    ) as agent:
        try:
            async for event in agent.stream(
                "Summarize why async Python is useful for I/O-heavy agent systems.",
                response_model=Summary,
            ):
                if event.type == "text_delta" and event.delta is not None:
                    print(event.delta, end="")
                elif event.type == "reasoning_delta" and event.delta is not None:
                    print()
                    print(f"[reasoning] {event.delta}")
                elif event.type == "tool_arguments_delta" and event.delta is not None:
                    print(event.delta, end="")
                elif event.type == "tool_call_started" and event.tool_call is not None:
                    print()
                    print(f"[tool start] {event.tool_call.name}")
                elif event.type == "tool_call_completed" and event.tool_result is not None:
                    print()
                    print(f"[tool done] {event.tool_result.name}: {event.tool_result.output}")
                elif event.type == "completed" and event.result is not None:
                    print()
                    print()
                    print("Reasoning summary:")
                    print(event.result.reasoning_summary)
                    print()
                    print("Structured result:")
                    print(event.result.output_data)
        except Exception as exc:
            print()
            print()
            print(f"Stream failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
