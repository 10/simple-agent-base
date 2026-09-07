import asyncio

from simple_agent_base import Agent, AgentConfig, ChatMessage, FilePart, TextPart


async def main() -> None:
    async with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
        result = await agent.run(
            [
                ChatMessage(
                    role="user",
                    content=[
                        TextPart("Summarize this PDF in one short paragraph."),
                        FilePart.from_file("report.pdf"),
                    ],
                )
            ]
        )
        print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
