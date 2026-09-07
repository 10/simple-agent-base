import asyncio

from simple_agent_base import Agent, AgentConfig, ChatMessage, ImagePart, TextPart


async def main() -> None:
    async with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
        result = await agent.run(
            [
                ChatMessage(
                    role="user",
                    content=[
                        TextPart("Describe this image."),
                        ImagePart.from_file("cat.png"),
                    ],
                )
            ]
        )

        print("Final text:")
        print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
