import asyncio

from simple_agent_base import Agent, AgentConfig, ChatMessage, ImagePart, TextPart


async def main() -> None:
    async with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
        chat = agent.chat()

        await chat.run(
            [
                ChatMessage(
                    role="user",
                    content=[
                        TextPart("Remember this image."),
                        ImagePart.from_file("photo.jpg"),
                    ],
                )
            ]
        )
        follow_up = await chat.run("What was in the image?")

        print("Follow-up reply:")
        print(follow_up.output_text)
        print()
        print("Conversation history:")
        for message in chat.history:
            print(f"- {message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
