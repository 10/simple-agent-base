import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    async with Agent(config=AgentConfig(model="gpt-5.6-sol")) as agent:
        chat = agent.chat()

        first = await chat.run("My name is Anson.")
        second = await chat.run("What name did I tell you?")

        print("First reply:")
        print(first.output_text)
        print()
        print("Second reply:")
        print(second.output_text)
        print()
        print("Conversation history:")
        for message in chat.history:
            print(f"- {message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
