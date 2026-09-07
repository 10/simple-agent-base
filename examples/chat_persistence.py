import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    async with Agent(config=AgentConfig(model="gpt-6-astra")) as agent:
        chat = agent.chat(system_prompt="You are concise.")

        await chat.run("My name is Anson.")
        await chat.run("My favorite color is teal.")

        payload = chat.export()

        print("Current chat history:")
        for message in chat.history:
            print(f"- {message.role}: {message.content}")

        print()
        print("Snapshot keys:")
        for key in payload:
            print(f"- {key}")

        print()
        restored = agent.chat_from_snapshot(payload)
        result = await restored.run("What is my favorite color?")

        print("Restored follow-up reply:")
        print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
