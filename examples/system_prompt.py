import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    async with Agent(
        config=AgentConfig(model="gpt-5.6-sol"),
        system_prompt="You are concise and helpful.",
    ) as agent:
        default_result = await agent.run("Say hello in five words or fewer.")
        override_result = await agent.run(
            "Explain Python decorators in one short paragraph.",
            system_prompt="You are a patient Python tutor teaching a beginner.",
        )

        chat = agent.chat(system_prompt="You are a terse interview coach.")
        first_chat = await chat.run("Help me prepare for a backend interview.")
        second_chat = await chat.run("Ask me the next question.")

        print("Agent default prompt result:")
        print(default_result.output_text)
        print()
        print("Per-run override result:")
        print(override_result.output_text)
        print()
        print("Chat session result 1:")
        print(first_chat.output_text)
        print()
        print("Chat session result 2:")
        print(second_chat.output_text)


if __name__ == "__main__":
    asyncio.run(main())
