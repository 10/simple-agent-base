from simple_agent_base import Agent, AgentConfig


def main() -> None:
    with Agent(config=AgentConfig(model="gpt-5.6-sol")) as agent:
        result = agent.run_sync("Say hello in one short sentence.")
        print("Final text:")
        print(result.output_text)


if __name__ == "__main__":
    main()
