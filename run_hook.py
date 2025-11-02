from agents import Agent, Runner, OpenAIChatCompletionsModel, RunContextWrapper, RunHooks, function_tool
from agents.run import RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any
import os
from tool import addition

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config_agent = RunConfig(
    model=model,
    model_provider="client",
    tracing_disabled=True
)

# Hooks
@dataclass
class CustomRunHook(RunHooks):
    async def on_llm_start(
        self,
        ctx: RunContextWrapper,
        agent: Agent,
        system_prompt: str,
        input_items: list
    ) -> None:
        print(f"\n[LLM START] Agent: {agent.name}")
        print(f"System: {system_prompt}")
        print(f"Input Items: {input_items}")

    async def on_agent_start(self, ctx: RunContextWrapper, agent: Agent) -> None:
        print(f'[AGENT START] Agent: {agent.name}')

    async def on_agent_end(self, ctx: RunContextWrapper, agent: Agent, output: Any) -> None:
        print(f'[AGENT END] Agent: {agent.name}')
        print(f'Output: {output}')

    async def on_handoff(self, ctx: RunContextWrapper, from_agent, to_agent) -> None:
        print(f'[HANDOFF] From: {from_agent.name} → To: {to_agent.name}')

    async def on_tool_start(self, ctx: RunContextWrapper, agent: Agent, tool) -> None:
        print(f'[TOOL START] Agent: {agent.name}, Tool: {tool.name}')

    async def on_tool_end(self, ctx: RunContextWrapper, agent: Agent, tool, result: Any) -> None:
        print(f'[TOOL END] Agent: {agent.name}, Tool: {tool.name} - {result}')

# Tools
addition_tool = function_tool(addition)

# Run
agent = Agent(
    name="Assistance",
    instructions="You are a helpful assistant.",
    tools=[addition_tool]
)

result = Runner.run_sync(
    starting_agent=agent,
    input="What is 2 + 3",
    run_config=config_agent,
    hooks=CustomRunHook()
)

print("\n" + "-"*60)
print("FINAL OUTPUT:")
print(result.final_output)
print("-"*60 + "\n")