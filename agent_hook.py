from agents import Agent, Runner, OpenAIChatCompletionsModel, RunContextWrapper, AgentHooks, function_tool
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
class CustomAgentHook(AgentHooks):
    async def on_start(self, ctx: RunContextWrapper, agent: Agent) -> None:
        print(f"\n[Agent]: {agent.name} \n")

    async def on_end(self, ctx: RunContextWrapper, agent: Agent, output: Any) -> None:
        print(f"\n[Agent]: {agent.name} \n[Output]: {output}\n")

    async def on_handoff(self, ctx: RunContextWrapper, agent: Agent, output: Any) -> None:
        print(f"\n[Agent]: {agent.name} \n[Output]: {output}\n")

    async def on_tool_start(self, ctx: RunContextWrapper, agent: Agent, tool) -> None:
        print(f"\n[Agent]: {agent.name} \n[Tools]: {tool}\n")

    async def on_tool_end(self, ctx: RunContextWrapper, agent: Agent, tool, result) -> None:
        print(
            f"\n[Agent]: {agent.name} \n[Tools]: {tool} \n[Result]: {result} ")

    async def on_llm_start(self, ctx: RunContextWrapper, agent: Agent, system_prompt, input_items) -> None:
        print(
            f"\n[Agent]: {agent.name} \n[System Prompt]: {system_prompt} \n[Input Items]: {input_items} ")

    async def on_llm_end(self, ctx: RunContextWrapper, agent: Agent, response) -> None:
        print(f"\n[Agent]: {agent.name} \n[Response]: {response}")

# Tools
addition_tool = function_tool(addition)

# Run
agent = Agent(
    name="Maths Assistance",
    instructions="You are a helpful assistant.",
    tools=[addition_tool],
    hooks=CustomAgentHook()
)

result = Runner.run_sync(
    starting_agent=agent,
    input="What is 2 + 3",
    run_config=config_agent,
)

print("\n" + "-"*60)
print("FINAL OUTPUT:")
print(result.final_output)
print("-"*60 + "\n")
