from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()  # load environment variables from .env file

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

async def run_agent():
    agent: Agent = Agent(
        name="Assistance",
        instructions="You are a helpful assistant.",
    )

    result = await Runner.run(
        agent,
        "Who is the founder of Pakistan?",
        run_config=config_agent,
    )

    print(result.final_output)

asyncio.run(run_agent())