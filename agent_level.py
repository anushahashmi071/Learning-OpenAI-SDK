from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env file

set_tracing_disabled(True) 

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

agent: Agent = Agent(
    name="Assistance",
    instructions="You are a helpful assistant.",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    )
)


result = Runner.run_sync(
    agent,
    "2 plus 3 equals what?",
)

print(result.final_output)
