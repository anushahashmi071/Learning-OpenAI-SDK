from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel # use for output type

load_dotenv()
set_tracing_disabled(True)

# Gemini API key use karein
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini ka base URL
)

class Calendar(BaseModel):
    date: str
    day: str

agent: Agent = Agent(
    name="Calendar",
    instructions="Tell the Date and Day of the week.",
    output_type=Calendar,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",  # Gemini model
        openai_client=client
    )
)

result = Runner.run_sync(
    agent,
    "What is the date today?",
)

print(result.final_output)
