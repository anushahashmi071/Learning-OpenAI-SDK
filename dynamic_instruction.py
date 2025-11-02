from agents import Agent, Runner, OpenAIChatCompletionsModel, RunContextWrapper, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
set_tracing_disabled(True)

# Gemini API key use karein
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini ka base URL
)


def dynamic_instruction(ctx: RunContextWrapper, agent: Agent) -> str:
    # Date aur day function ke andar calculate karein
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    day = now.strftime("%A")

    return f"""You are a helpful AI assistant.
    Today is {day}, {date}.
    When user asks about today's date or day, you must respond EXACTLY in this format:

    "Today is {day}, {date}."

    Do not rephrase or change the format."""


agent: Agent = Agent(
    name="Smart Assistance",
    instructions=dynamic_instruction,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash-exp",  # Gemini model
        openai_client=client
    )
)

result = Runner.run_sync(
    agent,
    "What is the date today?",
)

print(result.final_output)
