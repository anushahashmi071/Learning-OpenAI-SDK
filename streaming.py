from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()  # load environment variables from .env file

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

async def stream_agent():
  result = Runner.run_streamed(agent, "Who is the first governor of Pakistan?")
  async for event in result.stream_events():
    print(f"\nEvent: {event.type}\n")

    if event is not None:
      print(f'\nData: {event}\n')
    if event.type == "raw_response_event" and isinstance(event.data , ResponseTextDeltaEvent):
      print(f'\nDleta: {event.data}\n', flush=True)

asyncio.run(stream_agent())