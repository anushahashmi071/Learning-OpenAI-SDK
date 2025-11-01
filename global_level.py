from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, function_tool, enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from tool import addition, subtraction, multiplication, division
# Adjust AI agent's brain behavior to get exactly the response you want.
from agents import ModelSettings

load_dotenv()  # load environment variables from .env file

set_tracing_disabled(True)  # removes tracing logs

enable_verbose_stdout_logging()  # enable verbose logging to stdout for debugging

gemini_api_key = os.getenv("GEMINI_API_KEY")

# register the addition function as a tool
addition_tool = function_tool(addition)
subtraction_tool = function_tool(subtraction)
multiplication_tool = function_tool(multiplication)
division_tool = function_tool(division)

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(client)  # set the default client globally

agent: Agent = Agent(
    name="Assistance",
    instructions="You are a helpful assistant. Use the tools to DMAS.",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
    tools=[addition_tool, subtraction_tool,
           multiplication_tool, division_tool],
    model_settings=ModelSettings(
        temperature=0.2,
        top_p=0.3,  # Use only top 30% of vocabulary
    )
)

result = Runner.run_sync(
    agent,
    "2 plus 3 equals what?",
)

print(result.final_output)
