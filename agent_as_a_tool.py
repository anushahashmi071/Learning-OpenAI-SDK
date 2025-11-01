import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=client
)

# Create full agents first
spanish_agent = Agent(
    name="Spanish agent",
    instructions="Translate the user's message to Spanish.",
    model=gemini_model
)

french_agent = Agent(
    name="French agent",
    instructions="Translate the user's message to French.",
    model=gemini_model
)

# Convert agents to tools
spanish_tool = Agent.as_tool(
    spanish_agent,
    tool_name="translate_to_spanish",
    tool_description="Translate text to Spanish"
)

french_tool = Agent.as_tool(
    french_agent,
    tool_name="translate_to_french",
    tool_description="Translate text to French"
)

# Triage agent with agent tools
triage_agent = Agent(
    name="Triage_Agent",
    instructions="You are a helpful assistant. Use the appropriate translation tool when the user asks for translation.",
    model=gemini_model,
    tools=[spanish_tool, french_tool]
)

query = "'How are you?' Translate into French."

result = Runner.run_sync(
    starting_agent=triage_agent,
    input=query
)

print(result.final_output)
