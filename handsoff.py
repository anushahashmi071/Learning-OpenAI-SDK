import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, handoff
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

billing_agent = Agent(name="Billing agent",
                      instructions="Handle billing questions.",
                      model=gemini_model)

refund_agent = Agent(name="Refund agent",
                     instructions="Handle refunds.",
                     model=gemini_model)

# Agent with injected context in instructions
triage_agent = Agent(
    name="Triage_Agent",
    instructions="You are a helpful assistant.",
    model=gemini_model,
    handoffs=[billing_agent, refund_agent]
)


query = "I need to check refund status. Can you help?"

result = Runner.run_sync(
    starting_agent=triage_agent,
    input=query
)

print(result.final_output)
