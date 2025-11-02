import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# Base agent
base_agent = Agent(
    name="BaseAssistant",
    instructions="You are a helpful assistant.",
    model=gemini_model,
    model_settings=ModelSettings(temperature=0.7)
)

# Clone with different instructions
creative_agent = base_agent.clone(
    name="CreativeAssistant",
    instructions="You are a creative writing assistant. Always respond with vivid, imaginative language.",
    model=gemini_model,
    model_settings=ModelSettings(temperature=0.9)
)


query = "Hello, how are you?"

result_base = Runner.run_sync(base_agent, query)
result_creative = Runner.run_sync(creative_agent, query)

print("Base Agent:", result_base.final_output)
print("Creative Agent:", result_creative.final_output)
