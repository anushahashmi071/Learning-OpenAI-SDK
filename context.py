import os
from openai import AsyncOpenAI
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_client, set_default_openai_api, RunContextWrapper
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Global Config
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
set_default_openai_client(client)

# Context
@dataclass
class UserInfo:
    name: str
    uid: int

# Agent with a function that uses context AND agent
def get_instructions(wrapper: RunContextWrapper[UserInfo], agent: Agent) -> str:
    return f"""
    You are a helpful assistant. Use the following user information to answer questions:
    - Name: {wrapper.context.name}
    - User ID: {wrapper.context.uid}
    - Age: 47 years old (hardcoded for demo)
    
    When asked about the user's name, always say: "User name is {wrapper.context.name}."
    When asked about the user's age, always say: "User {wrapper.context.name} is 47 years old."
    When asked about the user's ID, always say: "User {wrapper.context.name} ID is {wrapper.context.uid}."
    """

# Agent with injected context in instructions
agent = Agent[UserInfo](
    name="Assistant",
    instructions=get_instructions,  # Pass the function with 2 parameters
    model="gemini-2.0-flash-exp",
)

async def main():
    context = UserInfo(name="John", uid=123)
    query = "What is the uid of the user?"
    
    # Use Runner.run() as a class method, passing the agent directly
    result = await Runner.run(
        starting_agent=agent,
        input=query,
        context=context
    )
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())