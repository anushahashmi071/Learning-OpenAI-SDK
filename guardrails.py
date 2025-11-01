from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
    input_guardrail,
)
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Define what our guardrail should output


class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str


# Create a simple, fast agent to do the checking
guardrail_agent = Agent(
    name="Homework Police",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
    model=gemini_model,
)

# Create our guardrail function


@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # Run our checking agent
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    # Return the result with tripwire status
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        # Trigger if homework detected
        tripwire_triggered=result.final_output.is_math_homework,
    )

# Main agent with guardrail attached
customer_support_agent = Agent(
    name="Customer Support Specialist",
    instructions="You are a helpful customer support agent for our software company.",
    input_guardrails=[math_guardrail],  # Attach our guardrail
    model=gemini_model,
)

# Testing the guardrail

# Testing the guardrail
async def test_homework_detection():
    print("Test 1: Math homework (should be blocked)")
    try:
        result = await Runner.run(customer_support_agent, "Can you solve 2x + 3 = 11 for x?")
        print("❌ Guardrail failed - homework request got through!")
        print(f"Response: {result.final_output}")
    except InputGuardrailTripwireTriggered as e:
        print("✅ Success! Homework request was blocked.")
    
    print("\nTest 2: Normal customer support question (should pass)")
    try:
        result = await Runner.run(customer_support_agent, "How do I reset my password?")
        print("✅ Success! Normal question passed through.")
        print(f"Response: {result.final_output}")
    except InputGuardrailTripwireTriggered:
        print("❌ Guardrail failed - normal question was blocked!")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_homework_detection())


# # Testing the guardrail
# async def test_homework_detection():
#     try:
#         # This should trigger the guardrail
#         await Runner.run(customer_support_agent, "Can you solve 2x + 3 = 11 for x?")
#         print("Guardrail failed - homework request got through!")

#     except InputGuardrailTripwireTriggered:
#         print("Success! Homework request was blocked.")
#         # Handle appropriately - maybe send a polite rejection message

# # Run the test
# if __name__ == "__main__":
#     asyncio.run(test_homework_detection())
