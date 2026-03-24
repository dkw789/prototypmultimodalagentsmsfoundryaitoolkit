"""Build Agent using Microsoft Agent Framework in Python
# Run this python script
> pip install anthropic agent-framework==1.0.0rc3
> python <this-script-path>.py
"""

import asyncio
import os

import httpx
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool, ToolTypes
from agent_framework.openai import OpenAIChatClient
from agent_framework.anthropic import AnthropicClient
from anthropic import AsyncAnthropicFoundry
from openai import AsyncOpenAI

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
openaiClient = AsyncOpenAI(
    base_url = "https://models.github.ai/inference",
    api_key = os.environ["GITHUB_TOKEN"],
    default_query = {
        "api-version": "2024-08-01-preview",
    },
)

AGENT_NAME = "ai-agent"
AGENT_INSTRUCTIONS = "You are Cora, an intelligent and friendly AI assistant for Zava, a home improvement brand. You help customers with their DIY projects by understanding their needs and recommending the most suitable products from Zava's catalog.\nYour role is to:\n- Engage with the customer in natural conversation to understand their DIY goals.\n- Ask thoughtful questions to gather relevant project details.\n- Be brief in your responses.\n- Provide the best solution for the customer's problem and only recommend a relevant product within Zava's product catalog.\n- Search Zava's product database to identify 1 product that best match the customer's needs.\n- Clearly explain what each recommended Zava product is, why it's a good fit, and how it helps with their project.\n\nYour personality is:\n- Warm and welcoming, like a helpful store associate\n- Professional and knowledgeable, like a seasoned DIY expert\n- Curious and conversational-never assume, always clarify\n- Transparent and honest-if something isn't available, offer support anyway\n\nIf no matching products are found in Zava's catalog, say:\n\"Thanks for sharing those details! I've searched our catalog, but it looks like we don't currently have a product that fits your exact needs. If you'd like, I can suggest some alternatives or help you adjust your project requirements to see if something similar might work.\""

# User inputs for the conversation
USER_INPUTS = [
    "hi",
    "hey",
    "Here's a photo of my living room. Based on the lighting and layout, recommend a Zava eggshell paint",
]

def create_mcp_tools() -> list[ToolTypes]:
    return [
        MCPStdioTool(
            name="zava-customer-sales-stdio".replace("-", "_"),
            description="MCP server for zava-customer-sales-stdio",
            load_prompts=False,
            command="python",
            args=[
                "/workspace/src/python/mcp_server/customer_sales/customer_sales.py",
                "--stdio",
                "--RLS_USER_ID=00000000-0000-0000-0000-000000000000",
            ]
        ),
    ]

async def main() -> None:
    async with (
        OpenAIChatClient(
            async_client=openaiClient,
            model_id="openai/gpt-5-mini"
        ).as_agent(
            instructions=AGENT_INSTRUCTIONS,
            tools=[
                *create_mcp_tools(),
            ],
        ) as agent
    ):
        # Process user messages
        for user_input in USER_INPUTS:
            print(f"\n# User: '{user_input}'")
            printed_tool_calls = set()
            async for chunk in agent.run([user_input], stream=True):
                # log tool calls if any
                function_calls = [
                    c for c in chunk.contents 
                    if c.type == "function_call"
                ]
                for call in function_calls:
                    if call.call_id not in printed_tool_calls:
                        print(f"Tool calls: {call.name}")
                        printed_tool_calls.add(call.call_id)
                if chunk.text:
                    print(chunk.text, end="")
            print("")
        
        print("\n--- All tasks completed successfully ---")

    # Give additional time for all async cleanup to complete
    await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished.")
