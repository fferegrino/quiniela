import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from mcp import Client
from openai import AsyncOpenAI, OpenAIError

load_dotenv()

MODEL = "gpt-4o-mini"
BASE_SYSTEM_PROMPT = (
    """Eres un asistente experto en fútbol mexicano especializado en ayudar a llenar quinielas de la Liga MX."""
)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def mcp_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return openai_tools


async def main() -> None:

    async with Client(os.environ["LIGA_MX_MCP_URL"]) as mcp_client:
        list_tools = await mcp_client.list_tools()
        openai_tools = mcp_tools_to_openai(list_tools.tools)
        if mcp_client.instructions:
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mcp_client.instructions}"
        else:
            system_prompt = BASE_SYSTEM_PROMPT

        print(f">>> System prompt: {system_prompt}")

        llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        print("Chat with OpenAI (type 'quit' or 'exit' to leave)\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Bye!")
                break

            messages.append({"role": "user", "content": user_input})

            try:
                response = await llm_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
            except OpenAIError as exc:
                print(f"Error: {exc}")
                messages.pop()
                continue

            assistant_message = response.choices[0].message.content or ""
            tool_calls = response.choices[0].message.tool_calls

            messages.append({"role": "assistant", "content": assistant_message})
            print(f"\nAssistant: {assistant_message}\n")
            print(f">>> Tool calls: {tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
