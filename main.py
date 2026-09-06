import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent
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


async def run_tool_call(mcp_client, tool_call):
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}

    result = await mcp_client.call_tool(name, arguments)

    return result


def tool_result_text(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)

    texts = [block.text for block in result.content if isinstance(block, TextContent) and block.text]
    if texts:
        return "\n".join(texts)

    return json.dumps({"is_error": bool(result.is_error)}, ensure_ascii=False)


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

            while True:
                # This is the tool calling loop

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

                choice = response.choices[0].message
                assistant_message = choice.content or ""
                tool_calls = choice.tool_calls

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": choice.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in choice.tool_calls
                            ],
                        }
                    )
                    tool_results = []
                    for tool_call in tool_calls:
                        result = await run_tool_call(mcp_client, tool_call)
                        text_result = tool_result_text(result)

                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": text_result,
                            }
                        )
                    messages.extend(tool_results)
                    # We call the LLM again with the tool results
                    continue
                else:
                    break

            messages.append({"role": "assistant", "content": assistant_message})
            print(f"\nAssistant: {assistant_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
