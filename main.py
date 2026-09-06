import asyncio
import json
import os
from typing import Any

import httpx2
from dotenv import load_dotenv
from mcp import Client, MCPError
from mcp.types import TextContent
from openai import AsyncOpenAI, OpenAIError

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 8
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


def tool_result_text(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)

    texts = [block.text for block in result.content if isinstance(block, TextContent) and block.text]
    if texts:
        return "\n".join(texts)

    return json.dumps({"is_error": bool(result.is_error)}, ensure_ascii=False)


async def run_tool_call(mcp_client: Client, tool_call: Any) -> str:
    name = tool_call.function.name
    raw_arguments = tool_call.function.arguments or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments for {name}: {exc}. Arguments were: {raw_arguments}"

    if not isinstance(arguments, dict):
        return f"Error: tool arguments must be a JSON object, got {type(arguments).__name__}."

    try:
        result = await mcp_client.call_tool(name, arguments)
    except (MCPError, httpx2.HTTPError, OSError, TimeoutError) as exc:
        return f"Error calling tool {name}: {exc}"

    text = tool_result_text(result)
    if result.is_error:
        return f"Error from tool {name}: {text}"
    return text


async def chat_turn(
    llm_client: AsyncOpenAI,
    mcp_client: Client,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
) -> str | None:
    """Run one user turn through the tool loop. Returns the assistant reply, or None on failure."""
    turn_start = len(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = await llm_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )
        except OpenAIError as exc:
            print(f"Error: {exc}")
            del messages[turn_start:]
            return None

        choice = response.choices[0].message
        tool_calls = choice.tool_calls

        if not tool_calls:
            assistant_message = choice.content or ""
            messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message

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
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            text_result = await run_tool_call(mcp_client, tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": text_result,
                }
            )

    print(f"Error: exceeded {MAX_TOOL_ROUNDS} tool rounds without a final answer.")
    del messages[turn_start:]
    return None


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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        print("Chat with OpenAI (type 'quit' or 'exit' to leave)\n")

        while True:
            try:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Bye!")
                break

            messages.append({"role": "user", "content": user_input})
            assistant_message = await chat_turn(llm_client, mcp_client, messages, openai_tools)
            if assistant_message is None:
                continue

            print(f"\nAssistant: {assistant_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
