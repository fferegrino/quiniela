import asyncio
import json
import os
from typing import Any, Literal

import httpx
import httpx2
import mlflow
from dotenv import load_dotenv
from mcp import Client, MCPError
from mcp.types import CallToolResult, TextContent, Tool
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from openai.types.responses.response_input_param import ResponseInputParam

from serpapi_news import (
    SerpApiError,
    enrich_articles_with_bodies,
    search_teams_news,
)

mlflow.openai.autolog()

load_dotenv()

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

# Both profiles use the Responses API (required for tools + reasoning on gpt-5.6).
# Switch with QUINIELA_PROFILE=mini|reasoning, or type /mini or /reasoning in chat.
PROFILES: dict[str, dict[str, str]] = {
    "mini": {"model": "gpt-5-mini", "reasoning_effort": "minimal"},
    "reasoning": {"model": "gpt-5.6", "reasoning_effort": "high"},
}
DEFAULT_PROFILE = os.getenv("QUINIELA_PROFILE", "reasoning")
MAX_TOOL_ROUNDS = 10


def resolve_profile(name: str) -> tuple[str, str, ReasoningEffort]:
    """Return (profile_name, model, reasoning_effort) for a known profile key."""
    key = name.strip().lower()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Choose one of: {known}")
    profile = PROFILES[key]
    return key, profile["model"], profile["reasoning_effort"]  # type: ignore[return-value]


BASE_SYSTEM_PROMPT = """Eres un asistente experto en fútbol mexicano especializado en ayudar a llenar quinielas de la Liga MX, conociendo la información de los equipos, sus jugadores, sus partidos, sus resultados, sus estadísticas, sus noticias, sus rumores, sus alineaciones, sus lesiones, etc.

Usa información de noticias recientes de los equipos para llenar quinielas, como lesiones, rumores, alineaciones, etc.

Puedes consultar noticias recientes de equipos con la herramienta search_team_news cuando necesites contexto de prensa, lesiones, rumores o forma reciente."""
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SEARCH_TEAM_NEWS_TOOL_NAME = "search_team_news"
SEARCH_TEAM_NEWS_TOOL: FunctionToolParam = {
    "type": "function",
    "name": SEARCH_TEAM_NEWS_TOOL_NAME,
    "description": (
        "Busca noticias recientes de equipos de la Liga MX / fútbol mexicano en Google News (SerpApi). "
        "Úsala para lesiones, forma reciente, rumores, alineaciones o cobertura de prensa que ayude a armar quinielas. "
        "Pasa uno o más nombres de clubes (por ejemplo: América, Cruz Azul, Guadalajara)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "teams": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Nombres de clubes a buscar, por ejemplo: ['América', 'Cruz Azul'].",
            },
            "when": {
                "type": "string",
                "enum": ["1d", "7d", "30d", "1y"],
                "description": "Ventana de antigüedad de las noticias. Por defecto: 7d.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Máximo de artículos por equipo (1-10). Por defecto: 5.",
            },
            "read_bodies": {
                "type": "boolean",
                "description": (
                    "Si es true, también descarga y extrae el texto de los primeros enlaces. "
                    "Es más lento; úsalo solo cuando los snippets no basten."
                ),
            },
        },
        "required": ["teams"],
        "additionalProperties": False,
    },
    "strict": False,
}


def mcp_tools_to_openai(tools: list[Tool]) -> list[FunctionToolParam]:
    """Convert MCP tool definitions into OpenAI Responses API function tool schemas."""
    openai_tools: list[FunctionToolParam] = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
                "strict": False,
            }
        )
    return openai_tools


def tool_result_text(result: CallToolResult) -> str:
    """Serialize an MCP tool result into text for a function_call_output item."""
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)

    texts = [block.text for block in result.content if isinstance(block, TextContent) and block.text]
    if texts:
        return "\n".join(texts)

    return json.dumps({"is_error": bool(result.is_error)}, ensure_ascii=False)


def _run_search_team_news(arguments: dict[str, Any]) -> str:
    """Execute the local SerpApi news tool and return JSON text for the model."""
    teams_raw = arguments.get("teams")
    if isinstance(teams_raw, str):
        teams = [teams_raw.strip()] if teams_raw.strip() else []
    elif isinstance(teams_raw, list):
        teams = [str(team).strip() for team in teams_raw if str(team).strip()]
    else:
        return "Error: 'teams' debe ser un arreglo no vacío con nombres de equipos."

    if not teams:
        return "Error: 'teams' debe ser un arreglo no vacío con nombres de equipos."

    when = arguments.get("when") or "7d"
    if not isinstance(when, str):
        return "Error: 'when' debe ser una cadena como '1d', '7d', '30d' o '1y'."

    limit_raw = arguments.get("limit", 5)
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return "Error: 'limit' debe ser un entero entre 1 y 10."
    limit = max(1, min(limit, 10))

    read_bodies = bool(arguments.get("read_bodies", False))

    articles = search_teams_news(teams, when=when, limit_per_team=limit)
    if read_bodies and articles:
        articles = enrich_articles_with_bodies(articles, limit=min(3, len(articles)))

    payload = {
        "teams": teams,
        "when": when,
        "count": len(articles),
        "articles": [
            {
                "title": article.title,
                "source": article.source,
                "date": article.date,
                "link": article.link,
                "snippet": article.snippet,
                "team": article.team,
                **({"body": article.body} if article.body else {}),
            }
            for article in articles
        ],
    }
    if not articles:
        payload["message"] = "No se encontraron noticias para los equipos solicitados."
    return json.dumps(payload, ensure_ascii=False)


async def run_tool_call(
    mcp_client: Client,
    tool_call: ResponseFunctionToolCall,
) -> str:
    """Execute one Responses API function call (local SerpApi news or MCP).

    Returns a string suitable for a ``function_call_output`` item. Invalid
    arguments and transport/tool failures are returned as error text so the
    model can recover instead of aborting the turn.
    """
    name = tool_call.name
    raw_arguments = tool_call.arguments or "{}"
    try:
        arguments: Any = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments for {name}: {exc}. Arguments were: {raw_arguments}"

    if not isinstance(arguments, dict):
        return f"Error: tool arguments must be a JSON object, got {type(arguments).__name__}."

    if name == SEARCH_TEAM_NEWS_TOOL_NAME:
        try:
            return await asyncio.to_thread(_run_search_team_news, arguments)
        except (SerpApiError, KeyError, OSError, httpx.HTTPError, TimeoutError) as exc:
            return f"Error calling tool {name}: {exc}"

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
    instructions: str,
    openai_tools: list[FunctionToolParam],
    user_input: str,
    previous_response_id: str | None,
    model: str,
    reasoning_effort: ReasoningEffort,
) -> tuple[str | None, str | None]:
    """Run one user turn via the Responses API, including tool-calling rounds.

    Uses ``previous_response_id`` so reasoning items stay available across
    function calls. Returns ``(assistant_text, latest_response_id)``. On API
    failure or too many tool rounds, returns ``(None, previous_response_id)``.
    """
    input_items: ResponseInputParam = [{"role": "user", "content": user_input}]
    response_id = previous_response_id

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = await llm_client.responses.create(
                model=model,
                instructions=instructions,
                input=input_items,
                tools=openai_tools,
                tool_choice="auto",
                reasoning={"effort": reasoning_effort},
                previous_response_id=response_id,
            )
        except OpenAIError as exc:
            print(f"Error: {exc}")
            return None, previous_response_id

        response_id = response.id
        function_calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]

        if not function_calls:
            return response.output_text or "", response_id

        input_items = []
        for tool_call in function_calls:
            text_result = await run_tool_call(mcp_client, tool_call)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": text_result,
                }
            )

    print(f"Error: exceeded {MAX_TOOL_ROUNDS} tool rounds without a final answer.")
    return None, previous_response_id


async def main() -> None:
    """Connect to the Liga MX MCP server and run the interactive chat loop."""
    async with Client(os.environ["LIGA_MX_MCP_URL"]) as mcp_client:
        list_tools = await mcp_client.list_tools()
        openai_tools = mcp_tools_to_openai(list_tools.tools) + [SEARCH_TEAM_NEWS_TOOL]
        if mcp_client.instructions:
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{mcp_client.instructions}"
        else:
            system_prompt = BASE_SYSTEM_PROMPT

        profile_name, model, reasoning_effort = resolve_profile(DEFAULT_PROFILE)

        print(f">>> System prompt: {system_prompt}")
        print(f">>> Profile: {profile_name} → {model} (reasoning_effort={reasoning_effort})")
        print(f">>> Local tools: {SEARCH_TEAM_NEWS_TOOL_NAME}")
        print(f">>> MCP tools: {', '.join(tool.name for tool in list_tools.tools) or '(none)'}")

        llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        previous_response_id: str | None = None

        print("Chat with OpenAI (type 'quit'/'exit', or '/mini' / '/reasoning' to switch)\n")

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

            if user_input.lower() in ("/mini", "/reasoning"):
                profile_name, model, reasoning_effort = resolve_profile(user_input[1:])
                previous_response_id = None  # model/effort change breaks response chaining
                print(f"Switched to {profile_name} → {model} (reasoning_effort={reasoning_effort})")
                continue

            assistant_message, previous_response_id = await chat_turn(
                llm_client,
                mcp_client,
                system_prompt,
                openai_tools,
                user_input,
                previous_response_id,
                model,
                reasoning_effort,
            )
            if assistant_message is None:
                continue

            print(f"\nAssistant: {assistant_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
