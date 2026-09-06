"""UI-agnostic Liga MX quiniela chat agent (OpenAI Responses + MCP + SerpApi)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, TypeVar

import httpx
import httpx2
import mlflow
from dotenv import load_dotenv
from mcp import Client, MCPError
from mcp.types import CallToolResult, TextContent, Tool
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from openai.types.responses.response_input_param import ResponseInputParam
from typing_extensions import Self

from serpapi_news import (
    SerpApiError,
    enrich_articles_with_bodies,
    search_teams_news,
)

mlflow.openai.autolog()
load_dotenv()

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

PROFILES: dict[str, dict[str, str]] = {
    "mini": {"model": "gpt-5-mini", "reasoning_effort": "minimal"},
    "reasoning": {"model": "gpt-5.6", "reasoning_effort": "high"},
}
DEFAULT_PROFILE = os.getenv("QUINIELA_PROFILE", "reasoning")
MAX_TOOL_ROUNDS = 10

BASE_SYSTEM_PROMPT = """Eres un asistente experto en fútbol mexicano especializado en ayudar a llenar quinielas de la Liga MX, conociendo la información de los equipos, sus jugadores, sus partidos, sus resultados, sus estadísticas, sus noticias, sus rumores, sus alineaciones, sus lesiones, etc.

Usa información de noticias recientes de los equipos para llenar quinielas, como lesiones, rumores, alineaciones, etc.

Puedes consultar noticias recientes de equipos con la herramienta search_team_news cuando necesites contexto de prensa, lesiones, rumores o forma reciente."""

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


def resolve_profile(name: str) -> tuple[str, str, ReasoningEffort]:
    """Return (profile_name, model, reasoning_effort) for a known profile key."""
    key = name.strip().lower()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}. Choose one of: {known}")
    profile = PROFILES[key]
    return key, profile["model"], profile["reasoning_effort"]  # type: ignore[return-value]


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
) -> tuple[str | None, str | None, str | None]:
    """Run one user turn via the Responses API, including tool-calling rounds.

    Uses ``previous_response_id`` so reasoning items stay available across
    function calls. Returns ``(assistant_text, latest_response_id, error)``.
    On API failure or too many tool rounds, ``assistant_text`` is ``None`` and
    ``error`` explains why; ``latest_response_id`` stays at the prior value.
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
            return None, previous_response_id, str(exc)

        response_id = response.id
        function_calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]

        if not function_calls:
            return response.output_text or "", response_id, None

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

    return None, previous_response_id, f"Exceeded {MAX_TOOL_ROUNDS} tool rounds without a final answer."


@dataclass
class TurnResult:
    """Outcome of one user message."""

    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


@dataclass
class QuinielaAgent:
    """Stateful quiniela chat session shared by CLI and UI frontends.

    Conversation continuity lives in OpenAI via ``previous_response_id``.
    Display history is the caller's responsibility.
    """

    profile: str = field(default_factory=lambda: DEFAULT_PROFILE)
    previous_response_id: str | None = None
    _llm_client: AsyncOpenAI | None = field(default=None, init=False, repr=False)
    _mcp_client: Client | None = field(default=None, init=False, repr=False)
    _openai_tools: list[FunctionToolParam] = field(default_factory=list, init=False, repr=False)
    _system_prompt: str = field(default="", init=False, repr=False)
    _profile_name: str = field(default="", init=False, repr=False)
    _model: str = field(default="", init=False, repr=False)
    _reasoning_effort: ReasoningEffort = field(default="medium", init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        return self._reasoning_effort

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def mcp_tool_names(self) -> list[str]:
        return [tool["name"] for tool in self._openai_tools if tool["name"] != SEARCH_TEAM_NEWS_TOOL_NAME]

    async def start(self) -> None:
        """Connect to MCP and prepare OpenAI tools. Idempotent."""
        if self._started:
            return

        api_key = os.environ["OPENAI_API_KEY"]
        mcp_url = os.environ["LIGA_MX_MCP_URL"]

        self._profile_name, self._model, self._reasoning_effort = resolve_profile(self.profile)
        self._llm_client = AsyncOpenAI(api_key=api_key)
        self._mcp_client = Client(mcp_url)
        await self._mcp_client.__aenter__()

        list_tools = await self._mcp_client.list_tools()
        self._openai_tools = mcp_tools_to_openai(list_tools.tools) + [SEARCH_TEAM_NEWS_TOOL]
        if self._mcp_client.instructions:
            self._system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{self._mcp_client.instructions}"
        else:
            self._system_prompt = BASE_SYSTEM_PROMPT
        self._started = True

    async def close(self) -> None:
        """Release the MCP connection."""
        if self._mcp_client is not None:
            await self._mcp_client.__aexit__(None, None, None)
            self._mcp_client = None
        self._llm_client = None
        self._started = False

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def set_profile(self, name: str) -> None:
        """Switch model profile and reset OpenAI conversation chaining."""
        self._profile_name, self._model, self._reasoning_effort = resolve_profile(name)
        self.profile = self._profile_name
        self.previous_response_id = None

    def reset_conversation(self) -> None:
        """Drop OpenAI conversation chaining without changing the profile."""
        self.previous_response_id = None

    async def ask(self, user_input: str) -> TurnResult:
        """Send one user message and return the assistant reply (or an error)."""
        if not self._started or self._llm_client is None or self._mcp_client is None:
            raise RuntimeError("QuinielaAgent is not started. Call start() or use async with.")

        text = user_input.strip()
        if not text:
            return TurnResult(text=None, error="Empty message.")

        reply, self.previous_response_id, error = await chat_turn(
            self._llm_client,
            self._mcp_client,
            self._system_prompt,
            self._openai_tools,
            text,
            self.previous_response_id,
            self._model,
            self._reasoning_effort,
        )
        return TurnResult(text=reply, error=error)


def ask_sync(agent: QuinielaAgent, user_input: str) -> TurnResult:
    """Run ``agent.ask`` from a synchronous host that owns no event loop."""
    return asyncio.run(agent.ask(user_input))


T = TypeVar("T")


class AsyncRunner:
    """Dedicated background event loop for sync hosts like Streamlit.

    Keeps long-lived async resources (MCP client) usable across UI reruns.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
