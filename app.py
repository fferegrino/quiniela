"""Streamlit UI for the quiniela chat agent.

Chat logic lives in ``chat_agent``; this file only handles presentation and session state.
"""

from __future__ import annotations

import queue
import time
from typing import Any

import streamlit as st

from chat_agent import (
    DEFAULT_PROFILE,
    PROFILES,
    SEARCH_TEAM_NEWS_TOOL_NAME,
    AsyncRunner,
    QuinielaAgent,
    ToolEvent,
    ToolUsage,
    format_tool_args,
)


def _runner() -> AsyncRunner:
    if "async_runner" not in st.session_state:
        st.session_state.async_runner = AsyncRunner()
    return st.session_state.async_runner


def _agent() -> QuinielaAgent:
    runner = _runner()
    if "agent" not in st.session_state:
        agent = QuinielaAgent(profile=DEFAULT_PROFILE)
        runner.run(agent.start())
        st.session_state.agent = agent
    return st.session_state.agent


def _reset_chat() -> None:
    agent: QuinielaAgent = st.session_state.agent
    agent.reset_conversation()
    st.session_state.messages = []


def _tools_payload(tools: list[ToolUsage]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "arguments": tool.arguments,
            "ok": tool.ok,
        }
        for tool in tools
    ]


def _render_tools(tools: list[dict[str, Any]]) -> None:
    if not tools:
        return
    labels = ", ".join(f"`{tool['name']}`" for tool in tools)
    with st.expander(f"Herramientas usadas ({len(tools)}): {labels}", expanded=False):
        for tool in tools:
            status = "ok" if tool.get("ok", True) else "error"
            args = format_tool_args(tool.get("arguments"))
            line = f"- `{tool['name']}` · {status}"
            if args:
                line += f" · `{args}`"
            st.markdown(line)


def main() -> None:
    st.set_page_config(page_title="Quiniela Agentica", page_icon="⚽", layout="centered")
    st.title("Quiniela Agentica")
    st.caption("Asistente para armar quinielas de la Liga MX")

    agent = _agent()
    runner = _runner()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Modelo")
        profile_keys = list(PROFILES.keys())
        current_index = profile_keys.index(agent.profile_name) if agent.profile_name in profile_keys else 0
        selected = st.selectbox(
            "Perfil",
            options=profile_keys,
            index=current_index,
            format_func=lambda key: f"{key} → {PROFILES[key]['model']} ({PROFILES[key]['reasoning_effort']})",
        )
        if selected != agent.profile_name:
            agent.set_profile(selected)
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown(f"**Modelo:** `{agent.model}`")
        st.markdown(f"**Reasoning:** `{agent.reasoning_effort}`")
        st.markdown(f"**Tools:** `{SEARCH_TEAM_NEWS_TOOL_NAME}`, MCP ({len(agent.mcp_tool_names)})")
        if st.button("Nueva conversación", use_container_width=True):
            _reset_chat()
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_tools(message.get("tools") or [])
            st.markdown(message["content"])

    prompt = st.chat_input("Pregunta por partidos, lesiones, rumores…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        events: queue.Queue[ToolEvent] = queue.Queue()

        def on_tool_event(event: ToolEvent) -> None:
            events.put(event)

        future = runner.submit(agent.ask(prompt, on_tool_event=on_tool_event))
        with st.status("Pensando…", expanded=True) as status:
            while not future.done():
                try:
                    while True:
                        event = events.get_nowait()
                        args = format_tool_args(event.arguments)
                        detail = f" `{args}`" if args else ""
                        if event.phase == "start":
                            status.update(label=f"Usando `{event.name}`…", state="running")
                            status.write(f"→ `{event.name}`{detail}")
                        else:
                            mark = "✓" if event.ok else "✗"
                            status.write(f"{mark} `{event.name}`")
                except queue.Empty:
                    pass
                time.sleep(0.05)

            # Drain any late events after completion.
            try:
                while True:
                    event = events.get_nowait()
                    args = format_tool_args(event.arguments)
                    detail = f" `{args}`" if args else ""
                    if event.phase == "start":
                        status.write(f"→ `{event.name}`{detail}")
                    else:
                        mark = "✓" if event.ok else "✗"
                        status.write(f"{mark} `{event.name}`")
            except queue.Empty:
                pass

            result = future.result()
            if result.tools:
                status.update(label=f"Listo · {len(result.tools)} herramienta(s)", state="complete")
            else:
                status.update(label="Listo", state="complete")

        tools_payload = _tools_payload(result.tools)
        _render_tools(tools_payload)

        if result.error:
            error_text = f"Error: {result.error}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "tools": tools_payload})
        elif result.text:
            st.markdown(result.text)
            st.session_state.messages.append({"role": "assistant", "content": result.text, "tools": tools_payload})
        else:
            st.warning("No hubo respuesta del modelo.")


if __name__ == "__main__":
    main()
