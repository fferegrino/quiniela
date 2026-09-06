"""Streamlit UI for the quiniela chat agent.

Chat logic lives in ``chat_agent``; this file only handles presentation and session state.
"""

from __future__ import annotations

import streamlit as st

from chat_agent import (
    DEFAULT_PROFILE,
    PROFILES,
    SEARCH_TEAM_NEWS_TOOL_NAME,
    AsyncRunner,
    QuinielaAgent,
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
            st.markdown(message["content"])

    prompt = st.chat_input("Pregunta por partidos, lesiones, rumores…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            result = runner.run(agent.ask(prompt))
        if result.error:
            error_text = f"Error: {result.error}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        elif result.text:
            st.markdown(result.text)
            st.session_state.messages.append({"role": "assistant", "content": result.text})
        else:
            st.warning("No hubo respuesta del modelo.")


if __name__ == "__main__":
    main()
