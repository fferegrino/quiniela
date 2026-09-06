"""CLI frontend for the quiniela chat agent."""

import asyncio

from chat_agent import (
    DEFAULT_PROFILE,
    SEARCH_TEAM_NEWS_TOOL_NAME,
    QuinielaAgent,
    resolve_profile,
)


async def main() -> None:
    """Connect to the Liga MX MCP server and run the interactive chat loop."""
    async with QuinielaAgent(profile=DEFAULT_PROFILE) as agent:
        print(f">>> System prompt: {agent.system_prompt}")
        print(f">>> Profile: {agent.profile_name} → {agent.model} (reasoning_effort={agent.reasoning_effort})")
        print(f">>> Local tools: {SEARCH_TEAM_NEWS_TOOL_NAME}")
        print(f">>> MCP tools: {', '.join(agent.mcp_tool_names) or '(none)'}")
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
                agent.set_profile(user_input[1:])
                print(f"Switched to {agent.profile_name} → {agent.model} (reasoning_effort={agent.reasoning_effort})")
                continue

            result = await agent.ask(user_input)
            if result.error:
                print(f"Error: {result.error}")
                continue
            if result.text is None:
                continue

            print(f"\nAssistant: {result.text}\n")


if __name__ == "__main__":
    # Validate default profile early for clearer startup errors.
    resolve_profile(DEFAULT_PROFILE)
    asyncio.run(main())
