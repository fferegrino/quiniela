import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """Eres un asistente experto en fútbol mexicano especializado en ayudar a llenar quinielas de la Liga MX."""


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
        except Exception as exc:
            print(f"Error: {exc}")
            messages.pop()
            continue

        assistant_message = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_message})
        print(f"\nAssistant: {assistant_message}\n")


if __name__ == "__main__":
    main()
