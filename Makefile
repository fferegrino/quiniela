.PHONY: format lint check ui cli

format:
	uv run ruff format .
	uv run ruff check --fix .
	uv run tombi format .

lint:
	uv run ruff check .
	uv run tombi check .

check: lint
	uv run ruff format --check .

cli:
	uv run python main.py

ui:
	uv run streamlit run app.py
