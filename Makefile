.PHONY: format lint check

format:
	uv run ruff format .
	uv run ruff check --fix .
	uv run tombi format .

lint:
	uv run ruff check .
	uv run tombi check .

check: lint
	uv run ruff format --check .
