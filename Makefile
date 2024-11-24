# Adapted from https://github.com/pydantic/logfire/blob/main/Makefile

.DEFAULT_GOAL := all

.PHONY: .uv  # Check that uv is installed
.uv:
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  # Check that pre-commit is installed
.pre-commit:
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install  # Install the package, dependencies, and pre-commit for local development
install: .uv .pre-commit
	uv sync --frozen
	pre-commit install --install-hooks

.PHONY: format  # Format the code
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint  # Lint the code
lint:
	uv run ruff check
	uv run ruff format --check --diff

.PHONY: typecheck  # Typecheck the code
typecheck:
	uv run mypy .

.PHONY: test  # Run the tests
test:
	uv run pytest -vv

.PHONY: testcov  # Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	uv run coverage html --show-contexts

.PHONY: test-fix-vcr  # Run the last failed tests and rewrite the VCR cassettes
test-fix-vcr:
	uv run pytest -vv --last-failed --last-failed-no-failures=none --record-mode=rewrite

.PHONY: docs  # Build the documentation
docs:
	uv run mkdocs build

.PHONY: docs-serve  # Build and serve the documentation
docs-serve:
	uv run mkdocs serve

.PHONY: all
all: format lint test
