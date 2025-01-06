# Adapted from https://github.com/pydantic/logfire/blob/main/Makefile

.DEFAULT_GOAL := all

# Thanks to https://dwmkerr.com/makefile-help-command/
.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: .uv
.uv:  # Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit:  # Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .uv .p  # Install the package, dependencies, and pre-commit for local developmentre-commit
	uv sync --frozen
	pre-commit install --install-hooks

.PHONY: format
format:  # Format the code
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint
lint:  # Lint the code
	uv run ruff check
	uv run ruff format --check --diff

.PHONY: typecheck
typecheck:  # Typecheck the code
	uv run mypy .

.PHONY: test
test:  # Run the tests
	uv run pytest -vv

.PHONY: testcov
testcov: test  # Run tests and generate a coverage report
	@echo "building coverage html"
	uv run coverage html --show-contexts

.PHONY: test-vcr-once
test-vcr-once:  # Run the tests and record new VCR cassettes
	uv run pytest -vv --record-mode=once

.PHONY: test-vcr-fix
test-vcr-fix:  # Run the last failed tests and rewrite the VCR cassettes
	uv run pytest -vv --last-failed --last-failed-no-failures=none --record-mode=rewrite

.PHONY: test-snapshots-create
test-snapshots-create:  # Run the tests and create new inline-snapshots
	uv run pytest -vv --inline-snapshot=create

.PHONY: test-snapshots-fix
test-snapshots-fix:  # Run the tests and fix inline-snapshots
	uv run pytest -vv --inline-snapshot=fix

.PHONY: docs
docs:  # Build the documentation
	uv run mkdocs build

.PHONY: docs-serve
docs-serve:  # Build and serve the documentation
	uv run mkdocs serve

.PHONY: dep-diagram
dep-diagram:  # Generate a dependency diagram
	uv run pydeps src/magentic --no-show --only "magentic." --rmprefix "magentic." -x "magentic.logger" --exclude-exact "magentic.chat_model"
	open -a Arc magentic.svg

.PHONY: all
all: format lint typecheck test
