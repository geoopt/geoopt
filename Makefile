.PHONY: help linter-check ruff-check test lint check sphinx-check
.DEFAULT_GOAL = help

PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-15s\033[0m%s\n", $$1, $$2}'

sphinx-check:
	@printf "Checking sphinx build...\n"
	SPHINXOPTS=-W make -C docs -f Makefile clean html
	@printf "\033[1;34mSphinx passes!\033[0m\n\n"

linter-check:  # Check geoopt with ruff
	@printf "Checking code with ruff...\n"
	ruff check geoopt
	@printf "\033[1;34mLint checks pass!\033[0m\n\n"

ruff-check: linter-check  # Run ruff linter checks

ruff:  # Fix auto-fixable issues with ruff.
	ruff check --fix geoopt

test:  # Test code using pytest.
	pytest -v geoopt tests --doctest-modules --html=testing-report.html --self-contained-html

lint: ruff-check sphinx-check  # Lint code using ruff and sphinx.

check: lint test # Both lint and test code. Runs `make lint` followed by `make test`.
