.PHONY: help docker dockstyle-check codestyle-check linter-check black test lint check
.DEFAULT_GOAL = help

PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-15s\033[0m%s\n", $$1, $$2}'

docker:  # Set up a Docker image for development.
	@printf "Creating Docker image...\n"
	${SHELL} ./scripts/container.sh --build

docker-test:  # Run tests in a Docker image.
	@printf "Testing in Docker image...\n"
	${SHELL} ./scripts/container.sh --test

dockstyle-check:  # Check geoopt with pydocstyle
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle geoopt
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

codestyle-check:  # Check geoopt with black
	@printf "Checking code style with black...\n"
	black --check --diff geoopt tests
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

linter-check:  # Check geoopt with pylint
	@printf "Checking code style with pylint...\n"
	pylint geoopt
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black geoopt tests

test:  # Test code using pytest.
	pytest -v geoopt tests --doctest-modules --html=testing-report.html --self-contained-html

lint: linter-check codestyle-check dockstyle-check # Lint code using black and pylint (no pydocstyle yet).

check: lint test # Both lint and test code. Runs `make lint` followed by `make test`.

clear-pycache:  # clear __pycache__ in the project files (may appear after running tests in docker)
	find -type d -name __pycache__ -exec rm -rf {} +

