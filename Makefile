.PHONY: help dockstyle-check codestyle-check linter-check black test lint check sphinx-check
.DEFAULT_GOAL = help

PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-15s\033[0m%s\n", $$1, $$2}'

docstyle-check:  # Check geoopt with pydocstyle
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle geoopt
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

sphinx-check:
	@printf "Checking sphinx build...\n"
	SPHINXOPTS=-W make -C docs -f Makefile clean html
	@printf "\033[1;34mSphinx passes!\033[0m\n\n"

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

lint: linter-check codestyle-check docstyle-check sphinx-check # Lint code using black, pylint, pydocstyle and sphinx.

check: lint test # Both lint and test code. Runs `make lint` followed by `make test`.


