# --------------------------------------------------------- Environment related targets

## Create a virtual environment
env:
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip

## Install pre-commit hooks
pre-commit:
	pip install pre-commit
	pre-commit install

## Install package
init:
	pip install --no-cache-dir -r requirements-dev.txt


# ---------------------------------------------------------------- Test related targets

PYTEST_ARGS = --show-capture no --full-trace --verbose --cov hubble/ --cov-report term-missing --cov-report html

## Run tests
test:
	pytest $(PYTEST_ARGS) $(TESTS_PATH)


# ---------------------------------------------------------- Code style related targets

SRC_CODE = finetuner/ tests/

## Run the flake linter
flake:
	flake8 $(SRC_CODE)

## Run the black formatter
black:
	black $(SRC_CODE)

## Run the isort import formatter
isort:
	isort $(SRC_CODE)

## Dry run the black formatter
black-check:
	black --check $(SRC_CODE)

## Dry run the isort import formatter
isort-check:
	isort --check $(SRC_CODE)

## Check code style
style: flake black-check isort-check # mypy
