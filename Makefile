# Common development commands. Requires the .venv (py3.12) to be active.
.PHONY: install lint typecheck test check docker-config docker-build

install:
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

check: lint typecheck test

docker-config:
	docker compose config

docker-build:
	docker compose build api
