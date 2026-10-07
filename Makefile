.PHONY: doctor validate test lint docs-check

doctor:
	python -m vision_jev.cli doctor

validate:
	python -m vision_jev.cli validate-data data/samples/example.jsonl

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

docs-check:
	python scripts/check_repo.py
