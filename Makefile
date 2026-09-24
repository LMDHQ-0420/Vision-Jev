.PHONY: doctor validate test lint docs-check

doctor:
	PYTHONPATH=src python -m vision_jev.cli doctor

validate:
	PYTHONPATH=src python -m vision_jev.cli validate-data data/samples/example.jsonl

test:
	PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

docs-check:
	PYTHONPATH=src python scripts/check_repo.py
