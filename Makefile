.PHONY: doctor validate test lint docs-check

doctor:
	python -m vision_jev.cli doctor

validate:
	python -m vision_jev.cli validate-data data/samples/example.jsonl

test:
	python -m unittest discover -s tests -p 'test_*.py'

lint:
	ruff check vision_jev tests scripts
	ruff format --check vision_jev tests scripts

docs-check:
	python scripts/check_repo.py
