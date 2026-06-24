# every-eval-ever — task runner. One target per task.
.PHONY: install repro test lint format clean help

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## editable install + dev tools
	python -m pip install -e . && python -m pip install -r requirements.txt

repro:  ## regenerate every paper number -> analysis_output/paper_numbers.json
	python scripts/reproduce_paper.py

test:  ## run the test suite
	python -m pytest -q

lint:  ## ruff + black --check
	ruff check src tests scripts && black --check src tests scripts

format:  ## auto-format
	ruff check --fix src tests scripts && black src tests scripts

clean:  ## remove caches and generated analysis outputs
	rm -rf .pytest_cache **/__pycache__ analysis_output/*.csv analysis_output/*.json
