.PHONY: build clean test docs lint

build:
	pip install -e .

clean:
	rm -rf build/ dist/ src/*.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf examples_output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete

test:
	python -m pytest tests/ -v

docs:
	PYTHONPATH=src python3 docs/generate_figures.py

lint:
	ruff check src/ tests/ examples/
