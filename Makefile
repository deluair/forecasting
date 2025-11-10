.PHONY: help install install-dev clean test coverage lint format type-check pre-commit build docs

help:
	@echo "Available commands:"
	@echo "  make install        - Install package in production mode"
	@echo "  make install-dev    - Install package in development mode with dev dependencies"
	@echo "  make clean          - Remove build artifacts and cache files"
	@echo "  make test           - Run tests with pytest"
	@echo "  make coverage       - Run tests with coverage report"
	@echo "  make lint           - Run linters (flake8)"
	@echo "  make format         - Format code with black and isort"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make pre-commit     - Run all pre-commit hooks"
	@echo "  make build          - Build distribution packages"
	@echo "  make docs           - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	rm -rf .tox
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src tests --count --statistics

format:
	black src tests examples
	isort src tests examples

type-check:
	mypy src --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

build: clean
	python -m build

docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation generated in docs/_build/html/index.html"

# Development helpers
check: lint type-check test
	@echo "All checks passed!"

ci: format lint type-check test
	@echo "CI checks complete!"
