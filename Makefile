.PHONY: install test lint format docs clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	playwright install

test:
	pytest

lint:
	flake8 nova tests
	mypy nova tests

format:
	black nova tests
	isort nova tests

docs:
	sphinx-build -b html docs docs/_build/html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 