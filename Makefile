.PHONY: help install run dev test clean

help:
	@echo "Available commands:"
	@echo "  install  - Install dependencies"
	@echo "  run      - Run production server"
	@echo "  dev      - Run development server with reload"
	@echo "  test     - Run tests"
	@echo "  clean    - Remove Python cache files"

install:
	uv sync

run:
	uv run uvicorn core.app.main:app

dev:
	uv run uvicorn core.app.main:app --reload

test:
	python -m pytest

clean:
	find . -type d -name "__pycache__" -delete
	find . -name "*.pyc" -delete