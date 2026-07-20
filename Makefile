UV ?= uv
CARGO ?= cargo
EXTRAS := --extra analysis --extra solvers --extra viz

.DEFAULT_GOAL := help

.PHONY: help install build format lint test test-fast test-rs test-py check smoke clean

help:
	@echo "  install    Install Python and Rust dependencies"
	@echo "  build      Rebuild the native extension"
	@echo "  format     Format Python and Rust source"
	@echo "  lint       Run Python and Rust static checks"
	@echo "  test       Run the complete Rust and Python test suites"
	@echo "  test-fast  Run Rust tests and fast Python tests"
	@echo "  check      Run lint and all tests"
	@echo "  smoke      Run the smallest reference experiment"
	@echo "  clean      Remove local build and cache files"

install:
	$(UV) sync $(EXTRAS)

build:
	$(UV) sync $(EXTRAS) --reinstall-package safe-observation-native

format:
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts
	$(CARGO) fmt

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings

test: test-rs test-py

test-fast: test-rs
	$(UV) run pytest -m "not slow"

test-rs:
	$(CARGO) test

test-py:
	$(UV) run pytest

check: lint test

smoke:
	$(UV) run safe-observation run configs/kuhn/static_opponent.toml

clean:
	$(CARGO) clean
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
