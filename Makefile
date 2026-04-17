##
## vFL — Makefile
## Thin wrapper over the cross-platform dev runner at scripts/dev.py.
##
## Canonical (platform-agnostic) workflow:
##     uv run python scripts/dev.py <command>
##
## `make <target>` and `uv run python scripts/dev.py <target>` are equivalent.
## Every lint/CI target FIXES first (ruff --fix, ruff format, cargo fmt,
## cargo clippy --fix), then CHECKS what remains — so you only see errors
## the tooling could not auto-resolve.
##

.PHONY: help check-env sync build docs fix lint lint-py lint-rs test test-py test-rs validate ci clean
.DEFAULT_GOAL := help

DEV := uv run python scripts/dev.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

check-env:              ## Verify uv, cargo, and rustc are on PATH
	@$(DEV) check-env

sync:                   ## uv sync (install Python + dev dependencies)
	@$(DEV) sync

build:                  ## Build Rust extension in-place (maturin develop)
	@$(DEV) build

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

docs:                   ## Serve docs site on http://localhost:8000 (zensical serve)
	@$(DEV) docs

# ---------------------------------------------------------------------------
# Quality gates (fix-first, then check)
# ---------------------------------------------------------------------------

fix:                    ## Run every auto-fixer; skip the check pass
	@$(DEV) fix

lint:                   ## Rust + Python: auto-fix, then check
	@$(DEV) lint

lint-py:                ## Python-only: ruff --fix + ruff format, then ruff/ty checks
	@$(DEV) lint-py

lint-rs:                ## Rust-only: cargo fmt + clippy --fix, then fmt/clippy checks
	@$(DEV) lint-rs

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:                   ## Run Rust + Python test suites
	@$(DEV) test

test-py:                ## pytest tests/ -v
	@$(DEV) test-py

test-rs:                ## cargo test --all
	@$(DEV) test-rs

# ---------------------------------------------------------------------------
# Combined workflows
# ---------------------------------------------------------------------------

validate:               ## Fast feedback: lint + Python tests
	@$(DEV) validate

ci:                     ## Mirror CI end-to-end: sync, build, lint, tests
	@$(DEV) ci

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean:                  ## Remove build + cache directories
	@$(DEV) clean

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:                   ## Show this help message
	@$(DEV) help
