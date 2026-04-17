# Contributing to VelocityFL

Thank you for your interest in VelocityFL! We welcome contributions from anyone who loves Rust, Python, or has a passion for federated learning.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Coding Standards](#coding-standards)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you agree to uphold a respectful and inclusive environment.

---

## Getting Started

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Rust | ≥ 1.75 | Core engine |
| Python | ≥ 3.9 | Frontend & tests |
| maturin | ≥ 1.5 | Build Python ↔ Rust bindings |
| Prefect | ≥ 3.0 | Workflow orchestration |

### Clone and Build

```bash
git clone https://github.com/ajbarea/vFL.git
cd vFL

# Install Python build tooling
pip install maturin

# Build the Rust extension and install in development mode
maturin develop

# Install Python dev dependencies
pip install -e ".[dev]"
```

---

## Project Structure

```
vFL/
├── Cargo.toml              # Rust workspace root
├── pyproject.toml          # Python package + maturin configuration
├── vfl-core/               # Rust crate (core engine + PyO3 bindings)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # PyO3 module entry-point
│       ├── orchestrator.rs # FL round management
│       ├── strategy.rs     # FedAvg / FedProx / FedMedian
│       └── security.rs     # Attack simulation & DP
├── python/
│   └── velocity/           # Python package
│       ├── __init__.py
│       ├── server.py       # VelocityServer (public API)
│       ├── strategy.py     # Strategy enum
│       ├── attacks.py      # Attack helpers
│       └── flows.py        # Prefect task & flow definitions
└── tests/                  # Python test suite (pytest)
```

---

## Development Workflow

### Rust changes

```bash
# Check compilation and run Rust unit tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

### Python changes

```bash
# Re-compile Rust extension after Rust code changes
maturin develop

# Lint + format Python
ruff check python/ tests/
ruff format python/ tests/

# Run Python tests
PYTHONPATH=python pytest tests/ -v
```

---

## Running Tests

### All Rust tests

```bash
cargo test
```

### All Python tests

```bash
PYTHONPATH=python pytest tests/ -v
```

### Full integration test (Rust extension required)

```bash
maturin develop
pytest tests/ -v
```

---

## Coding Standards

### Rust

- Follow standard Rust idioms; run `cargo fmt` before committing.
- No `clippy` warnings allowed (`cargo clippy -- -D warnings`).
- Document all public items with `///` doc-comments.
- All new logic must have accompanying unit tests in the same module.

### Python

- Code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/).
- Type hints are required for all public functions and class attributes.
- Use docstrings (Google style) for all public modules, classes, and functions.
- Tests live in `tests/` and follow `pytest` conventions.

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for any new functionality.

3. **Ensure all tests pass** (both Rust and Python).

4. **Run linters** and fix any issues:
   ```bash
   cargo fmt && cargo clippy -- -D warnings
   ruff check python/ tests/ && ruff format python/ tests/
   ```

5. **Open a pull request** against `main` with a clear description of what you changed and why.

---

## Areas Where We Need Help

- 🦀 **Rust**: gRPC/Tonic client-server implementation, differential privacy module, safetensors weight I/O.
- 🐍 **Python**: Hugging Face / PEFT integration, Prefect dashboard widgets, agentic research assistant.
- 📚 **Docs**: tutorials, architecture diagrams, benchmark results.
- 🧪 **Testing**: integration tests with real FL clients, adversarial robustness benchmarks.

---

Powered by the speed of Rust, the reach of Hugging Face, and the clarity of Prefect. 🚀
