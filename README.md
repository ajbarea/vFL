# ⚡ VelocityFL (vFL)

[![Tests](https://github.com/ajbarea/vFL/actions/workflows/tests.yml/badge.svg)](https://github.com/ajbarea/vFL/actions/workflows/tests.yml)
[![Documentation](https://github.com/ajbarea/vFL/actions/workflows/docs.yml/badge.svg)](https://github.com/ajbarea/vFL/actions/workflows/docs.yml)
[![Codecov](https://codecov.io/gh/ajbarea/vFL/graph/badge.svg?token=rcYwirIHWk)](https://codecov.io/gh/ajbarea/vFL)

VelocityFL is a federated learning orchestration project with a Rust core and a Python-first interface.

---

## What is this? 🧭

VelocityFL provides:
- 🦀 **Rust core (`vfl-core`)** for aggregation, attack simulation, and round orchestration
- 🐍 **Python package (`python/velocity`)** for researcher-facing APIs and a fallback pure-Python orchestrator
- 🖥️ **Typer CLI (`velocity`)** for local experimentation and quick capability inspection
- 📚 **Zensical docs (`docs/`)** deployed via GitHub Actions

---

## Current capabilities ✅

### Aggregation strategies
- `FedAvg`
- `FedProx`
- `FedMedian`

### Built-in attack simulations
- `model_poisoning`
- `sybil_nodes`
- `gaussian_noise`
- `label_flipping`

---

## Quick start 🚀

### 1) Clone and install

```bash
git clone https://github.com/ajbarea/vFL.git
cd vFL

uv sync
uv run maturin develop
```

### 2) Run a minimal Python example

```python
from velocity import VelocityServer, Strategy

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",
    strategy=Strategy.FedAvg,
)

server.simulate_attack("gaussian_noise", intensity=0.05)
summaries = server.run(min_clients=1, rounds=1)
print(summaries)
```

### 3) Use the CLI

```bash
velocity --help
velocity version
velocity strategies
velocity run --model-id test/model --dataset test/dataset --rounds 1 --min-clients 1
velocity simulate-attack model_poisoning --intensity 0.2
```

---

## CLI reference (quick) 💻

- `velocity version` — print package version
- `velocity strategies` — list available strategies
- `velocity run ...` — run rounds and print JSON summaries
- `velocity simulate-attack ...` — register one attack and run a round

Full reference: [`docs/cli.md`](docs/cli.md)

---

## Development 🧪

### Run tests

```bash
# Rust
cargo test --all

# Python
uv run pytest tests/ -v
```

### Build docs locally

```bash
uv run zensical build --clean
```

---

## Documentation 📚

- Source: [`docs/`](docs/)
- Config: [`zensical.toml`](zensical.toml)
- Docs workflow: [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
- Test + coverage workflow: [`.github/workflows/tests.yml`](.github/workflows/tests.yml)

Published site: https://ajbarea.github.io/vFL/

---

## Repository layout 🗂️

```text
vFL/
├── vfl-core/                 # Rust crate and PyO3 bindings
├── python/velocity/          # Python package + CLI
├── tests/                    # Python test suite
├── docs/                     # Zensical documentation source
├── .github/workflows/        # CI workflows (tests + docs)
├── pyproject.toml            # Python packaging and tooling
├── Cargo.toml                # Rust workspace manifest
└── zensical.toml             # Docs build config
```

---

## Performance 📊

On a 1M-parameter model with 10 clients, the Rust aggregation path is
**~45× faster** than the pure-Python fallback through the Python API
(10.5 ms vs 468 ms, `FedAvg`). On a 10M-parameter model, pure Python
becomes impractical while Rust stays at ~138 ms per round.

Full methodology, all shape tiers, and honest caveats (PyO3 marshaling
overhead, FedMedian's sort cost, WSL noise) live in
[`docs/benchmarks.md`](docs/benchmarks.md). Reproduce with `make bench`.

---

## Coverage 📈

[![Codecov](https://codecov.io/gh/ajbarea/vFL/graph/badge.svg?token=rcYwirIHWk)](https://codecov.io/gh/ajbarea/vFL)

![Sunburst](https://codecov.io/gh/ajbarea/vFL/graphs/sunburst.svg?token=rcYwirIHWk)
![Grid](https://codecov.io/gh/ajbarea/vFL/graphs/tree.svg?token=rcYwirIHWk)
![Icicle](https://codecov.io/gh/ajbarea/vFL/graphs/icicle.svg?token=rcYwirIHWk)

---

## License 📄

[Apache 2.0](LICENSE)