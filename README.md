# ⚡ VelocityFL (vFL)

![VelocityFL](docs/assets/velocity-hero.png)

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

The fastest path is the built-in simulator — useful for checking the
install and the attack surface before wiring up real data:

```python
from velocity import VelocityServer, Strategy

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",  # record-keeping string; real loading is below
    strategy=Strategy.FedAvg,
)

server.simulate_attack("gaussian_noise", intensity=0.05)
summaries = server.run(min_clients=1, rounds=1)
print(summaries)
```

For a **real** federated round on a real model + dataset, install the
`[hf,torch]` extras and use `load_federated`:

```bash
uv pip install 'velocity-fl[hf,torch]'
```

```python
from velocity.datasets import load_federated

split = load_federated(
    "ylecun/mnist",
    num_clients=5,
    partition="shard",
    shards_per_client=2,  # McMahan-style non-IID — ~2 digit classes per client
    batch_size=64,
    seed=0,
)
print([c.num_samples for c in split.clients])
```

End-to-end runs live at [`examples/mnist_fedavg.py`](examples/mnist_fedavg.py)
(shard partition) and [`examples/cifar10_fedavg_dirichlet.py`](examples/cifar10_fedavg_dirichlet.py)
(Dirichlet-α partition). Observed convergence is snapshotted in
[`docs/convergence.md`](docs/convergence.md).

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
├── examples/                 # End-to-end demos (e.g. MNIST FedAvg)
├── tests/                    # Python test suite
├── docs/                     # Zensical documentation source
├── .github/workflows/        # CI workflows (tests + docs)
├── pyproject.toml            # Python packaging and tooling
├── Cargo.toml                # Rust workspace manifest
└── zensical.toml             # Docs build config
```

---

## Performance 📊

The claim vFL defends is on **aggregation** — the one step the library
owns. Client-side local training is PyTorch's territory; we don't time
it and we don't claim to speed it up.

On a 1M-parameter model with 10 clients, the Rust aggregation kernel
runs **~92× faster** than the pure-Python fallback through the Python
API (4.75 ms vs 438 ms, `FedAvg`). At 10M params, Rust stays at ~49 ms
per aggregation; pure Python becomes impractical to measure.

End-to-end, this matters less than the raw ratio suggests: on the
[MNIST FedAvg demo](examples/mnist_fedavg.py) (5 clients, ~109K params),
aggregation is ~10 ms of a ~1.3-second round — the rest is torch local
training. The aggregation-speedup lever compounds at robust-aggregator
(Krum, Bulyan), high-client-count, and small-update scales, not at
small-model simulation.

Full methodology, all shape tiers, and honest caveats (PyO3 marshaling
overhead, FedMedian's sort cost, WSL noise) live in
[`docs/benchmarks.md`](docs/benchmarks.md). Convergence evidence lives
in [`docs/convergence.md`](docs/convergence.md). Reproduce with
`make bench` (kernel) and `uv run python examples/mnist_fedavg.py`
(end-to-end).

---

## Coverage 📈

[![Codecov](https://codecov.io/gh/ajbarea/vFL/graph/badge.svg?token=rcYwirIHWk)](https://codecov.io/gh/ajbarea/vFL)

![Sunburst](https://codecov.io/gh/ajbarea/vFL/graphs/sunburst.svg?token=rcYwirIHWk)
![Grid](https://codecov.io/gh/ajbarea/vFL/graphs/tree.svg?token=rcYwirIHWk)
![Icicle](https://codecov.io/gh/ajbarea/vFL/graphs/icicle.svg?token=rcYwirIHWk)

---

## License 📄

[Apache 2.0](LICENSE)