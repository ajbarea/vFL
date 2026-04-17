# ⚡️ VelocityFL (vFL)

> **The "uv" of Federated Learning.**

VelocityFL is a high-performance, Rust-backed federated learning orchestrator designed to unify the fragmented FL ecosystem. It combines the raw execution speed of Rust, the massive model library of Hugging Face, and the world-class observability of Prefect.

Researchers write Python (PyTorch/Hugging Face). The infrastructure runs Rust. The result is a "zero-latency" experimentation engine that scales from a single laptop to 100k+ edge devices.

---

## 🚀 The Vision

Current FL frameworks (Flower, FATE, PySyft) are powerful but often feel like "plumbing" projects — slow to initialise, heavy on memory, and difficult to observe. VelocityFL adopts the **Astral Philosophy**:

1. **Rust Core**: All networking (gRPC/mTLS), serialisation (safetensors), and orchestration logic are written in Rust.
2. **Python Frontend**: Researchers stay in the ecosystem they love (Transformers, PEFT, Datasets).
3. **Observability First**: A native Prefect dashboard replaces cryptic CLI logs with real-time flight control.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🦀 **Rust-Driven Orchestration** | Blazing-fast client-server communication with zero Python GIL overhead |
| 📊 **Prefect Dashboard** | First-class UI for monitoring convergence, client health, and global model drift |
| 🤗 **Native Hugging Face Integration** | Load datasets and transformers directly; PEFT (LoRA/QLoRA) for federated fine-tuning on low-end hardware; safetensors & Xet for ultra-fast, memory-mapped model weight loading |
| 🛡️ **Security Sandbox** | Built-in modules for Differential Privacy, Model Poisoning detection, and Dataset Attack simulations |
| 🤖 **Agentic Memory** | LLM-backed research assistant that analyses experiment logs and suggests aggregation strategy changes |

---

## 🛠 The Stack

| Component | Technology | Role |
|---|---|---|
| Engine | Rust / PyO3 | High-performance core & Python bindings |
| UI / Orchestration | Prefect | Visual workflow management & observability |
| Models / Weights | Hugging Face Safetensors | Fast, secure weight serialisation |
| Storage | Hugging Face Xet | Deduplicated storage for massive model versions |
| Fine-tuning | PEFT | Resource-efficient federated fine-tuning |
| Messaging | gRPC (Tonic) | Secure, low-latency client-server sync |

---

## 💻 Quickstart (Developer Preview)

### 1. Install VelocityFL

```bash
# Install build tooling
pip install maturin

# Clone and build the Rust extension
git clone https://github.com/ajbarea/vFL.git
cd vFL
maturin develop
pip install -e ".[all]"
```

### 2. Define your Federated Task (`research.py`)

```python
from velocity import VelocityServer, Strategy
from prefect import flow

@flow(name="Fed-FineTune-Llama")
def train_llama_federated():
    # 1. Point to your HF Model & Strategy
    vfl = VelocityServer(
        model_id="meta-llama/Llama-3-8B",
        dataset="huggingface/ultrafeedback",
        strategy=Strategy.FedAvg,
        storage="hf-xet://my-namespace/research-repo",
    )

    # 2. Start the Rust-backed orchestrator
    # The Prefect UI will automatically track this run!
    vfl.run(min_clients=10, rounds=5)

if __name__ == "__main__":
    train_llama_federated()
```

### 3. Launch the Dashboard

```bash
prefect server start
# View your FL convergence, client latency, and loss curves in 4K.
```

---

## 🛡 Security & Attack Simulation

VelocityFL isn't just about training; it's about testing resilience. Toggle attack vectors to see how your model holds up:

```python
vfl.simulate_attack("model_poisoning", intensity=0.2)
vfl.simulate_attack("sybil_nodes", count=5)
vfl.simulate_attack("gaussian_noise", intensity=0.05)
vfl.simulate_attack("label_flipping", fraction=0.3)
```

---

## 🗂 Project Structure

```
vFL/
├── Cargo.toml              # Rust workspace
├── pyproject.toml          # Python package + maturin config
├── vfl-core/               # Rust crate (core engine + PyO3 bindings)
│   └── src/
│       ├── lib.rs          # PyO3 module entry-point
│       ├── orchestrator.rs # FL round management
│       ├── strategy.rs     # FedAvg / FedProx / FedMedian
│       └── security.rs     # Attack simulation
├── python/
│   └── velocity/           # Python package
│       ├── server.py       # VelocityServer (public API)
│       ├── strategy.py     # Strategy enum
│       ├── attacks.py      # Attack helpers
│       └── flows.py        # Prefect task & flow definitions
└── tests/                  # pytest suite
```

---

## 🧰 Developer Docs & CLI

- Zensical docs source: [`docs/`](docs/) with project config in [`zensical.toml`](zensical.toml)
- Documentation workflow: [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
- Test + coverage workflow: [`.github/workflows/tests.yml`](.github/workflows/tests.yml)

Quick CLI commands:

```bash
velocity --help
velocity strategies
velocity run --model-id test/model --dataset test/dataset --rounds 1 --min-clients 1
```

---

## 🧪 Running Tests

```bash
# Rust unit tests (14 tests)
cargo test

# Python tests (18 tests)
PYTHONPATH=python pytest tests/ -v

# Full integration (after maturin develop)
pytest tests/ -v
```

---

## 🛣 Roadmap

- **Alpha**: Core Rust engine with PyO3 bindings for PyTorch.
- **Beta**: Native uv-style package management for FL clients; gRPC/mTLS client-server.
- **v1.0**: Full Agentic Research Assistant integration & HF Hub "one-click" deploy.

---

## 🤝 Contributing

We are building the **uv of Federated Learning**. If you know Rust, Python, or have a deep love for Hugging Face, we want your help!

Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

*Powered by the speed of Rust, the reach of Hugging Face, and the clarity of Prefect. Let's build the future of private AI. 🚀*
