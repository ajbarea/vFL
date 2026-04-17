# Getting Started

## Install dependencies

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
cargo test
PYTHONPATH=python pytest tests/ -v
```

## Basic Python usage

```python
from velocity import VelocityServer, Strategy

vfl = VelocityServer(
    model_id="meta-llama/Llama-3-8B",
    dataset="huggingface/ultrafeedback",
    strategy=Strategy.FedAvg,
)

vfl.simulate_attack("model_poisoning", intensity=0.2)
vfl.run(min_clients=10, rounds=5)
```
