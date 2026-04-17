# Attacks

Byzantine-style attack simulations baked into the Rust core. Use them to stress-test your strategy and compare resilience across aggregation algorithms.

## Catalog

| Attack | Parameter | What it does |
|---|---|---|
| `model_poisoning` | `intensity ∈ [0, 1]` | Scales a subset of client weight vectors by a corrupting factor. Directly attacks the aggregator. |
| `sybil_nodes` | `count ≥ 1` | Injects `count` fake clients that submit near-identical malicious updates, amplifying their vote. |
| `gaussian_noise` | `intensity ≥ 0` (σ) | Adds N(0, σ²) noise to client updates. Simulates unreliable clients or rough channels. |
| `label_flipping` | `fraction ∈ [0, 1]` | Flips labels on `fraction` of participating clients — classic data-poisoning primitive. |

## Register via Python

```python
from velocity import VelocityServer, Strategy

server = VelocityServer(
    model_id="demo/model",
    dataset="demo/dataset",
    strategy=Strategy.FedMedian,  # robust strategy — pairs well with attacks
)

server.simulate_attack("model_poisoning", intensity=0.3)
server.simulate_attack("sybil_nodes", count=5)

summaries = server.run(min_clients=10, rounds=5)
for s in summaries:
    for result in s["attack_results"]:
        print(result)
```

Multiple attacks can be registered before a round — they are all applied.

## Register via CLI

```bash
uv run velocity simulate-attack model_poisoning --intensity 0.2
uv run velocity simulate-attack sybil_nodes --count 5
uv run velocity simulate-attack gaussian_noise --intensity 0.1
uv run velocity simulate-attack label_flipping --fraction 0.25
```

Each CLI invocation registers one attack and runs one round, emitting a single JSON summary on stdout.

## Reading the results

Each round summary contains an `attack_results` list. In Python you can hydrate each entry into an `AttackResult` dataclass:

```python
from velocity.attacks import AttackResult

for s in summaries:
    for raw in s["attack_results"]:
        result = AttackResult.from_dict(raw)
        print(result)
# [model_poisoning] Poisoned 2 updates at intensity=0.30 (severity=0.412, clients=2)
```

| `AttackResult` field | Type | Description |
|---|---|---|
| `attack_type` | `str` | Which attack produced this result. |
| `clients_affected` | `int` | Number of clients touched this round. |
| `severity` | `float` | Aggregator-relative impact score in `[0, 1]`. |
| `description` | `str` | Human-readable summary. |

## Designing a resilience experiment

A minimal template for comparing strategies under attack:

```python
from velocity import VelocityServer, Strategy

def run_under_attack(strategy: Strategy) -> list[float]:
    server = VelocityServer(
        model_id="demo/model",
        dataset="demo/dataset",
        strategy=strategy,
    )
    server.simulate_attack("model_poisoning", intensity=0.3)
    return [s["global_loss"] for s in server.run(min_clients=10, rounds=10)]

fedavg_losses    = run_under_attack(Strategy.FedAvg)
fedmedian_losses = run_under_attack(Strategy.FedMedian)
```

See [Strategies](strategies.md) for when each aggregation algorithm is the right pairing.

## Adding a new attack

Attack kernels live under `vfl-core/src/attacks/`. The shape mirrors the strategy contract — implement the mutation in Rust, expose it through the orchestrator's `register_attack` dispatch, then add the identifier to `VALID_ATTACKS` in `python/velocity/attacks.py` and document the parameter here.
