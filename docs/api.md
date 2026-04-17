# API Reference

The public surface of VelocityFL — Python package and Rust-exposed module.

---

## `velocity` — Python package

### `velocity.VelocityServer`

High-level orchestrator for federated learning experiments. Wraps the Rust-native `Orchestrator` and exposes a researcher-friendly Python API.

```python
class VelocityServer(
    model_id: str,
    dataset: str,
    strategy: Strategy = Strategy.FedAvg,
    storage: str = "local://checkpoints",
    layer_shapes: dict[str, int] | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `run` | `run(min_clients: int = 1, rounds: int = 1)` | `list[dict]` | Execute N rounds. Each dict has `round`, `num_clients`, `global_loss`, `attack_results`. |
| `simulate_attack` | `simulate_attack(attack_type, *, intensity=0.1, count=1, fraction=0.1)` | `None` | Queue an attack for the next round. Can be called before or after `run()`. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `global_weights` | `dict[str, list[float]]` | Current global model weights after the last completed round. |
| `history` | `list[dict]` | JSON-decoded list of all completed round summaries. |

See [Configuration](configuration.md) for field semantics and defaults.

---

### `velocity.Strategy`

Enum of supported aggregation algorithms.

```python
class Strategy(str, Enum):
    FedAvg    = "FedAvg"
    FedProx   = "FedProx"
    FedMedian = "FedMedian"
```

Inherits from `str`, so `Strategy.FedAvg == "FedAvg"` is `True` — you can pass either the enum member or the string literal anywhere a strategy is expected.

See [Strategies](strategies.md) for semantics and decision guide.

---

### `velocity.attacks.AttackResult`

Frozen dataclass describing the outcome of one simulated attack in a round.

```python
@dataclass(frozen=True)
class AttackResult:
    attack_type: str
    clients_affected: int
    severity: float
    description: str

    @classmethod
    def from_dict(cls, d: dict) -> AttackResult: ...
```

---

### `velocity.attacks.VALID_ATTACKS`

Frozen set of recognized attack identifiers.

```python
VALID_ATTACKS: frozenset[str] = frozenset({
    "model_poisoning",
    "sybil_nodes",
    "gaussian_noise",
    "label_flipping",
})
```

---

## `velocity._core` — Rust-backed module

Compiled by `maturin develop`. Imported lazily by `velocity.server`; absent in pure-Python fallback mode.

| Symbol | Kind | Description |
|---|---|---|
| `Orchestrator` | class | Owns per-experiment round state. Accepts `ClientUpdate[]`, returns `RoundSummary`. |
| `ClientUpdate` | class | Rust-side update struct. `num_samples: int`, `weights: dict[str, list[float]]`. |
| `RoundSummary` | class | Round result. `round`, `num_clients`, `global_loss`, `attack_results` (JSON). |
| `Strategy` | class | Strategy factory. Constructors: `Strategy.fed_avg()`, `Strategy.fed_prox(mu)`, `Strategy.fed_median()`. |
| `aggregate` | function | Standalone aggregation kernel — useful for testing. |
| `apply_gaussian_noise` | function | Standalone noise kernel. |

> **Not a stable API** — `velocity._core` is an implementation detail. Prefer `VelocityServer` and `Strategy`. The native module exists so aggregation stays on the hot path — not so you can reach into it.

---

## CLI surface

See the [CLI Reference](cli.md) for every command, flag, and exit code.

---

## Python package imports at a glance

```python
from velocity import VelocityServer, Strategy
from velocity.attacks import AttackResult, VALID_ATTACKS
```

These two lines cover every researcher workflow. Anything else you reach for is an escape hatch — worth pausing to ask whether it should graduate into the public API.
