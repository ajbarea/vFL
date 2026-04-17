# Strategies

VelocityFL ships three aggregation strategies. All three are implemented in Rust and exposed through `velocity.Strategy`. Pick one based on your threat model and client heterogeneity.

## Decision guide

| If you have… | Reach for |
|---|---|
| IID clients, no adversary, you want a baseline | **`FedAvg`** |
| Heterogeneous clients, drifting local updates | **`FedProx`** |
| Untrusted clients, possible Byzantine updates | **`FedMedian`** |

---

## `FedAvg`

Weighted average by local sample count — the McMahan et al. (2017) baseline.

```text
w_{t+1} = Σ_k (n_k / n) · w_{t+1}^k
```

**Use when** clients are IID and trusted. Fast, stable, easy to reason about.

```python
from velocity import VelocityServer, Strategy
server = VelocityServer(model_id=..., dataset=..., strategy=Strategy.FedAvg)
```

---

## `FedProx`

FedAvg-style aggregation with a proximal term that penalizes local updates for drifting too far from the global model. From Li et al. (2020).

```text
minimize over w:   F_k(w) + (μ/2) · ‖w - w_t‖²
```

**Use when** clients are heterogeneous — differing data distributions, compute budgets, or local epoch counts. The proximal term `μ` dampens client drift.

| Parameter | Default | Effect |
|---|---|---|
| `mu` | `0.01` | Higher → more conservative updates, slower convergence, better stability on non-IID data. |

```python
server = VelocityServer(model_id=..., dataset=..., strategy=Strategy.FedProx)
```

---

## `FedMedian`

Coordinate-wise median of client updates. From Yin et al. (2018).

```text
w_{t+1}[i] = median( w_{t+1}^k[i]  for k = 1..K )
```

**Use when** you cannot trust every client. Robust against up to ⌊K/2⌋ Byzantine updates — a poisoned client cannot shift the median, only extend its tail.

```python
server = VelocityServer(model_id=..., dataset=..., strategy=Strategy.FedMedian)
```

> **Pair with attack simulation** — `FedMedian` is the natural companion to the [attacks catalog](attacks.md). Run the same experiment with `FedAvg` and `FedMedian`, then compare `global_loss` trajectories to see resilience in action.

---

## Adding your own

Strategies are defined in `vfl-core/src/strategy.rs`. To add a new one:

1. Add a variant to the `Strategy` enum and implement the aggregation in Rust.
2. Expose a constructor via PyO3 (e.g. `Strategy::fed_trimmed_mean(trim_ratio)`).
3. Add a matching variant to `python/velocity/strategy.py`.
4. Wire it up in `VelocityServer._map_strategy`.
5. Add a test in `tests/` and a row to this page.

See [Architecture](architecture.md) for the full layer map.
