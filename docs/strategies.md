# Strategies

VelocityFL ships five aggregation strategies. All five are implemented in Rust and exposed as frozen Python dataclasses in `velocity.strategy`. Pick one based on your threat model and client heterogeneity.

## Decision guide

| If you have… | Reach for |
|---|---|
| IID clients, no adversary, you want a baseline | **`FedAvg`** |
| Heterogeneous clients, drifting local updates | **`FedProx(mu=…)`** |
| Untrusted clients, possible Byzantine updates, <½ of clients compromised | **`FedMedian`** |
| Untrusted clients, up to `f` compromised, want a single winner | **`Krum(f=…)`** |
| Untrusted clients, up to `f` compromised, want to average `m` survivors | **`MultiKrum(f=…, m=…)`** |

All five are value objects: compare with `==`, safe to hash, safe to share between threads.

```python
from velocity import FedAvg, FedProx, FedMedian, Krum, MultiKrum

FedAvg() == FedAvg()                  # True
FedProx(mu=0.01) != FedProx(mu=0.1)   # True
```

---

## `FedAvg`

Weighted average by local sample count — the McMahan et al. (2017) baseline.

```text
w_{t+1} = Σ_k (n_k / n) · w_{t+1}^k
```

**Use when** clients are IID and trusted. Fast, stable, easy to reason about.

```python
from velocity import VelocityServer, FedAvg
server = VelocityServer(model_id=..., dataset=..., strategy=FedAvg())
```

---

## `FedProx`

FedAvg-style aggregation with a proximal term that penalizes local updates for drifting too far from the global model. From Li et al. (2020).

```text
minimize over w:   F_k(w) + (μ/2) · ‖w - w_t‖²
```

**Use when** clients are heterogeneous — differing data distributions, compute budgets, or local epoch counts. The proximal term `μ` dampens client drift.

| Field | Default | Effect |
|---|---|---|
| `mu` | `0.01` | Higher → more conservative updates, slower convergence, better stability on non-IID data. |

```python
from velocity import VelocityServer, FedProx
server = VelocityServer(model_id=..., dataset=..., strategy=FedProx(mu=0.05))
```

---

## `FedMedian`

Coordinate-wise median of client updates. From Yin et al. (2018).

```text
w_{t+1}[i] = median( w_{t+1}^k[i]  for k = 1..K )
```

**Use when** you cannot trust every client. Robust against up to ⌊K/2⌋ Byzantine updates — a poisoned client cannot shift the median, only extend its tail.

```python
from velocity import VelocityServer, FedMedian
server = VelocityServer(model_id=..., dataset=..., strategy=FedMedian())
```

> **Pair with attack simulation** — `FedMedian` is the natural companion to the [attacks catalog](attacks.md). Run the same experiment with `FedAvg` and `FedMedian`, then compare `global_loss` trajectories to see resilience in action.

---

## `Krum`

Select the single client whose update looks most like its `n − f − 2` nearest neighbours, by squared Euclidean distance. From Blanchard et al. (2017).

```text
score(i) = Σ_{j ∈ N_i} ‖w_i - w_j‖²     where N_i = the (n-f-2) closest clients
w_{t+1}  = w_{argmin_i score(i)}
```

**Use when** you assume up to `f` of the `n` clients are Byzantine and you want a provably-bounded winner rather than a blended average.

| Field | Constraint | Effect |
|---|---|---|
| `f` | `n ≥ 2f + 3` | Byzantine-tolerance bound. Raises if the round has fewer than `2f + 3` clients. |

```python
from velocity import VelocityServer, Krum
server = VelocityServer(model_id=..., dataset=..., strategy=Krum(f=2))
# Needs at least 2*2 + 3 = 7 clients per round.
```

The round summary exposes the winner's index so you can audit selections:

```python
summaries = server.run(min_clients=7, rounds=1)
summaries[0]["selected_client_ids"]   # e.g. [3]
```

> **Breakdown point** — Krum provably converges when strictly fewer than `n − 2f − 2` of the `n` clients are Byzantine. Falling below that threshold (too many attackers, or too few honest clients) silently degrades robustness; keep `n ≫ 2f + 3` in practice.

---

## `MultiKrum`

Run the Krum scoring, then return the uniform (not sample-weighted) mean of the `m` lowest-scoring updates. From El Mhamdi et al. (2018).

```text
score(i) = same as Krum
S        = indices of the m lowest scores
w_{t+1}  = (1/m) · Σ_{i ∈ S} w_i
```

**Use when** you want Krum's outlier suppression *and* the variance reduction of averaging. Tunes between the two extremes: `m=1` collapses to Krum, `m=n−f` is Multi-Krum's default.

| Field | Default | Constraint |
|---|---|---|
| `f` | — | `n ≥ 2f + 3`. |
| `m` | `n − f` | Must satisfy `1 ≤ m ≤ n − f`. `None` uses the default. |

```python
from velocity import VelocityServer, MultiKrum
server = VelocityServer(model_id=..., dataset=..., strategy=MultiKrum(f=2, m=5))
```

> **Why uniform, not sample-weighted** — Byzantine clients can lie about `num_samples` to amplify their pull in a weighted average. Multi-Krum deliberately discards that signal; this is pinned by `test_multi_krum_m_equals_n_minus_f_is_uniform_mean` and `strategy::tests::multikrum_uniform_weighting_ignores_sample_counts`.

---

## CLI shorthand

The CLI accepts `Name` for parameter-free strategies and `Name:key=value[,key=value]` for the parameterised ones:

```bash
velocity run  --strategy FedAvg              --model-id demo/m --dataset demo/d
velocity run  --strategy FedProx:mu=0.05     --model-id demo/m --dataset demo/d
velocity run  --strategy Krum:f=2            --model-id demo/m --dataset demo/d --min-clients 7
velocity run  --strategy MultiKrum:f=2,m=5   --model-id demo/m --dataset demo/d --min-clients 7
velocity sweep --strategies FedAvg,Krum:f=1  --rounds 5
```

Sweep TOML files accept either the string form or a dict form — see [Sweep spec](sweep-spec.md).

---

## Adding your own

Strategies are defined in `vfl-core/src/strategy.rs`. To add a new one:

1. Add a variant to the `Strategy` enum and implement the aggregation kernel in Rust. Return an `Aggregation { weights, selected_client_ids }`.
2. Expose a PyO3 constructor in `vfl-core/src/lib.rs` (e.g. `Strategy::trimmed_mean(trim_ratio)`).
3. Add a frozen dataclass to `python/velocity/strategy.py` and include it in `ALL_STRATEGIES`.
4. Wire the isinstance dispatch in `VelocityServer._map_strategy`.
5. Extend `parse_strategy` if the new dataclass has non-trivial coercion needs.
6. Add tests in `tests/` and a section to this page.

See [Architecture](architecture.md) for the full layer map.
