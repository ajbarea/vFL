# Benchmarks

VelocityFL's pitch is "the uv of Federated Learning" — a Rust core under a
Python API. That's a performance claim, and it should be measured, not
marketed. This page documents the measurement methodology and shows the
current numbers.

## What we measure

Every FL round is: clients produce weights (user-side, out of our hands) →
the server aggregates them. The only thing the library controls is the
aggregation step, so that's what we time. Client-weight construction is
outside the timed region.

Two layers of measurement serve different purposes:

| Layer | Harness | File | Answers |
|---|---|---|---|
| **Rust aggregate, raw** | [divan](https://docs.rs/divan) | `vfl-core/benches/aggregate.rs` | How fast is the inner loop? |
| **Through the Python API** | [pytest-benchmark](https://pytest-benchmark.readthedocs.io) | `tests/bench/test_round_speed.py` | How fast is the user-visible path (`_rust.Orchestrator.run_round`) vs the pure-Python fallback? |

The second layer is the one the tagline is defended on — real users call
through PyO3, not Rust directly.

## How to reproduce

```bash
uv sync
uv run maturin develop --release
make bench
```

Or invoke the harnesses directly:

```bash
cargo bench --bench aggregate
uv run pytest tests/bench/ --benchmark-only --benchmark-columns=mean,stddev,rounds --benchmark-sort=mean
```

## Shape tiers

Three tiers, chosen to span realistic FL model sizes:

| tier | layers | total params | rough analogue |
|---|---|---|---|
| `tiny` | 4 | ~970 | smoke test / FedAvg sanity check |
| `medium` | 10 | ~1.0M | ResNet-18-scale |
| `large` | 16 | ~10.0M | ResNet-50-scale |

All tiers use **10 clients**. Weights are deterministic f32 in [-0.1, 0.1].

## Results

**Snapshot: 2026-04-17. Hardware: AMD Ryzen 5 3600X (6C/12T, Zen 2), 9.7 GB RAM, WSL2 on Windows. rustc 1.95.0, Python 3.12.3, uv 0.11.5, PyO3 0.21, release build (`maturin develop --release`).**

### Rust aggregation — raw (divan, mean)

No PyO3 boundary, no Python — the theoretical best case. Aggregation
kernel only, measured against pre-built `Vec<f32>` client updates.

| strategy | tiny | medium | large |
|---|---|---|---|
| FedAvg | 2.6 µs | 4.1 ms | 74.6 ms |
| FedProx | 2.6 µs | 3.8 ms | 69.9 ms |
| FedMedian | 380 µs | 372 ms | 3.76 s |

Read: FedAvg on a 10M-parameter model with 10 clients aggregates in ~75 ms.
FedMedian's coordinate-wise sort is ~50× more expensive at scale.

### Through the Python API (pytest-benchmark, mean)

This is the number users actually feel. `_rust.Orchestrator.run_round`
with pre-built client updates, crossing the PyO3 boundary once per round,
compared against a pure-Python FedAvg on the same inputs.

| tier | Rust FedAvg | Rust FedProx | Rust FedMedian | Python FedAvg | **Speedup (Rust FedAvg / Python)** |
|---|---|---|---|---|---|
| tiny (~1K) | 8.76 µs | 8.80 µs | 438 µs | 400 µs | **45.7×** |
| medium (~1M) | 10.5 ms | 11.1 ms | 441 ms | 468 ms | **44.6×** |
| large (~10M) | 138 ms | 131 ms | 4.02 s | *not measured* | — |

Pure-Python FedAvg at the `large` tier is deliberately skipped — one
iteration takes several minutes, which is itself the finding.

**Speedup lands around 45×** at both tiers where the comparison is
tractable. That holds up the "uv of Federated Learning" framing at the
order-of-magnitude level; further gains from Rust would require
algorithmic changes, not tighter code.

## Findings worth calling out

**PyO3 overhead is real.** Subtracting raw divan from the Python-surface
number gives the boundary cost: ~6 µs at tiny, ~6 ms at medium, ~60 ms
at large — roughly 40–50% of the large-tier compute time is spent
marshaling `list[float]` across the FFI. This is the single biggest
opportunity for further speedup; `numpy.ndarray` / buffer-protocol paths
are a known follow-up, not done here.

**FedProx is not server-side distinct from FedAvg.** In
`vfl-core/src/strategy.rs`, `FedProx` dispatches to the same aggregation
kernel as `FedAvg` — the proximal term is a *client-side* regularizer
applied during local training, not during server aggregation. The
near-identical times are correct, not a measurement artifact. Pick
FedProx for the convergence behaviour, not the speed.

**FedMedian is expensive.** Coordinate-wise sorting across clients is
branchy and non-SIMD-friendly; it costs 40–50× FedAvg at every tier. If
you don't need Byzantine robustness, don't pay for it.

**WSL2 on a shared desktop CPU is noisy.** Standard deviations on the
`large` tier sit in the 5–15% range. Directional claims like "Rust is
~45× faster" are safe; single-digit-percent regressions will be invisible
on this hardware. The [CodSpeed](https://codspeed.io) macro runners are
the answer for continuous measurement — tracked as a follow-up, not yet
integrated.

## Follow-ups

- [ ] CodSpeed CI integration (bare-metal macro runners, PR-comment perf
  tracking)
- [ ] Buffer-protocol PyO3 path to halve FFI marshaling cost
- [ ] Include a crowd-scale tier (50–100 clients) once CodSpeed lands so
  the noise floor is tight enough to see the effect
