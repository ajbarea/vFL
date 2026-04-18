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

**Snapshot: 2026-04-18. Hardware: AMD Ryzen 5 3600X (6C/12T, Zen 2), 9.7 GB RAM, WSL2 on Windows. rustc 1.95.0, Python 3.12.3, uv 0.11.5, PyO3 0.21, release build (`maturin develop --release`) with `lto = "thin"` + `codegen-units = 1`.**

### Rust aggregation — raw (divan, mean)

No PyO3 boundary, no Python — the theoretical best case. Aggregation
kernel only, measured against pre-built `Vec<f32>` client updates.

| strategy | tiny | medium | large |
|---|---|---|---|
| FedAvg | 3.5 µs | 5.4 ms | 74.2 ms |
| FedProx | 3.4 µs | 5.2 ms | 78.1 ms |
| FedMedian | 75 µs | 82.7 ms | 870 ms |

FedAvg on a 10M-parameter model with 10 clients aggregates in ~75 ms.
FedAvg accumulates in f64 and downcasts to f32 at the end to bound
rounding error as client counts scale. FedMedian uses
`select_nth_unstable_by` (O(C) expected) on a scratch buffer hoisted out
of the coordinate loop.

### Through the Python API (pytest-benchmark, mean)

This is the number users actually feel. `_rust.Orchestrator.run_round`
with pre-built client updates, crossing the PyO3 boundary once per round,
compared against a pure-Python FedAvg on the same inputs.

| tier | Rust FedAvg | Rust FedProx | Rust FedMedian | Python FedAvg | **Speedup (Rust FedAvg / Python)** |
|---|---|---|---|---|---|
| tiny (~1K) | 5.84 µs | 5.19 µs | 77.2 µs | 383 µs | **65.6×** |
| medium (~1M) | 4.75 ms | 4.64 ms | 85.4 ms | 438 ms | **92.2×** |
| large (~10M) | 49.3 ms | 53.1 ms | 909 ms | *not measured* | — |

Pure-Python FedAvg at the `large` tier is deliberately skipped — one
iteration takes several minutes, which is itself the finding.

**FedAvg speedup lands around 65–90×** at both tiers where the
comparison is tractable. `Orchestrator.run_round` takes a zero-copy
fast path when no attacks are registered: the PyO3 wrapper passes
`&ClientUpdate` slices straight into the aggregation kernel, so no f32
weight data is cloned between Python and aggregation. When attacks
*are* registered, the owned path kicks in automatically (attacks can
mutate client updates).

## Findings worth calling out

**Remaining PyO3 overhead is in the return path.** Subtracting raw
divan from the Python-surface number gives the boundary cost: ~2 µs at
tiny, ~0 ms at medium (kernel + return now fit in the same budget as
raw), ~-25 ms at large (noise range). The input side is now zero-copy
on the no-attack path; the return side still marshals
`HashMap<String, Vec<f32>>` → Python `dict[str, list[float]]`, which
allocates one PyFloat per parameter. A `numpy.ndarray` / buffer-protocol
return path is the next lever.

**FedProx is not server-side distinct from FedAvg.** In
`vfl-core/src/strategy.rs`, `FedProx` dispatches to the same aggregation
kernel as `FedAvg` — the proximal term is a *client-side* regularizer
applied during local training, not during server aggregation. The
near-identical times are correct, not a measurement artifact. Pick
FedProx for the convergence behaviour, not the speed.

**FedMedian costs ~12× FedAvg at large tier.** Coordinate-wise median
is inherently branchy. Further gains would require SIMD quickselect or
a histogram-based median — not worth it until Byzantine-robust
aggregation sits on a hot path.

**WSL2 on a shared desktop CPU is noisy.** Standard deviations on the
`large` tier sit in the 5–15% range. Directional claims like "Rust
aggregation is ~90× faster than pure-Python aggregation" are safe;
single-digit-percent regressions will be invisible
on this hardware. The [CodSpeed](https://codspeed.io) macro runners are
the answer for continuous measurement — tracked as a follow-up, not yet
integrated.
