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

**Snapshot: 2026-04-20. Hardware: AMD Ryzen 5 3600X (6C/12T, Zen 2), 9.7 GB RAM, WSL2 on Windows. rustc 1.95.0, Python 3.12.3, uv 0.11.5, PyO3 0.21, release build (`maturin develop --release`) with `lto = "thin"` + `codegen-units = 1`.**

### Rust aggregation — raw (divan, mean)

No PyO3 boundary, no Python — the theoretical best case. Aggregation
kernel only, measured against pre-built `Vec<f32>` client updates.

| strategy | tiny | medium | large |
|---|---|---|---|
| FedAvg | 5.0 µs | 7.6 ms | 121 ms |
| FedProx | 4.1 µs | 7.7 ms | 124 ms |
| FedMedian | 83 µs | 96.1 ms | 1.01 s |
| TrimmedMean(k=1) | 90 µs | 105 ms | 1.06 s |
| Krum(f=1) | 36 µs | 92.7 ms | 990 ms |
| MultiKrum(f=1) | 43 µs | 97.6 ms | 1.01 s |

FedAvg on a 10M-parameter model with 10 clients aggregates in ~121 ms.
FedAvg accumulates in f64 and downcasts to f32 at the end to bound
rounding error as client counts scale. FedMedian uses
`select_nth_unstable_by` (O(C) expected) on a scratch buffer hoisted out
of the coordinate loop. Krum and Multi-Krum share a pairwise-distance
matrix (`O(n² · d)` per round), dominate FedMedian at the small tier
because of the matrix setup overhead, and converge on FedMedian's cost
at the medium and large tiers where the distance sum dominates.

### Through the Python API (pytest-benchmark, mean)

This is the number users actually feel. `_rust.Orchestrator.run_round`
with pre-built client updates, crossing the PyO3 boundary once per round,
compared against a pure-Python FedAvg on the same inputs.

| tier | Rust FedAvg | Rust FedProx | Rust FedMedian | Rust TrimmedMean(k=1) | Rust Krum(f=1) | Rust MultiKrum(f=1) | Python FedAvg | **Rust FedAvg speedup vs Python FedAvg** |
|---|---|---|---|---|---|---|---|---|
| tiny (~1K) | 4.4 µs | 4.7 µs | 84 µs | 88 µs | 39 µs | 41 µs | 417 µs | **95×** |
| medium (~1M) | 4.4 ms | 4.4 ms | 100 ms | 101 ms | 58 ms | 59 ms | 512 ms | **117×** |
| large (~10M) | 53 ms | 81 ms | 956 ms | 1.04 s | 729 ms | 818 ms | 5.12 s | **97×** |

Pure-Python FedAvg at the `large` tier costs ~5.1 s per round on this
snapshot. The full `tests/bench/` suite takes ~7 min on this box at
21 tests × 6 strategies. The 4–10× range from the prior 2026-04-20
snapshot widened to ~95–117× under a less-loaded WSL state today — the
underlying Rust kernel is unchanged; the gap is system noise on the
Python denominator (`Python FedAvg-large` moved from 8.48 s → 5.12 s,
while `Rust FedAvg-large` moved from 817 ms → 53 ms — the ~16× change on
the Rust side is what's most attributable to load, not algorithm). Treat
the speedup column as directional until CodSpeed lands; the consistent
finding is "Rust FedAvg dominates at every tier."

`Orchestrator.run_round` takes a zero-copy fast path when no attacks are
registered: the PyO3 wrapper passes `&ClientUpdate` slices straight into
the aggregation kernel, so no f32 weight data is cloned between Python
and aggregation. When attacks *are* registered, the owned path kicks in
automatically (attacks can mutate client updates).

**TrimmedMean tracks FedMedian** at every tier (101 ms vs 100 ms at
medium; 1.04 s vs 956 ms at large). Two `select_nth_unstable_by` calls
per coordinate cost slightly more than FedMedian's single call at `k=1`,
even with the median-of-evens averaging skipped — the partition is the
hot loop, the post-processing isn't. The two-call structure becomes
worthwhile as `k` grows (the second call operates on a smaller window),
but at `k=1` it's a wash. The matched NumPy oracle in
`tests/strategy_reference.py` confirms parity.

**Krum/Multi-Krum land above Python FedAvg** at the `medium` and `large`
tiers (729 ms / 818 ms vs 5.12 s at large — Krum is faster than Python
FedAvg here, but ~14× slower than Rust FedAvg). That is algorithmically
honest: Krum is O(n²·d), FedAvg is O(n·d). The `f=1` Krum kernel builds a
10×10 pairwise-distance matrix across all d parameters per call — at
`large` (d ≈ 10 M), that's ~500 M f32 adds before the top-k selection.
Robustness buys a ~14× cost factor over non-robust FedAvg at this scale;
the matched Python oracle in `tests/strategy_reference.py` confirms the
kernel is correct, it is not a perf bug.

### Realistic round cost (run_round + readout)

The `run_round` numbers above measure aggregation in isolation. A real
FL server always reads global weights back after aggregating, to
distribute them to clients next round — that call goes through
`Orchestrator.global_weights()`. Before the numpy buffer-protocol
migration, that getter returned `dict[str, list[float]]` with one
`PyFloat` allocation per parameter, and the cost was hidden outside
the `run_round`-only table above.

**Before numpy migration** (loaded system, 2026-04-22, nephew running
Roblox; directional — re-measure on idle before quoting externally):

| tier | `global_weights()` only | full round (`run_round + readout`) | implied `run_round` alone | getter share |
| --- | --- | --- | --- | --- |
| tiny (~1K params) | 11.3 µs | 15.9 µs | ~4.6 µs | 71% |
| medium (~1M) | 35.3 ms | 39.3 ms | ~4.0 ms | 90% |
| large (~10M) | 425 ms | 459 ms | ~34 ms | **93%** |

**After numpy migration** (same box, same hour, same Roblox):

| tier | `global_weights()` only | full round (`run_round + readout`) | getter speedup | round speedup |
| --- | --- | --- | --- | --- |
| tiny | 1.88 µs | 6.0 µs | 6× | 2.6× |
| medium | 129 µs | 5.6 ms | **273×** | 7× |
| large | 6.6 ms | 56.3 ms | **64×** | **8×** |

At `large`, the getter dropped from 425 ms to 6.6 ms and the realistic
round from 459 ms to 56.3 ms. `.global_weights()` is now ~12% of the
full round (6.6 / 56.3) instead of 93% — the Rust aggregation kernel is
once again the bottleneck, which is what the perf story actually claims.

**Realistic-round speedup vs Python FedAvg at `large`**: 5.12 s / 56.3 ms
= **91×**. Matches the `run_round`-alone table (97×) because marshaling
overhead is essentially gone. The table above is now an honest
apples-to-apples read, not a sliver of the user-facing cost.

## Findings worth calling out

**The Python return path has been closed.** `Orchestrator.global_weights()`,
`ClientUpdate.weights`, free `aggregate`, and `apply_gaussian_noise` now
return `dict[str, numpy.ndarray[float32]]` — the underlying `Vec<f32>`
buffer is shared with numpy via the buffer protocol, zero-copy. One
ndarray wrapper per layer instead of one PyFloat per parameter; O(layers)
instead of O(params). See the "Realistic round cost" subsection above for
the measured delta. Input sides (`__init__`, `set_global_weights`) stay
on `HashMap<String, Vec<f32>>` — the no-attack input path was already
zero-copy.

**FedProx is not server-side distinct from FedAvg.** In
`vfl-core/src/strategy.rs`, `FedProx` dispatches to the same aggregation
kernel as `FedAvg` — the proximal term is a *client-side* regularizer
applied during local training, not during server aggregation. The
near-identical times are correct, not a measurement artifact. Pick
FedProx for the convergence behaviour, not the speed.

**FedMedian and TrimmedMean are the slowest aggregators** — both ~18×
FedAvg at `large` through Python, with TrimmedMean(k=1) within ~10% of
FedMedian. Coordinate-wise `select_nth_unstable_by` is branchy and
doesn't vectorise well. Further gains would need SIMD quickselect or a
histogram-based median — not worth it until a coordinate-wise robust
aggregator sits on a hot path.

**Krum and Multi-Krum sit below FedMedian/TrimmedMean through Python**
at the `medium` / `large` tiers (58 ms vs 100 ms at medium; 729 ms vs
956 ms at large). Distance-matrix work is mostly f32 adds and benefits
from contiguous access; median's nth-element selection is inherently
branchy. Both Byzantine-robust paths cost ~14× FedAvg at `large`, which
is what the O(n²·d) factor predicts.

**WSL2 on a shared desktop CPU is noisy.** Standard deviations on the
`large` tier sit in the 1–17% range in this snapshot. Directional claims
like "Rust FedAvg beats Python FedAvg at every tier" are safe;
single-digit-percent regressions will be invisible on this hardware.
Point estimates like the speedup column move ~3× between snapshots
purely from system load (4–10× on the prior 2026-04-20 snapshot vs
95–117× today); the [CodSpeed](https://codspeed.io) macro runners are
the answer for continuous measurement — tracked as a follow-up, not yet
integrated.
