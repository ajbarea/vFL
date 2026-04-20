# IMPL: current session — Krum + Multi-Krum

Session-by-session checklist for what's actively in flight. When this
PR ships, its contents get replaced by the next session's plan and a
dated one-liner lands in [ROADMAP → Completed](ROADMAP.md#completed).

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). This file is
the "what are we actually building this PR" scratchpad.

## Why this PR

`velocity._core.Strategy` currently exposes `fed_avg`, `fed_prox`, and
`fed_median`. Byzantine-robust aggregation beyond coordinate-wise
median is the next gap — and the one load-bearing for the performance
story, not just defensive coverage. Krum is O(n²) in clients; the
Rust-vs-Python gap widens with algorithmic complexity, so each robust
aggregator gets measured against a pure-Python reference before any
speedup claim is made. Krum is the first in the ROADMAP chain
(Krum → Multi-Krum → Bulyan → Trimmed Mean → Geometric median);
Multi-Krum is strictly a generalisation (Krum == Multi-Krum with
`m=1`), so they ship together.

## Decisions

- **Krum and Multi-Krum ship in one PR.** Multi-Krum is Krum with
  `m > 1`; bundling keeps the math + Python surface + tests coherent.
- **Single Rust kernel, two strategy variants.** `Strategy::Krum { f }`
  and `Strategy::MultiKrum { f, m }` both dispatch to the same
  distance-scoring routine; selection of 1 vs top-`m` is a parameter.
- **`f` is caller-provided, not inferred.** ArKrum's parameter-free
  variant (arXiv:2505.17226) is a ROADMAP follow-up, not this PR.
- **O(n² · d) distance matrix, not O(n² · log n · d).** Squared
  Euclidean, pre-flatten once per round. No fancy approximate
  nearest-neighbours — at the scales we care about (≤100 clients) the
  dense matrix is simpler and well-vectorised.
- **Measurement discipline.** No speedup claims in `docs/strategies.md`
  or `docs/benchmarks.md` without a pure-Python reference row at the
  same tier. If Rust Krum isn't measurably faster than Python Krum at
  `medium` + `large`, the PR ships Krum but withholds the speedup
  claim.

## Scope

### New code

- **`vfl-core/src/strategy.rs`** — two new `Strategy` variants
  (`Krum { f: usize }`, `MultiKrum { f: usize, m: usize }`) and one
  shared kernel (`krum_select`). Per-layer flatten → per-client
  squared-distance matrix → per-client Krum score (sum of `n-f-2`
  smallest squared distances) → select 1 or top-`m` by score → average
  (Multi-Krum) or pick (Krum). Owned `Vec<f32>` return, same shape
  as existing kernels.
- **`vfl-core/src/lib.rs`** — PyO3 factory methods
  `Strategy.krum(f: int)` and `Strategy.multi_krum(f: int, m: int)`
  alongside the existing `fed_avg` / `fed_median`.
- **`python/velocity/_core.pyi`** — stubs for the two new factories.
- **`python/velocity/strategy.py`** — add `Strategy.Krum` and
  `Strategy.MultiKrum` enum members; `VelocityServer` strategy
  translation picks up `f` / `m` from a strategy-params dict (open
  question Q1).
- **`tests/strategy_reference.py`** — pure-NumPy Krum / Multi-Krum
  implementation used as the correctness oracle. Lives under `tests/`
  so it isn't shipped in the wheel.

### Edited code

- **`python/velocity/strategy.py`** — enum gains two members; server
  translation learns about `f` / `m`.
- **`python/velocity/server.py`** — wire strategy params through
  (depends on Q1).
- **`docs/strategies.md`** — Krum + Multi-Krum sections with the math,
  the breakdown point (`f < n/3` for Byzantine guarantees), and a
  "when to reach for it" sentence. No speedup claim unless the bench
  shows one.
- **`docs/benchmarks.md`** — add Krum / Multi-Krum rows at `small`,
  `medium`, `large` tiers, with matching pure-Python reference rows.

### Tests

- **`tests/test_strategy.py`** — existing file; add:
  - Krum equals Multi-Krum with `m=1` on fixed inputs.
  - Multi-Krum with `m=n` equals FedAvg (all clients selected, uniform
    weights) — a sanity boundary.
  - Rust output matches the NumPy reference (tolerance `1e-5`) across
    randomised fixtures with 5, 10, 20 clients and 3 honest-client
    minority settings.
  - Byzantine breakdown: inject `f` blatantly-bad clients and assert
    the aggregate lands closer to the honest mean than FedAvg does.
- **`tests/test_aggregation_properties.py`** — existing; extend
  hypothesis strategies to cover Krum / Multi-Krum shape invariants
  (output layer names match input, output dtype is `f32`, no NaNs).
- **`tests/bench/test_round_speed.py`** — existing bench harness; add
  Krum / Multi-Krum rows. Pure-Python reference runs at `medium` +
  `large` only (`small` is noise-dominated).
- **Rust-side `cargo test`** — numeric parity tests against a
  hand-computed 3-client fixture (same pattern as the existing
  `fedmedian_odd_clients` test).

## Out of scope

- **ArKrum** (parameter-free `f` estimation). Separate ROADMAP item.
- **Aggregation-aware attacks** (inner-product manipulation,
  PGD poisoning). Listed under ROADMAP → Attacks; useful to have once
  Krum/Bulyan ship but not a prerequisite.
- **Bulyan.** Bulyan composes Multi-Krum with coordinate-wise trimmed
  mean, so its PR lands after this one and after the trimmed-mean PR.
- **Buffer-protocol / numpy return path.** Independent ROADMAP item;
  slotted separately because it touches every kernel's return type.

## Open questions

1. **Strategy parameters through the Python enum.** The current
   `velocity.strategy.Strategy` is a plain `str` enum — `FedProx`'s
   `mu` is handled inside `VelocityServer.run` by reading a
   separate config field, which is ugly but working. Options:
   (a) keep the string enum, extend the server's params dict;
   (b) switch to a dataclass-style enum that carries parameters
   (`Strategy.Krum(f=2)`); (c) expose two parallel APIs (string for
   the CLI, parametric for Python). Lean: (b), but that's an API
   break for `FedProx` callers — so if we go (b), migrate `FedProx`
   in the same PR and note it in the CHANGELOG.
2. **Rust return shape for Multi-Krum.** Average of top-`m` selected
   clients is the canonical form (El Mhamdi et al.). Do we also
   expose the *selected client indices* for auditability? Lean: yes,
   as a separate getter on `RoundSummary` (`selected_client_ids:
   list[int]`). Small surface, big debugging value.
3. **Default `m` for Multi-Krum.** Literature often uses
   `m = n - f`. Make that the default when `m` is omitted? Lean: yes,
   with the rationale documented in the docstring.

## Definition of done

- [ ] `Strategy::Krum { f }` and `Strategy::MultiKrum { f, m }` land
  in `vfl-core/src/strategy.rs` with Rust unit tests.
- [ ] PyO3 factories `Strategy.krum(f)` and `Strategy.multi_krum(f, m)`
  exposed and typed in `_core.pyi`.
- [ ] `tests/test_strategy.py` asserts Rust = NumPy reference within
  `1e-5` tolerance on randomised fixtures.
- [ ] Byzantine-robustness assertion: `f` bad clients don't steer Krum
  further from the honest mean than FedAvg at the same setup.
- [ ] Bench rows added at `small`, `medium`, `large` with pure-Python
  reference rows at `medium` + `large`. No speedup claim in
  `docs/benchmarks.md` unless the measured Rust run is faster than
  the measured Python run at the same tier.
- [ ] `docs/strategies.md` updated with math, breakdown point, and
  "when to reach for it."
- [ ] No new proxy/mock measurements. If a field is called `loss` /
  `distance` / `score`, it's real or it's NaN.

## Up next (non-binding)

Once this ships, the likely session order is:

1. **Trimmed mean** — independent, simple coordinate-wise k-partial
   sort. Unlocks Bulyan.
2. **Bulyan** — composes Multi-Krum + trimmed mean.
3. **Buffer-protocol numpy return path** — touches every kernel; best
   done after the robust suite stabilises so we migrate all return
   types at once.
4. **CodSpeed + crowd-scale bench tier** — makes all of the above
   measurable under a tight enough noise floor to see single-digit
   regressions.
