# IMPL: current session — Trimmed Mean

Session-by-session checklist for what's actively in flight. When this
PR ships, its contents get replaced by the next session's plan and a
dated one-liner lands in [ROADMAP → Completed](ROADMAP.md#completed).

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). This file is
the "what are we actually building this PR" scratchpad.

## Why this PR

Trimmed mean is the next robust aggregator and the prerequisite for
Bulyan (Bulyan = Multi-Krum → coordinate-wise trimmed mean). It's also
load-bearing for the perf story: a coordinate-wise k-partial sort
vectorises better than FedMedian's full median, so the Rust-vs-Python
gap should be visible at `medium` and `large` tiers. Cheaper than
FedMedian per coordinate, dimension-independent, simpler than Krum's
O(n²·d) — a strict midpoint between FedAvg and the heavyweight robust
aggregators.

## Decisions

- **Symmetric trim only.** `Strategy::TrimmedMean { k }` drops the `k`
  smallest *and* `k` largest values per coordinate, then means the
  remaining `n - 2k`. Asymmetric trim (`k_lo`, `k_hi`) is a follow-up
  if anyone asks; the literature default and phalanx's implementation
  are both symmetric.
- **`k` is caller-provided.** Same call as Krum's `f` — the Byzantine
  budget is a deployment decision, not something we infer. Validation:
  `k >= 0` and `2*k < n` (need at least one element to mean). `k = 0`
  reduces to FedAvg with uniform weights (sanity boundary tested).
- **Per-coordinate, not per-client.** Different coordinates retain
  different client subsets; there is no single "selected" client list.
  `selected_client_ids` therefore returns `0..n` (all participating)
  for trimmed mean — same convention as FedMedian. The docstring
  documents this explicitly so callers don't read the field as
  "survivors of the trim."
- **Two `select_nth_unstable` calls per coordinate.** First selects
  the `k`-th order statistic (separates the `k` smallest); second
  operates on `scratch[k..]` to select the `(n-2k-1)`-th (separates
  the `k` largest from the middle band). Sum the middle, divide by
  `n - 2k`. O(n) average per coordinate — same complexity as
  FedMedian, with a tighter inner loop because no median-of-evens
  branch.
- **Uniform weighting over the kept elements.** Not sample-weighted —
  matches the Yin et al. 2018 specification and our Multi-Krum
  convention.
- **Measurement discipline.** Same rule as the Krum PR: no speedup
  claim in `docs/strategies.md` or `docs/benchmarks.md` without a
  pure-Python reference row at the same tier. If Rust trimmed mean
  isn't measurably faster than the NumPy reference at `medium` +
  `large`, the PR ships the kernel but withholds the speedup claim.

## Scope

### New code

- **`vfl-core/src/strategy.rs`** — one new variant
  `Strategy::TrimmedMean { k: usize }` and one kernel
  (`trimmed_mean`). Shape mirrors `fed_median`: hoist per-layer client
  slices out of the coordinate loop, reuse a single scratch buffer per
  layer, two `select_nth_unstable_by` calls per coordinate. Owned
  `Vec<f32>` return, populates `selected_client_ids` with `0..n`.
- **`vfl-core/src/lib.rs`** — PyO3 factory
  `Strategy.trimmed_mean(k: int)` alongside the existing factories.
- **`python/velocity/_core.pyi`** — stub for the new factory.
- **`python/velocity/strategy.py`** — add
  `@dataclass(frozen=True) class TrimmedMean: k: int`, extend the
  `Strategy` union, append to `ALL_STRATEGIES`.

### Edited code

- **`python/velocity/server.py`** — strategy translation already
  isinstance-dispatches; one new arm for `TrimmedMean`.
- **`python/velocity/cli.py`** — string-form parser already handles
  `Name:key=value`; verify `trimmedmean:k=2` round-trips. Test added.
- **`tests/strategy_reference.py`** — add a NumPy reference
  (`numpy.partition` per coordinate) used as the parity oracle.
- **`docs/strategies.md`** — Trimmed Mean section: math, breakdown
  point (`f <= k`), comparison to FedMedian (cheaper for small `k`,
  same when `k = (n-1)/2`), and a "when to reach for it" sentence.
  No speedup claim unless the bench shows one.
- **`docs/benchmarks.md`** — add Trimmed Mean rows at `small`,
  `medium`, `large` tiers, with matching pure-NumPy reference rows
  at `medium` + `large`.

### Tests

- **`tests/test_strategy.py`** — add:
  - `TrimmedMean(k=0)` equals FedAvg with uniform sample weights on a
    fixed input (sanity boundary).
  - Rust output matches NumPy reference (tolerance `1e-5`) across
    randomised fixtures with 5, 10, 20 clients and 2–3 `k` values.
  - Byzantine breakdown: inject `k` blatantly-bad clients; assert the
    aggregate is closer to the honest mean than FedAvg's is.
  - Validation errors: `k < 0`, `2*k >= n`.
- **`tests/test_aggregation_properties.py`** — extend hypothesis
  strategies to cover `TrimmedMean` shape invariants (output layer
  names match input, dtype `f32`, no NaNs, `selected_client_ids`
  equals `list(range(n))`).
- **`tests/bench/test_round_speed.py`** — add Trimmed Mean rows.
  Pure-NumPy reference at `medium` + `large` only.
- **Rust-side `cargo test`** — numeric parity against a hand-computed
  5-client fixture (mirroring `fedmedian_even_clients_averages_middle_pair`).
  Edge-case tests: `k=0` reduces to uniform mean, `2*k+1 == n` reduces
  to single-element median, `k < 0`/`2*k >= n` error paths.

## Out of scope

- **Asymmetric trim (`k_lo`, `k_hi`).** Separate item if a caller asks.
- **Bulyan.** This PR is a prerequisite, not the destination.
  Bulyan's PR composes `MultiKrum` + `TrimmedMean` and lands next.
- **Buffer-protocol / numpy return path.** Still queued — touches
  every kernel's return type; best done after the robust suite
  stabilises so the migration is one PR.

## Definition of done

- [ ] `Strategy::TrimmedMean { k }` lands in
  `vfl-core/src/strategy.rs` with Rust unit tests.
- [ ] PyO3 factory `Strategy.trimmed_mean(k)` exposed and typed in
  `_core.pyi`.
- [ ] `tests/test_strategy.py` asserts Rust = NumPy reference within
  `1e-5` tolerance on randomised fixtures.
- [ ] Byzantine-robustness assertion: `k` bad clients don't steer
  trimmed mean further from the honest mean than FedAvg at the same
  setup.
- [ ] Bench rows added at `small`, `medium`, `large` with pure-NumPy
  reference rows at `medium` + `large`. No speedup claim in
  `docs/benchmarks.md` unless the measured Rust run is faster than the
  measured Python run at the same tier.
- [ ] `docs/strategies.md` updated with math, breakdown point, and
  "when to reach for it."
- [ ] No new proxy/mock measurements. Measurement-named fields are
  real or NaN.

## Up next (non-binding)

Once this ships:

1. **Bulyan** — composes `MultiKrum` + `TrimmedMean`. Both kernels
   then exist; Bulyan is a thin orchestration layer plus the
   coordinate-wise trim already landed here.
2. **Buffer-protocol numpy return path** — touches every kernel; best
   done after Bulyan so the robust suite is stable.
3. **CodSpeed + crowd-scale bench tier (50–100 clients)** — needed to
   resolve trimmed mean / Krum / Bulyan speedups under a tight noise
   floor.
