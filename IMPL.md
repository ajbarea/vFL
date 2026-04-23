# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Last shipped

**Bulyan aggregator** (2026-04-23). `Strategy::Bulyan { f, m }` composes
Multi-Krum selection with coordinate-wise trimmed mean over the `m = n - 2f`
survivors. Zero new algorithmic math: refactored `krum_select` to expose
`krum_select_indices` so Bulyan and Multi-Krum share the Phase-1 kernel;
Phase 2 feeds the subset into the existing trimmed-mean kernel with `k = f`.
Validates Bulyan's `n >= 4f + 3` breakdown bound. Python `Bulyan` dataclass
slots into the existing Strategy union; `_core.Strategy.bulyan(f, m=None)`
matches the multi_krum binding shape. Numpy oracle in
`tests/strategy_reference.py` composes `multi_krum_reference` +
`trimmed_mean_reference` so Hypothesis-driven parity tests cover both
phases end-to-end. Bench rows at all three tiers added to
`docs/benchmarks.md` (1.54 s at `large`; MultiKrum + TrimmedMean minus the
`n → m` subset discount).

Commit refs TBD — populated after `/aj-auto-commit` merges.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **`rand` 0.8 → 0.9 bump** — deferred from the pyo3 PR to isolate
   attribution. Touches `gaussian_noise` and the Dirichlet partitioner;
   mechanical.
2. **`[torch-cpu]` CI extra** — promotes convergence coverage from
   nightly to per-PR; see [ROADMAP → CI](ROADMAP.md#ci).
3. **Geometric median / RFA** — Weiszfeld's algorithm, ~50% Byzantine
   breakdown. Natural next robust aggregator now that Bulyan ships; see
   [ROADMAP → Aggregation strategies](ROADMAP.md#aggregation-strategies).
4. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
