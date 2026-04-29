# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Last shipped

**Geometric Median (RFA)** (2026-04-25). `Strategy::GeometricMedian { eps, max_iter }`
shipped as the eighth aggregation strategy via Weiszfeld iteration
(Pillutla et al., IEEE TSP 2022). Sample-weighted aggregation with
~50% Byzantine breakdown (tolerates up to ⌊(n-1)/2⌋ malicious clients)
without explicit thresholding. Rust kernel in `vfl-core`, PyO3 binding
+ Python dataclass in `velocity.strategy`, Hypothesis oracle in
`tests/strategy_reference.py`. Docs propagated across `cli.md`,
`api.md`, `strategies.md`, `configuration.md` (PR #14 ship, PR #16 docs).

Commit refs: `25c789c` (feat), `71c69ba` (docs).

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **`rand` 0.8 → 0.9 bump** — deferred from the pyo3 PR to isolate
   attribution. Touches `gaussian_noise` and the Dirichlet partitioner;
   mechanical.
2. **`[torch-cpu]` CI extra** — promotes convergence coverage from
   nightly to per-PR; see [ROADMAP → CI](ROADMAP.md#ci).
3. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
