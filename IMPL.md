# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Last shipped

**numpy buffer-protocol return path** (2026-04-22). Commits
`bbeb705` (bench probe), `349c698` (numpy return types), `e4c5ade`
(pyo3/numpy 0.21 → 0.23). See [ROADMAP → Completed](ROADMAP.md#completed)
for the dated summary and `docs/benchmarks.md` for the measured before/after.

One DoD item deferred: **no `CHANGELOG.md` entry for the `.append()`-on-weight-value
break under `0.1.0-alpha`**, because no `CHANGELOG.md` file exists in this repo
yet. Creating one is a repo-wide convention decision, not a per-PR decision —
flag for the next session if breaking-change tracking starts to matter.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **Bulyan** — composes `MultiKrum` + `TrimmedMean`. Both kernels ship.
   Thin orchestration layer; see
   [ROADMAP → Aggregation strategies](ROADMAP.md#aggregation-strategies).
2. **`rand` 0.8 → 0.9 bump** — deferred from the pyo3 PR to isolate
   attribution. Touches `gaussian_noise` and the Dirichlet partitioner;
   mechanical.
3. **`[torch-cpu]` CI extra** — promotes convergence coverage from
   nightly to per-PR; see [ROADMAP → CI](ROADMAP.md#ci).
4. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
