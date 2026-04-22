# IMPL: current session — numpy buffer-protocol return path

Session-by-session checklist for what's actively in flight. When this
PR ships, its contents get replaced by the next session's plan and a
dated one-liner lands in [ROADMAP → Completed](ROADMAP.md#completed).

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). This file is
the "what are we actually building this PR" scratchpad.

## Why this PR

The aggregator API has stabilised (FedAvg + FedProx + FedMedian + Krum +
MultiKrum + TrimmedMean all shipped). Bulyan is the obvious next
aggregator, but the perf lever has been sitting right next to us: PyFloat
marshaling in the return-type path.

Phase 1 measurement (2026-04-22, loaded system — Roblox running on box):

| tier | `global_weights()` only | full round (`run_round + readout`) | implied `run_round` alone | **getter share** |
| --- | --- | --- | --- | --- |
| tiny (~1K params) | 11.3 µs | 15.9 µs | ~4.6 µs | 71% |
| medium (~1M) | 35.3 ms | 39.3 ms | ~4.0 ms | 90% |
| large (~10M) | 425 ms | 459 ms | ~34 ms | **93%** |

At `large` tier, 93% of the realistic per-round cost is PyFloat
marshaling in `.global_weights()` — not the Rust aggregation kernel.
The "97× vs Python" speedup in `docs/benchmarks.md` is run_round-only;
the realistic-round speedup a user experiences is closer to 11×. Fixing
the getter should recover that delta.

## Decisions

- **Output-only migration.** Four return sites change; inputs stay
  `HashMap<String, Vec<f32>>`. ROADMAP's reasoning (the input side is
  already zero-copy on the no-attack path) still holds — retouching the
  input path would be churn with no measurable benefit.
- **Two-commit split for risk isolation.** Commit 1 adds `numpy = "0.21"`
  (matching current pyo3) and changes the four return types to
  `HashMap<String, Py<PyArray1<f32>>>`. Commit 2 bumps `pyo3 0.21 → 0.23`
  (and `numpy 0.21 → 0.23` to match). Advantage: the perf win lands in
  commit 1; the pyo3 API migration (Bound<'py, T> lifetime, _bound
  constructors) is a cosmetic follow-up with no perf impact. If pyo3
  0.23 has unexpected issues, commit 1 still shipped.
- **`rand` 0.8 → 0.9 bump deferred.** ROADMAP says "cheap when done with
  the pyo3 bump above" but stacking two migrations on one PR risks
  debugging attribution. Separate session.
- **Breaking for `0.1.0-alpha`.** Callers that use `.append()` on layer
  values break (ndarray has no `.append()`); `np.concatenate` or
  preallocated writes replace that idiom. Noted in CHANGELOG on cut.
  Iteration, indexing, and scalar math survive — most call sites are
  untouched.
- **Measurement discipline.** Phase 1 baseline recorded in this file
  (table above) and in a new `tests/bench/test_round_speed.py` block
  (`test_rust_global_weights`, `test_rust_run_round_plus_readout`). Phase
  2 must re-measure both, record the delta in `docs/benchmarks.md`, and
  withdraw the migration if the getter cost doesn't drop meaningfully
  at `medium` + `large`. Same posture the Trimmed Mean PR took: "no
  speedup claim without a same-workload reference."

## Scope

### New code (Phase 1 — shipped this session)

- **`tests/bench/test_round_speed.py`** — two new tests:
  - `test_rust_global_weights` — `Orchestrator.global_weights()` in
    isolation after one `run_round` to populate the store. Tier-parameterised.
  - `test_rust_run_round_plus_readout` — realistic FL-round cost
    (`run_round + global_weights()` together). Compare directly to
    `test_python_aggregate` for the honest apples-to-apples speedup.

### Edited code (Phase 2 — this PR's actual work)

#### Commit 1 — numpy return types (pyo3 stays at 0.21)

- **`vfl-core/Cargo.toml`** — add `numpy = "0.21"` (matches current
  pyo3). No pyo3 bump.
- **`vfl-core/src/lib.rs`** — four return sites change from
  `HashMap<String, Vec<f32>>` to `HashMap<String, Py<PyArray1<f32>>>`,
  built with `Vec::into_pyarray(py).into()`:
  - L105: `PyClientUpdate::weights` getter
  - L276: `PyOrchestrator::global_weights`
  - L320: free `aggregate` function return
  - L339: `apply_gaussian_noise` tuple first element
- **`pyproject.toml`** — promote `numpy` from `[torch]` extra to
  `[project].dependencies`. Pin floor matches whatever the `[torch]`
  extra currently uses.
- **`python/velocity/_core.pyi`** — four return-type sites:
  `dict[str, list[float]]` → `dict[str, numpy.typing.NDArray[numpy.float32]]`.
  Add `import numpy` + `import numpy.typing` at top.
- **`python/velocity/server.py`** — verify `_PurePythonOrchestrator`
  returns the same shape (ndarray dict) for transparent swap. Likely
  needs `{k: np.asarray(v, dtype=np.float32) for k, v in agg.items()}`
  on the exit edge.

#### Commit 2 — pyo3 0.21 → 0.23 bump (cosmetic, no perf impact)

- **`vfl-core/Cargo.toml`** — `pyo3 = "0.23"`, `numpy = "0.23"`.
- **`vfl-core/src/lib.rs`** — API migration:
  - `Bound<'py, T>` lifetime pattern across every `#[pyfunction]` and
    `#[pymethods]` site (6 blocks by grep).
  - `PyDict::new_bound` / `PyList::new_bound` / `PyTuple::new_bound`
    where present (the `_bound` transitional names are gone in 0.23 —
    plain `PyDict::new(py)` is now the Bound form).
  - `Py<T>::as_ref(py)` → `.bind(py)` where it appears.
  - `PyCell` → `Bound<PyClass>`.

### Tests

- **Existing unit suite** — any test asserting weight-dict equality with
  a literal list (`assert weights[l] == [0.1, 0.2]`) fails because
  ndarray `==` is element-wise. Grep for this pattern pre-migration;
  migrate to `np.testing.assert_allclose` or `.tolist()` on the LHS.
- **`make test-unit`** — must stay green (current baseline: 2322 passed).
- **`cargo test`** — must stay green.
- **`tests/bench/` on migrated code** — the Phase 1 tests are the Phase
  2 verification; re-run, compare to baseline above.

### Docs

- **`docs/benchmarks.md`** — add a "Realistic round cost (run_round +
  readout)" subsection with the Phase 1 baseline and Phase 2 delta. Keep
  the existing `run_round`-alone table untouched for continuity. Note
  the WSL2-load caveat on the Phase 1 numbers; mark Phase 2 numbers as
  "idle system" only after re-running on idle.
- **`python/velocity/_core.pyi`** docstring hint, if present, should
  reflect the new return shape.
- **CHANGELOG** — note the breaking `.append()` removal under the
  `0.1.0-alpha` unreleased-changes section (or create one if it doesn't
  exist).

## Out of scope

- **Input-side migration.** Inputs (`ClientUpdate.__init__`,
  `set_global_weights`, `gaussian_noise` first arg) stay
  `HashMap<String, Vec<f32>>`. Zero-copy fast path already exists on
  the no-attack code path; no measured win from changing.
- **`rand` 0.8 → 0.9.** Separate session. Touches `gaussian_noise` and
  the Dirichlet partitioner; attribution is cleaner when isolated.
- **Bulyan.** Obvious next aggregator. Separate session after this
  perf PR ships and the bench table is updated.
- **CodSpeed CI + crowd-scale bench tier.** Both still queued under
  ROADMAP → Performance. This PR only extends the existing WSL2-measured
  bench harness; the noise-floor upgrade is its own work.
- **`[torch-cpu]` CI extra for convergence coverage.** Separate PR in
  the **CI** ROADMAP section.

## Definition of done

- [ ] `tests/bench/test_round_speed.py` Phase 1 additions committed and
  landed on main with the baseline captured in this file (above).
- [ ] Commit 1 — numpy return types:
  - [ ] `vfl-core/Cargo.toml` adds `numpy = "0.21"`.
  - [ ] Four return sites in `lib.rs` return `Py<PyArray1<f32>>` via
    `.into_pyarray(py)`.
  - [ ] `_core.pyi` reflects new return types; static type checks pass
    (`uv run ty check`).
  - [ ] `_PurePythonOrchestrator` returns ndarray dict too.
  - [ ] `make test-unit` stays green (2322 passed).
  - [ ] Phase 1 bench tests re-measured; `global_weights()` at `large`
    drops from ~425 ms to < 5 ms (target: O(layers) = 16 layers, each a
    numpy wrapper).
- [ ] Commit 2 — pyo3 0.23 bump:
  - [ ] `Cargo.toml` bumps pyo3 and numpy to 0.23.
  - [ ] `lib.rs` migrated to Bound<'py, T> idiom.
  - [ ] `cargo test` green, `make test-unit` green.
  - [ ] No perf regression vs commit 1's numbers.
- [ ] `docs/benchmarks.md` updated with before/after table; WSL2 noise
  caveat preserved; git SHAs of before and after measurements cited.
- [ ] CHANGELOG note for `0.1.0-alpha` on the `.append()` break.
- [ ] No new proxy/mock measurements. Measurement-named fields are real
  or NaN (same rule as Trimmed Mean PR).

## Up next (non-binding)

Once this ships:

1. **Bulyan** — composes `MultiKrum` + `TrimmedMean`. Both kernels now
   exist; thin orchestration layer with one new `Strategy::Bulyan`
   variant.
2. **`rand` 0.8 → 0.9 bump** — deferred from this PR; touches
   `gaussian_noise` and Dirichlet partitioner. Mechanical.
3. **`[torch-cpu]` CI extra** — promotes convergence coverage from
   nightly to per-PR; ROADMAP → CI section.
4. **CodSpeed + crowd-scale bench tier** — the noise-floor upgrade that
   makes single-digit-percent regression detection meaningful.
