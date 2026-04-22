# vFL — Dependency Rationale

Why each dep exists, what it's load-bearing for, and when to reconsider. The
`pyproject.toml` is the source of truth for *what*; this file is the source of
truth for *why*.

Shared toolchain-pin rationale (pytest 9.0.3 floor, ruff/ty floors,
`requires-python = ">=3.11,<3.14"`) lives with the
[aj-sisters](../.claude/skill-context.md) drift-detection skill — those are
cross-repo policy, not vFL-specific decisions.

---

## Runtime dependencies

### `prefect>=3.0`

Workflow orchestration for FL rounds (`velocity.flows`) and user-facing
`@flow` / `@task` entry points (`velocity.server`). Used for retries,
structured logging via `get_run_logger`, and the user-extension pattern
documented in `VelocityServer`.

- **Load-bearing:** `velocity.flows` imports `prefect` at module scope — it
  is a hard runtime dep, not optional.
- **Reconsider when:** users ask for a non-Prefect path (Airflow, Dagster,
  bare asyncio). At that point, split into a `prefect` extra and make
  `velocity.flows` a conditional import.

### `pydantic>=2.0`

All `RunSpec`, `AttackSpec`, `RunResult`, `SweepResult` models
(`velocity.sweep`) plus strategy coercion in `velocity.strategy`. v2 is the
current major; v1 is end-of-life.

- **Load-bearing:** TOML sweep configs round-trip through pydantic
  validation; schema drift would fail loudly on load, which is what we want.
- **Reconsider when:** never — this is the right tool for this job.

### `typer>=0.12`

`velocity` CLI entry point (`velocity.cli`). Typer wraps Click with type
hints; `BadParameter` is used for clean error surfacing to the shell.

- **Load-bearing:** the published `velocity` script in `[project.scripts]`
  imports `typer`.
- **Reconsider when:** the CLI grows past ~10 subcommands or needs plugin
  discovery — at that scale, plain Click or a custom dispatch may read
  better than Typer's type-hint magic.

---

## Optional extras

Extras let users install what they need without paying for the rest. The
baseline `uv sync` installs the runtime deps above; extras gate the heavier
or domain-specific stacks.

### `[hf]` — Hugging Face datasets / models

```
transformers>=4.40
peft>=0.10
datasets>=3.0,<4.0
safetensors>=0.4
```

Needed by the example scripts (`mnist_fedavg`, `cifar10_fedavg_dirichlet`,
`mnist_fedprox_dirichlet`, `mnist_multikrum_vs_byzantine`) and
`velocity.datasets.load_dataset`. The `datasets>=3.0,<4.0` upper bound is
deliberate: `datasets` 2.x depends on pyarrow/pandas APIs that break under
modern transitive pins, and 3.x is the current stable schema baseline.
4.x would be a breaking upgrade worth reviewing explicitly.

### `[torch]` — PyTorch + torchvision

```
torch>=2.0
torchvision>=0.15
```

Needed by `velocity.training` (the SGD-based local trainer) and every
example that touches real models. Kept out of the baseline because torch is
~1.5 GB installed and most downstream use cases (algorithm research,
convergence tests against stubs) don't need it.

### `[agent]` — MCP server surface

```
fastmcp>=2.0
```

Exposes vFL tools over the Model Context Protocol via `velocity.mcp_app`.
FastMCP is tracked at `>=2.0` since its 2.x API is stable enough to float
a floor. If a UI surface lands (Prefab or otherwise), it goes in its own
`[ui]` extra rather than back into this one — see ROADMAP.

### `[all]` — Convenience bundle

```
velocity-fl[hf,torch]
```

Meta-extra for "install the experimentation bits". Deliberately excludes
`[agent]` — the MCP surface is experimental and pre-1.0 on both FastMCP
and Prefab, and shipping it under `[all]` would pull alpha deps into
mainline users' environments.

---

## Dev dependencies

### Test stack

- **`pytest>=9.0.3`** — matches phalanx-fl and kourai-khryseai floors. 9.x
  dropped Python 3.8 support (now a non-issue with our 3.11 floor) and
  ships assertion-rewrite fixes that 8.x lacked.
- **`pytest-asyncio>=0.23`** — `asyncio_mode = "auto"` in `[tool.pytest]`;
  required by any async MCP/Prefect test helpers.
- **`pytest-benchmark>=4.0`** — `tests/bench/` macro benches (opt-in via
  explicit path; the default `pytest` skips them via `norecursedirs`).
- **`pytest-cov>=5.0`** — coverage instrumentation; read by Codecov in CI.
- **`hypothesis>=6.100`** — property-based tests in
  `tests/test_aggregation_properties.py` — exactly where hypothesis earns
  its keep: algebraic invariants that must hold for any valid input, with
  automatic counterexample minimization.

### Build + lint stack

- **`maturin>=1.5,<2.0`** — builds the Rust `_core` extension (see
  `[tool.maturin]`). Upper-bounded because maturin 2.x will likely be a
  breaking upgrade for the build-backend contract.
- **`ruff>=0.9`** — combined linter + formatter. `target-version = "py311"`
  means ruff lints for 3.11+ idioms (PEP 604 unions, `datetime.UTC`,
  `itertools.pairwise`). The lint rule set (`E F W I N UP B SIM RUF`) is
  intentionally broad — we want ruff to catch modernization opportunities,
  not just style nits.
- **`ty>=0.0.25,<0.1`** — Astral's type checker. Pre-1.0; every 0.0.x
  release can land new diagnostics. The `<0.1` ceiling is defensive:
  when Astral cuts 0.1, we review the new lint set deliberately rather
  than absorb it via a silent resolution. `[tool.ty.analysis]
  allowed-unresolved-imports` explicitly whitelists optional-extras
  imports (torch, datasets) so baseline `uv sync` doesn't drown the
  type check in missing-module noise.

### Docs

- **`zensical`** — the docs-site generator. Unversioned floor because the
  site config lives in `zensical.toml` (versioned separately). Same tooling
  lives in phalanx-fl and kourai-khryseai — aj-sisters audits for drift.

### Agent dev

- **`fastmcp>=2.0`** — duplicated from the `[agent]` extra so developers
  have the MCP server available during `uv sync` without needing to pass
  `--extra agent`. If this becomes the pattern for more extras, consider
  a `dev-agent` group instead.

---

## Rust crate dependencies (`vfl-core`)

Defined in `vfl-core/Cargo.toml`, not `pyproject.toml`.

- **`pyo3 = "0.21"`** — Python bindings for the Rust aggregation kernel
  (`velocity._core`). 0.21 predates the current 0.22/0.23 line; not
  blocking, but an upgrade pass would be cheap.
- **`serde = "1"` + `serde_json = "1"`** — serialization for the few
  Python↔Rust boundary types that travel as JSON rather than through
  PyO3's type coercion. Stable and unavoidable.
- **`rand = "0.8"` + `rand_distr = "0.4"`** — used by the Dirichlet
  partitioner and any noise-injection strategies. `rand` 0.9 is out; the
  upgrade requires a small API migration in the partitioner.
- *(safetensors was removed here pending the checkpoint-I/O feature
  that would actually use it; see ROADMAP.)*

### Rust dev deps

- **`approx = "0.5"`** — float-comparison assertions in the kernel's unit
  tests.
- **`divan = "0.1"`** — benchmark harness for `benches/aggregate.rs`.
  Chosen over `criterion` for faster compilation and cleaner output.
- **`proptest = "1"`** — property-based testing on the Rust side (the
  Python `hypothesis` tests already cover most invariants end-to-end, but
  `proptest` catches bugs in the kernel before they surface at the FFI
  boundary).

---

## Open questions / known weights

These are the deps worth revisiting on the next housekeeping pass.

- **`pyo3 = "0.21"`** — two minor versions behind current (0.23 line).
- **`rand = "0.8"`** — one minor version behind current (0.9 API has
  small but breaking changes in the partitioner's call sites).
- **Prefect as a hard runtime dep** — if a user asks for non-Prefect
  orchestration, `velocity.flows` becomes a conditional import and
  `prefect` moves to an extra.
- **`[agent]` extra scope** — if a UI surface lands, split into
  `[mcp]` + `[ui]` rather than re-conflating them.

See the "Dependency hygiene" section of
[ROADMAP.md](ROADMAP.md#dependency-hygiene) for the current
decision log on these.
