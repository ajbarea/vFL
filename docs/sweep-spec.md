# `velocity sweep` — spec

Parallel strategy × attack matrix runs with auto-comparison. The single feature that
answers "can I compare strategies without waiting 12 hours?" and "can the agent pick
the best strategy?"

## CLI surface

```bash
# Declarative (recommended) — one JSON/TOML file drives the whole matrix
velocity sweep path/to/experiment.toml

# Ad-hoc — quick strategy comparison without writing a config
velocity sweep --strategies FedAvg,FedMedian --rounds 10 --clients 10

# Full matrix — strategies × attacks
velocity sweep --strategies FedAvg,FedMedian --attacks gaussian_noise,model_poisoning \
  --rounds 20 --clients 10

# Tuning
velocity sweep ... --parallel 4     # process pool size; default = min(cpu_count, #runs)
velocity sweep ... --out out/       # output dir; default = out/<ISO-ts>-sweep/
velocity sweep ... --seed 42        # reproducibility; seed per run = seed + run_index
```

## Experiment config (TOML, idiomatic for Python projects)

```toml
# experiment.toml — shared settings + per-run overrides
[shared]
model_id      = "demo/mnist-cnn"
dataset       = "ylecun/mnist"
storage       = "local://checkpoints"
rounds        = 20
min_clients   = 10
seed          = 42

[[runs]]
name      = "baseline-fedavg"
strategy  = "FedAvg"

[[runs]]
name      = "fedmedian-baseline"
strategy  = "FedMedian"

[[runs]]
name      = "fedavg-under-poisoning"
strategy  = "FedAvg"
[[runs.attacks]]
type      = "model_poisoning"
intensity = 0.5

[[runs]]
name      = "fedmedian-under-poisoning"
strategy  = "FedMedian"
[[runs.attacks]]
type      = "model_poisoning"
intensity = 0.5
```

Keep the shape (`[shared]` + `[[runs]]`) close to phalanx-fl's `shared_settings` +
`simulation_strategies` — researchers already know it — but use TOML so it plays
nicely with `pyproject.toml` tooling and isn't a wall of quotes and commas.

## Execution model

1. **Load config** → `list[RunSpec]` (pydantic model). CLI flags and TOML produce
   the same `RunSpec` list; one codepath downstream.
2. **Fan out** via `concurrent.futures.ProcessPoolExecutor`. Each worker owns one
   `VelocityServer` + Rust `Orchestrator`. No shared state, no locks.
3. **Per-run directory** `out/<ts>-sweep/<run_name>/` with:
   - `config.json` — the resolved RunSpec
   - `rounds.csv` — one row per round: `round,num_clients,global_loss,attack_results`
   - `summary.json` — final metrics: total time, final loss, per-round means
4. **Top-level sweep dir** `out/<ts>-sweep/` also contains:
   - `manifest.json` — git commit, uv lockfile hash, host CPU/RAM, package versions
     (adopt phalanx-fl's `MANIFEST.json` pattern — it's the one thing they nailed)
   - `comparison.json` — machine-readable comparison table
   - `comparison.md` — human-readable ranking + per-metric winner

**Why multiprocessing not threading:** each run spins a Rust orchestrator and
eventually a torch model; torch's global state + GIL + CUDA context mean process
isolation is the only safe answer. Ray would give us remote clusters but adds ~8s
startup per worker — flat-out wrong tradeoff for the "fast local sweep" use case.

**Why not asyncio:** rounds are CPU-bound (aggregation + eventual torch training),
not I/O-bound. Asyncio buys nothing here.

## Report format (`comparison.md`)

```markdown
# Sweep: 2026-04-18T09:45:12

4 runs, 2 parallel, total wall 12.3s (serial would be 47.1s — 3.8× speedup)

| Run                        | Strategy   | Attack            | Final loss | Mean loss | Elapsed |
|----------------------------|------------|-------------------|-----------:|----------:|--------:|
| fedmedian-under-poisoning  | FedMedian  | model_poisoning   | 0.082      | 0.134     | 11.9s   |
| baseline-fedavg            | FedAvg     | —                 | 0.091      | 0.142     | 11.2s   |
| fedmedian-baseline         | FedMedian  | —                 | 0.093      | 0.145     | 11.5s   |
| fedavg-under-poisoning     | FedAvg     | model_poisoning   | 0.412      | 0.389     | 11.4s   |

Winner (lowest final loss): **fedmedian-under-poisoning**
Most robust under attack: **FedMedian** (Δ loss 0.011 vs 0.321 for FedAvg)
```

## Modularity / DRY — current state

1. **`parse_strategy`** (`python/velocity/strategy.py`) is the single source
   of truth for coercing strings / dicts / instances into a `Strategy`
   sum-type instance. CLI, TOML, and sweep loader all route through it —
   adding a new strategy means one dataclass + one `ALL_STRATEGIES` entry.
2. **`server.py::_map_strategy`** isinstance-dispatches on each dataclass
   variant. Two lines per new strategy (one `isinstance` check + one Rust
   factory call).
3. **`attacks.py::VALID_ATTACKS`** still duplicates the Rust match arms in
   `orchestrator.rs::register_attack`. The Rust layer should expose a
   `valid_attacks()` free function; Python reads from it. Listed as an open
   item in ROADMAP.
4. **`simulate_attack` kwargs** (intensity/count/fraction) are a union of
   all attacks' parameters. Still pending: switch to
   `simulate_attack(attack_type: str, **params)` once the Rust side exposes
   per-attack parameter schemas.

## Agent integration (follow-up, not MVP)

Once `comparison.json` exists, the MCP agent gets two tools:
- `sweep_run(config_path)` — kick off a sweep, return the sweep dir
- `sweep_compare(sweep_dir)` — read `comparison.json`, return a ranked summary

The agent can then answer "which strategy should I use?" by actually running the
sweep and reading the verdict.

## What lives where

| Component                     | Language | Why                                    |
|-------------------------------|----------|----------------------------------------|
| `RunSpec` / TOML loader       | Python   | pydantic + stdlib `tomllib`; I/O-bound |
| Process pool + fan-out        | Python   | `concurrent.futures`; orchestration    |
| Per-run `VelocityServer`      | Python   | Existing surface; no change            |
| Aggregation kernel            | **Rust** | Already there; hot numeric             |
| Attack simulation             | **Rust** | Already there; hot numeric             |
| CSV / JSON writers            | Python   | stdlib; not hot                        |
| `comparison.md` renderer      | Python   | String formatting; not hot             |
| MANIFEST capture (git, deps)  | Python   | Subprocess + string munging; not hot   |

Nothing in this feature needs to move to Rust. The Rust payoff is already in the
per-round aggregation that each worker calls.

## Out of scope for MVP

- HTML dashboard (phalanx ships one; CLI + markdown is enough for v1)
- Live-queueing (adding runs mid-sweep — phalanx does this; add after MVP if asked)
- Plotting — `rounds.csv` is consumable by pandas/matplotlib; ship later
- Distributed sweep across machines — one machine first

## Success criteria

1. `velocity sweep --strategies FedAvg,FedMedian --rounds 5` runs both concurrently
   and produces `comparison.md` in under 2× single-strategy wall time (ideally ~1×).
2. Adding a new strategy requires touching only `strategy.rs`
   + `strategy.py` + a `_map_strategy` isinstance arm — no CLI or sweep changes.
3. Adding a new attack requires touching only `security.rs` — Python is a thin
   passthrough.
4. `make ci` stays green with the new code.
