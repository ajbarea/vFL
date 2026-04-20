# Architecture

VelocityFL splits federated learning into three tightly-scoped layers: a Rust core that owns every hot loop, a PyO3 bridge that makes those types feel native in Python, and a researcher-facing Python package that handles orchestration and ergonomics.

## Layer map

| Layer | Crate / Package | Responsibility |
|---|---|---|
| **Rust core** | `vfl-core` | Aggregation math, attack kernels, round state. No Python awareness. |
| **PyO3 bindings** | `vfl-core` (with `pyo3` feature) | Exports Rust types as `velocity._core.*`. Zero-copy where possible. |
| **Python package** | `python/velocity/` | `VelocityServer`, `Strategy`, CLI, Prefect flows, pure-Python fallback. |
| **CLI** | `python/velocity/cli.py` | Typer app — thin adapter over `VelocityServer`. |
| **Docs** | `docs/` + `zensical.toml` | This site, deployed via GitHub Actions. |

## Round lifecycle

```text
  ┌──────────────────────────────────────────────────────────────────┐
  │  Python                                                          │
  │                                                                  │
  │   VelocityServer.run(min_clients, rounds)                        │
  │      │                                                           │
  │      ├─► build _core.Orchestrator  ─────────┐                    │
  │      │                                      │                    │
  │      ├─► register pending attacks           │                    │
  │      │                                      ▼                    │
  │      └─► for r in rounds:                ┌─────────────────────┐ │
  │            generate ClientUpdate[]   ──► │  Rust hot path      │ │
  │            orchestrator.run_round()  ◄── │  • aggregate        │ │
  │            collect RoundSummary          │  • apply attacks    │ │
  │                                          │  • update globals   │ │
  │                                          └─────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
```

Every arrow that crosses into the Rust box is a PyO3 call. Every arrow inside that box is compiled code.

## Why Rust for the hot path

- **Aggregation is O(clients × parameters)** — in Python this loop dominates a round. In Rust it's a tight inner loop over `Vec<f32>`.
- **Byzantine-robust strategies** (e.g. `FedMedian`) need coordinate-wise sorts. SIMD-friendly in Rust, painfully slow in NumPy when layer shapes are awkward.
- **Attack simulation** must be deterministic and fast — the whole point of Byzantine testing is running many variants.

## Why Python for the outer loop

- **Hugging Face, PEFT, PyTorch** — the research stack lives here.
- **Prefect** — first-class flow/task wrappers give you observability without custom logging glue.
- **Typer** — CLI scaffolding that researchers actually read.

## Pure-Python fallback

The package imports `velocity._core` lazily. When the native module isn't present (fresh clone without `maturin develop`, CI jobs that only build docs, etc.), `VelocityServer` transparently uses `_PurePythonOrchestrator`. It's numerically simpler and slower, but every test still passes. This is what lets the docs build on GitHub-hosted runners without a Rust toolchain.

## Module map

```text
vFL/
├── vfl-core/               # Rust crate
│   ├── src/
│   │   ├── lib.rs          # PyO3 module definition
│   │   ├── strategy.rs     # FedAvg / FedProx / FedMedian / TrimmedMean / Krum / MultiKrum
│   │   ├── orchestrator.rs # round state + attack dispatch
│   │   └── security.rs     # attack types + simulations (model_poisoning, sybil_nodes, gaussian_noise, label_flipping)
│   └── Cargo.toml
├── python/velocity/
│   ├── __init__.py         # re-exports VelocityServer, Strategy
│   ├── server.py           # orchestrator wrapper + fallback
│   ├── strategy.py         # Python-side enum
│   ├── attacks.py          # AttackResult dataclass, VALID_ATTACKS
│   ├── flows.py            # Prefect flow wrappers
│   └── cli.py              # Typer app
├── tests/                  # pytest suite
├── docs/                   # this site
├── scripts/dev.py          # logged fix-then-check runner (Makefile delegates here)
└── .claude/skills/         # Claude Code slash-command library (see Skills page)
```

## Next

- [Configuration](configuration.md) — fields on `VelocityServer`.
- [API Reference](api.md) — public Python and Rust surface.
- [Skills](skills.md) — the `.claude/skills/` library and how contributors use it.
