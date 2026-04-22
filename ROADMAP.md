# vFL Roadmap

The living long-horizon plan for VelocityFL. Each section names work still
ahead, with enough context that anyone — including us in three weeks — can
pick it up cold. Items we've decided against don't belong here.

When an item ships, its scope block is removed and a dated one-liner lands
in [Completed](#completed) at the bottom. This file stays about what's
next; the log at the bottom preserves the trail.

Session-by-session execution (the "what are we doing this PR") lives in
[IMPL.md](IMPL.md), not here.

## Agent stack

- **Real (non-mock) training tool, confirmation-gated** — `run_demo` in
  `python/velocity/mcp_app.py` still calls the mock `VelocityServer.run`;
  add a sibling tool that triggers a real round and requires explicit user
  confirmation before launch.
- **Prefab `PrefabApp` return types** — MCP tools currently return plain
  `dict` / `list[dict]`. Migrate to typed Prefab returns so the Claude UI
  can render them natively. Keep separate from the memory/caching PR.
- **Memory compaction** — `recent_runs.md` grows unbounded. Needs a
  bounded-retention strategy (last-N runs, or size-capped with rollup into
  a narrative summary).

A2A specialist agents (convergence auditor, robustness auditor, etc.)
are scoped under [Live experiment leaderboard](#live-experiment-leaderboard)
rather than duplicated here — they're the analysis layer over the
leaderboard data, not standalone infra.

## Deploy

- **Horizon deploy** — Prefect Horizon hosted-deploy path for vFL flows.

## CI

- **CPU-only torch extra for fast-CI convergence coverage** — `tests.yml`
  runs `uv sync` (no extras), which prunes torch and skips
  `tests/test_convergence.py` entirely. Nightly (`nightly.yml`) installs
  `[torch]` and runs `examples/mnist_fedavg.py` plus the hermetic
  convergence tests, so regressions *are* caught — just on a daily
  cadence, not per-PR. Fix: add a `[torch-cpu]` extra that pins the
  ~150–200 MB CPU wheels from PyTorch's `+cpu` index (separate from the
  full `[torch]` extra, which pulls ~2.5 GB of CUDA runtime). Wire
  `tests.yml` to `uv sync --extra torch-cpu` so the Gaussian-blobs
  convergence proof runs on every push. Tradeoff: fast CI gets slower by
  the torch install time (one-time with uv's cache), in exchange for
  end-to-end FL coverage on every PR instead of every night.

## Docs

- **Claude Desktop wiring guide** — `docs/configuration.md` documents
  `VFL_USER_ID` but not the full `mcpServers` JSON block or the `fastmcp
  run` commands (stdio for Claude Desktop, `--transport http --port 8765`
  for local inspection). Users can't wire the MCP server up without it.

## Aggregation strategies

Worth porting from phalanx-fl (`intellifl/simulation_strategies/`) — that codebase
wraps Flower and has production-grade implementations we can reimplement as
pure Rust kernels. Current vFL has `FedAvg`, `FedProx`, `FedMedian` only.

These kernels are load-bearing for the perf story, not just Byzantine
coverage. FedAvg is O(n) in clients; Krum is O(n²); Bulyan stacks Krum
with coordinate-wise trimmed mean; Trimmed Mean is a k-partial sort per
coordinate. The robust aggregators are algorithmically heavier than
FedAvg — the Rust-vs-Python gap grows with them. Measure each after
implementation before quoting speedups.

- **Bulyan** (El Mhamdi et al., 2018) — Multi-Krum → coordinate-wise trimmed
  mean. Breakdown point `(n - 2f - 3)`. Strongest distance-based defense in
  phalanx; gold-standard for Byzantine-robust evaluation. Now implementable
  entirely in-tree: Multi-Krum + Trimmed Mean kernels both ship.
- **Geometric median / RFA** — Weiszfeld's algorithm, ~50% Byzantine breakdown
  without explicit thresholding. Natural fit for Rust (iterative, arithmetic-heavy).
- **ArKrum** (arXiv:2505.17226) — parameter-free Krum that estimates `f` via
  median filtering + SSE segmentation. Removes the "you must know `f` in
  advance" constraint. Good v2 once Krum ships.

## Client-removal defenses

A distinct axis from the pure-stateless aggregators above: rather than
picking a robust combiner each round, these strategies maintain
per-client score state across rounds and *permanently drop* clients
that cross a threshold. They compose over any aggregator. phalanx-fl
has working Flower-based implementations under
`intellifl/simulation_strategies/*_removal_strategy.py`; our port keeps
the algorithms and moves the hot path into Rust.

Rust angle is real here, not handwaving: per-client state is a
fixed-shape struct (score, EMA, last-round distance, removal flag),
the round work is an O(n²·d) distance matrix plus an O(n) score
update, and everything vectorises. phalanx's implementations call
`sklearn.KMeans` on every round for outlier detection — that's a
prime Python cost to replace with a direct threshold on the
reconstructed score distribution.

- **PID-based removal** (phalanx `pid_based_removal_strategy.py`,
  arXiv:2402.12780) — treats per-client deviation from federation
  centroid as a control signal; `kp·distance + ki·integral +
  kd·derivative` drives a removal threshold set at
  `mean + num_std_dev · std`. Rust side owns the per-client history
  ring and the scalar PID update; Python just passes the gains.
- **Trust / reputation** (phalanx `trust_based_removal_strategy.py`) —
  beta-weighted exponential smoothing of per-client distances, with
  two-phase removal (first round drops the single worst, later
  rounds batch-drop below `trust_threshold`). Straightforward Rust
  EMA per client; no sklearn dependency.
- **RFA-based removal** — geometric-median aggregation paired with
  single-worst-deviation removal per round. Uses Weiszfeld's
  algorithm (listed under Aggregation strategies above) plus the
  removal loop. Shares the geometric-median kernel.
- **Krum / Multi-Krum / Trimmed-mean removal** — compose the
  aggregator-of-the-same-name with a removal step keyed on the
  Krum score (or coordinate-wise trimmed-mean distance). Only worth
  porting after the base aggregators land; the removal layer is
  ~30 lines on top once the kernel exists.
- **Termination policies** (phalanx `termination_policies.py`) —
  `GRACEFUL` / `STRICT` / `ADAPTIVE` behaviour when removal thins the
  federation below `min_fit_clients`. Orchestration, not a Rust
  kernel; Python-side enum + handler is fine. Only meaningful once
  removal strategies exist.

Out of scope here: phalanx's Flower-coupled `flwr.server.strategy`
base class — we reimplement the algorithms against our own PyO3
boundary rather than copying the wrapper.

## Attacks

Current vFL registry (`security::AttackType`): `ModelPoisoning`, `SybilNodes`,
`GaussianNoise`, `LabelFlipping`. `LabelFlipping` is a stub — it records an
attack result but doesn't actually mutate client data. Items below come from
`phalanx-fl/intellifl/attack_utils/{poisoning,weight_poisoning}.py`, each with
paper citations already documented in-place there.

- **Targeted label flipping** — source-class → target-class with `flip_ratio`.
  Models real adversary goals (e.g. "stop sign → speed limit"), more useful
  than bijective permutation. Should replace the current label-flipping stub.
- **Backdoor trigger / BadNets** (Gu et al., 2017) — pixel-pattern trigger
  stamped onto a fraction of images + relabel to target class. The canonical
  FL backdoor attack; phalanx has square/cross patterns with auto-contrast.
- **Boosted scaling** (Baruch et al., NeurIPS 2019 — "A Little Is Enough") —
  scale update by `n_total / n_malicious` to exactly cancel FedAvg dilution.
  Drop-in upgrade over the current naive constant-factor model poisoning.
- **Inner-product manipulation** (Xie et al., 2020) — aggregation-aware,
  L2-bounded perturbation that defeats Krum/Multi-Krum/Bulyan. Needed to
  stress-test the robust aggregators once they land.
- **Alternating-min / PGD poisoning** (Fang et al., USENIX 2020 + Bagdasaryan
  et al., AISTATS 2020 + Bhagoji et al., ICML 2019) — optimization-based
  attack via projected gradient descent in weight space, FedAvg-aware trust
  region. Research-grade; only worth it once the robust-aggregation suite is
  built out and we want to demonstrate we can break it.
- **Byzantine perturbation with norm-clip** (Sun et al., 2019) — Gaussian
  perturbation with optional L2-norm clipping for defense evasion. Small
  delta over our existing `GaussianNoise`, useful when benchmarking against
  norm-based defenses.

Out of scope: phalanx's `token_replacement` (tokenizer-dependent, LLM-specific)
and its deprecated `gradient_scaling` (superseded by `boosted_scaling`).

### Attack forensics

- **Streaming weight-snapshot stats** — phalanx captures pre/post-attack
  weight histograms + summary stats per (client, round) in
  `attack_utils/weight_snapshots.py`. Useful: it's what the
  `robustness_auditor` agent on the leaderboard would actually
  consume. Weak: it recomputes min/max/sum/sum² in three separate
  passes per array, allocates a fresh numpy array every snapshot, and
  writes JSON to disk per client per round (100 clients · 100 rounds
  = 10k tiny files). Rust port: single-pass Welford online stats +
  streaming histogram (P²-estimator or fixed-bin) per client, all
  owned by a fixed-size ring buffer on the orchestrator. Write a
  single columnar snapshot file per run, not per client-round. Feeds
  directly into the leaderboard store under Live experiment
  leaderboard.

## Datasets

Current loader (`velocity.datasets.load_federated`) handles any HF
image-classification dataset with standard `image`/`label` columns
plus column aliases — MNIST and CIFAR-10 are live in
`docs/convergence.md`. Everything below extends breadth without
rewriting the loader. phalanx-fl has working versions of each under
`intellifl/dataset_loaders/image_transformers/` and its text loaders.

- **CIFAR-100 / MedMNIST 2D** — already free via the existing loader;
  the only missing piece is per-dataset normalisation constants in a
  small lookup table (phalanx has these in its image-transformer
  files). No perf story; a one-line test matrix extension.
- **FEMNIST natural partition** — FEMNIST ships with a writer-id field
  that defines the federated partition (each writer ≈ one client).
  Needs `velocity.partition.natural(labels, group_ids)` — an O(n)
  groupby pass, pure Python is fine. Adds the canonical non-IID FL
  benchmark dataset; currently the first thing missing to make the
  leaderboard honest across "real" FL benchmarks.
- **Text-classification path** — AG News, MedQuAD, generic HF text
  datasets (phalanx has both). Requires a tokenisation step and an
  embedding layer in the reference model, not just a new transform.
  Non-trivial scope — gated on whether we want text attacks on the
  leaderboard at all (see Attacks → out-of-scope). If we do, the
  tokeniser call happens once at load time and caches; nothing about
  it is a Rust hot-path candidate.
- **Reference model zoo** (phalanx `network_models/`) — small CNN for
  FEMNIST, wider CNN for CIFAR-100, minimal transformer for text.
  Pure PyTorch; lives next to the examples, not in the vFL wheel.
  Value is leaderboard reproducibility, not perf — a run fingerprint
  that names "the FEMNIST reference CNN v1" is worth more than 200
  bespoke models on the board.

Out of scope: phalanx's medical datasets (lung photos, FLAIR, ITS) —
licensing + dataset-size overhead isn't justified until someone asks
for medical-FL benchmarks specifically.

## Performance

- **Buffer-protocol / numpy return path** — the remaining PyO3 cost is
  on the *output* side: `HashMap<String, Vec<f32>>` → `dict[str,
  list[float]]` allocates one `PyFloat` per parameter. At the `large`
  tier (10M params) this dominates; the input side is already zero-copy
  on the no-attack path. Fix: return `numpy.ndarray` via the buffer
  protocol to share the underlying f32 buffer with zero copies. Named
  as the next lever in `docs/benchmarks.md:98-105`.
  - **Call sites** (`vfl-core/src/lib.rs`): `Orchestrator.global_weights`
    (L240), `ClientUpdate.weights` getter (L79), free `aggregate` (L281),
    `gaussian_noise` (L299). `run_round` itself returns a `PyRoundSummary`
    struct — the marshaling cost lives in the *getters* and the direct
    `aggregate` / `gaussian_noise` returns, not in `run_round`'s return.
  - **Rust dep**: add `numpy = "0.21"` (paired with pyo3 0.21) to
    `vfl-core/Cargo.toml`. `PyArray1::from_slice` / `IntoPyArray` builds
    the ndarray without cloning the `Vec<f32>`.
  - **Python dep**: promote `numpy` from the `torch` extra to
    `[project].dependencies` in `pyproject.toml`. Update
    `python/velocity/_core.pyi` return types: `dict[str, list[float]]`
    → `dict[str, numpy.typing.NDArray[np.float32]]` (4 sites: L20, L52,
    L56, L58).
  - **Breaking for 0.1.0-alpha**: iteration and indexing still work,
    but `.append()` on layer values breaks — callers switch to
    `np.concatenate` or preallocated writes. Note in CHANGELOG on cut.
  - **Measurement**: re-run `tests/bench/test_round_speed.py` at `large`
    tier. Current Rust FedAvg `large` is 49.3 ms through the Python
    surface (`docs/benchmarks.md:83`) vs 74.2 ms raw divan (L65) — the
    apparent *speedup* from raw→Python is WSL2 noise, not real; the
    true boundary cost hides in the getter call that follows
    `run_round`. Target: the subsequent `.global_weights` read becomes
    O(layers) not O(params). Update `docs/benchmarks.md` snapshot after.
- **FedMedian SIMD quickselect or histogram median** — FedMedian still
  runs ~12× FedAvg at large tier. Coordinate-wise `select_nth_unstable_by`
  is branchy and doesn't vectorise well. Worth revisiting only when
  Byzantine-robust aggregation sits on a hot path (not the default).
- **CodSpeed CI integration** — bare-metal macro runners with
  PR-comment perf tracking, so single-digit-percent regressions become
  visible instead of being absorbed by WSL2 noise.
- **Crowd-scale bench tier (50–100 clients)** — current benches use 10
  clients at every shape tier. Above 50 clients, Python's per-object
  overhead grows and Krum's O(n²) kernel blows up — the regime where
  the Rust lever is largest and is currently not measured. Depends on
  CodSpeed for a noise floor tight enough to see the effect. Listed as
  a follow-up in `docs/benchmarks.md:130-132`.

## Live experiment leaderboard

Longer-horizon: turn every run into comparable data. Today each round
emits a `RoundSummary` that lands in SQLite via `velocity.db` and then
gets forgotten. The goal is to make those runs rankable along several
axes so researchers landing on the docs site can answer "what strategy
should I reach for on FEMNIST under label-flipping?" without reading
the code. Each bullet below is scoped to stand on its own; the whole
stack only becomes interesting once the aggregation and attack suites
below are wider than they are today.

- **Experiment ingestion + config fingerprint** — extend `velocity.db`
  so every run stores a stable fingerprint:
  `(dataset, partition, partition_params, strategy, strategy_params,
  attack, attack_params, seed, vfl_version)`. This is what makes two
  runs comparable. Depends on dataset + attack configs being fully
  serialisable (they mostly already are via the existing dataclasses).
- **Per-axis ranking engine** — independent leaderboards, not a
  single composite score. Axes: final-round accuracy, rounds-to-target
  accuracy, wall-clock at fixed bench tier, Byzantine robustness
  delta (accuracy drop under attack vs no-attack baseline on the same
  data + strategy), sample efficiency (accuracy per total client
  sample). Per-axis because any weighted combination buries the
  tradeoffs that make the comparison interesting.
- **Pareto frontier per (dataset × attack) pair** — rather than a
  single "winner," surface the non-dominated set across
  accuracy/robustness/wall-clock. This is the honest answer to "what
  should I use" — there usually isn't one.
- **Theoretical complexity labels, not rankings** — tag aggregators
  with their asymptotic cost (FedAvg: O(n·d); Krum: O(n²·d);
  Bulyan: O(n²·d + n·d·log n)). Static lookup, surfaced next to each
  strategy's measured row. Explicitly *not* a ranking input —
  asymptotic class doesn't predict wall-clock inside the regimes we
  measure.
- **Cross-config normalisation** — the hard part. Can a FEMNIST run
  be compared to a CIFAR-10 run? Only on normalised axes
  (accuracy-relative-to-centralised-ceiling, not raw accuracy; rounds
  as a fraction of IID-FedAvg rounds-to-ceiling, not absolute). The
  ceilings themselves need to be measured and stored per dataset as
  reference runs. Don't ship cross-dataset ranking until this is
  solid — it's the fastest way to publish misleading numbers.
- **A2A specialist agents over the store** — Claude-backed (per the
  Claude-only stack decision), each surfaced as an MCP tool that
  queries the leaderboard store rather than invents numbers.
  Candidates: `convergence_auditor` (why did run X diverge — class
  imbalance from the partition? LR too high?), `robustness_auditor`
  (how much did attack Y drop accuracy vs the no-attack baseline on
  matched configs?), `complexity_labeller` (static asymptotic lookup,
  above), `hyperparameter_sage` (given a target config, returns the
  top-k α / μ / f values observed in matched runs, with sample
  count + variance, and flags when sample size is too low to
  recommend).
- **Sage guard-rails** — any sage answer must quote sample size and
  variance. "α=0.3 was top-3 over 47 runs on MNIST+shard+no-attack,
  IQR ±0.008 final accuracy" is useful; "use α=0.3" is cargo cult.
  Hard fail the tool call when the matched-run count is below a
  threshold (start: 10) rather than returning a confident-looking
  guess.
- **Public Zensical leaderboard page** — auto-rendered from the
  store. Researchers pick dataset + attack, see the Pareto frontier
  per axis, click into the fingerprint for repro. Depends on the
  Zensical site (`aj-docs-site` skill) and a stable store schema.
- **Prerequisites** — this section only becomes worth shipping once:
  (a) the aggregation suite includes at least Krum, Multi-Krum,
  Bulyan, Trimmed Mean (so there's something to rank);
  (b) the attack suite beyond `GaussianNoise` + `ModelPoisoning` is
  real (boosted scaling, targeted label flipping, inner-product —
  all under Attacks); (c) dataset breadth beyond MNIST + CIFAR-10
  (FEMNIST and Shakespeare are the canonical FL-benchmark additions
  once the HF loader handles text).
- **Out of scope for the first cut** — LLM-specific attacks
  (token_replacement et al. remain out of scope per the Attacks
  section). Leaderboards over arbitrary tasks (the first cut is
  vision-classification only — extending to NLP / tabular is a
  separate slice once the store schema has earned its keep).

## Dependency hygiene

Captured from the ECOSYSTEM.md audit (2026-04-22). The "obvious wins"
(prefab-ui removal, safetensors Rust-dep removal, ty `<0.1` cap) shipped
with that audit; the items below need a real decision before they move.

- **`pyo3` 0.21 → 0.23 bump** — two minors behind current. API changes
  between 0.21 and 0.23 touch `PyCell`, `PyTuple::new_bound`, and the
  `Bound<'py, T>` lifetime pattern; mostly mechanical but ripples
  through every `#[pyfunction]` and `#[pymethods]` site in
  `vfl-core/src/lib.rs`. Batch with the numpy return-path work in
  **Performance** (which also wants 0.21 → 0.23) to avoid doing the
  migration twice.
- **`rand` 0.8 → 0.9 bump** — one minor behind. 0.9 renames
  `thread_rng` → `rng`, removes `SliceRandom::choose_multiple` in
  favour of `IteratorRandom`, and tightens trait bounds on
  `Distribution`. Touches the Dirichlet partitioner and the
  `gaussian_noise` path in the kernel. Cheap when done with the pyo3
  bump above; fiddly on its own.
- **`[agent]` extra split — `[mcp]` + `[ui]`** — today `[agent]` holds
  only `fastmcp`. When a UI surface (Prefab or successor) lands, it
  goes in a separate `[ui]` extra rather than rejoining `[agent]` —
  the two surfaces have different upgrade cadences and a user can
  legitimately want one without the other. `[agent]` stays as a
  meta-extra that pulls `[mcp,ui]` together, matching how `[all]`
  works.
- **Prefect as a hard runtime dep — revisit trigger** — today
  `velocity.flows` imports `prefect` at module scope, making it a
  hard baseline dep (~50 MB installed). Move to a `[prefect]` extra
  with a conditional import in `velocity.flows` if (and only if) a
  user asks for a non-Prefect orchestration path. Don't do it
  pre-emptively — the extras-cascade adds real cognitive cost.
- **Checkpoint I/O (unblocks re-adding `safetensors`)** — the Rust
  `safetensors = "0.4"` dep was removed because nothing imports it.
  When `velocity.checkpoint` lands (fast-secure weight serialisation
  for warm-start and fine-tune resume), re-add the Rust dep with the
  feature that actually uses it.

## Completed

Dated one-liners for shipped roadmap-scale work. Most recent first. The
commit history and `docs/benchmarks.md` / `docs/convergence.md` are the
authoritative record; this log is the human index into them.

- **2026-04-22** — Zero-copy numpy buffer-protocol return path across the
  PyO3 boundary. `ClientUpdate.weights`, `Orchestrator.global_weights`,
  free `aggregate`, and `apply_gaussian_noise` now return
  `dict[str, numpy.ndarray[float32]]` sharing the Rust `Vec<f32>` buffer.
  At `large` tier (10M params, 16 layers): getter dropped 425 ms → 6.6 ms
  (64×); realistic round cost 459 ms → 56.3 ms (8×); realistic-round
  speedup vs Python FedAvg 11× → 91× (now matches the advertised
  `run_round`-only 97×). Bumped pyo3 + numpy 0.21 → 0.23 alongside.
  Before/after tables in `docs/benchmarks.md`.
- **2026-04-20** — Trimmed Mean coordinate-wise Byzantine-robust aggregator
  (`Strategy::TrimmedMean { k }`) shipped with bench + Python-reference
  rows in `docs/benchmarks.md`. Dimension-independent k-partial sort per
  coordinate; cheaper than FedMedian, simpler than Bulyan. Unblocks Bulyan
  (which composes Multi-Krum with coordinate-wise trimmed mean).
- **2026-04-20** — Krum + Multi-Krum Byzantine-robust aggregators
  (`Strategy::Krum { f }`, `Strategy::MultiKrum { f, m }`) shipped together
  with shared `krum_select` Rust kernel, dataclass-strategy migration
  (`FedAvg | FedProx | FedMedian | Krum | MultiKrum`), and
  `RoundSummary.selected_client_ids`. Bench + Python-reference rows in
  `docs/benchmarks.md`.
- **2026-04-20** — Real Hugging Face dataset loader
  (`velocity.datasets.load_federated`) with column aliasing, canonical
  train/test/validation split preference, and partition dispatch.
  MNIST + CIFAR-10 convergence demos measured in
  `docs/convergence.md`. Rust `ExperimentConfig.dataset` kept as a
  record-keeping string; Python is the real entry point.
- **2026-04-18** — Dirichlet-α partitioner (`velocity.partition.dirichlet`)
  shipped alongside IID and McMahan-shard, all framework-independent
  under a single `velocity.partition` module. Convergence test
  coverage under both shard and Dirichlet regimes.
- **2026-04-18** — Pure-Python FedAvg baseline at the `large` tier of
  `tests/bench/` so future Rust speedup claims have a same-workload
  reference. Recorded in `docs/benchmarks.md`.
- **2026-04-18** — Real end-to-end FedAvg training loop through the
  PyO3 boundary: client-side PyTorch local training, Rust aggregation,
  honest per-round eval. MNIST demo + hermetic Gaussian-blobs
  convergence test, both gated on nightly CI.
