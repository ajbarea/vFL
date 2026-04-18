# Convergence

Benchmarks measure how fast the aggregation kernel runs. **This page
measures whether the framework actually does federated learning** — i.e.,
whether a real model trained across real non-IID clients via the
VelocityFL orchestrator converges to a useful classifier.

Performance numbers (`docs/benchmarks.md`) and convergence numbers (this
page) are kept separate on purpose. A fast aggregator that doesn't
converge is worthless; a slow one that does is at least correct. The
benchmarks defend the *speed* claim; this page defends the *FL works at
all* claim.

## What we measure

A run "converges" if all of the following hold across rounds:

- **Test loss decreases** monotonically (with slack — FedAvg on non-IID
  data is not a contraction, single-round wiggles are allowed).
- **Test accuracy improves** from random-guess baseline to a useful
  level for the task.
- **`RoundSummary.global_loss`** matches the loss the caller actually
  measured on the test set — the Rust core stores what the caller passes
  in via `reported_loss=`, never a proxy.

The Rust orchestrator does not see models or data, so it cannot invent a
loss. `RoundSummary.global_loss` is `NaN` unless the caller passes a
real value to `run_round(..., reported_loss=...)`.

## Reproducing

The hermetic convergence proof runs on every CI build:

```bash
uv run pytest tests/test_convergence.py -v
```

The MNIST demo requires the `[torch]` extra and downloads MNIST on first
run (~12 MB into `./data/`):

```bash
uv pip install 'velocity-fl[torch]'
uv run maturin develop --release
uv run python examples/mnist_fedavg.py
```

## Results

### Hermetic — Gaussian-blobs, 4 clients, non-IID, 8 rounds

Synthetic 2D Gaussian-blobs (4 classes, well-separated). Sharded
non-IID across 4 clients (each client sees ~1 class). Tiny MLP
(2→16→4). Runs in ~14s on CPU, no network. Lives at
`tests/test_convergence.py`.

The test asserts:

- `loss[final] < loss[initial] * 0.5`
- No round regresses by more than 25% over the previous round.
- Final test accuracy ≥ 0.85.
- Final accuracy strictly greater than first-round accuracy.

This is the regression guard — if a future change to aggregation,
weight serialisation, or the PyO3 boundary breaks real FL, this test
fails before the merge.

### MNIST FedAvg — 5 clients, non-IID, 10 rounds

Snapshot: 2026-04-18, AMD Ryzen 5 3600X, single CPU thread per round
of local training. MNIST sharded so each client only sees 2 of the 10
digit classes (McMahan et al. 2017 setup). Small MLP
(784→128→64→10, ~109K params), batch size 64, lr=0.01,
1 local epoch per round.

| round | pre-loss | post-loss | post-acc | sec/round |
|------:|---------:|----------:|---------:|----------:|
| init  |   2.3072 |        —  |   0.110  |       —   |
|     1 |   2.3072 |   1.8213  |   0.415  |    11.58  |
|     2 |   1.8213 |   1.1628  |   0.668  |    11.53  |
|     3 |   1.1628 |   0.8906  |   0.707  |    11.31  |
|     4 |   0.8906 |   0.6394  |   0.836  |    12.94  |
|     5 |   0.6394 |   0.6160  |   0.787  |    11.59  |
|     6 |   0.6160 |   0.4760  |   0.861  |    11.28  |
|     7 |   0.4760 |   0.4697  |   0.842  |    12.39  |
|     8 |   0.4697 |   0.3819  |   0.883  |    11.69  |
|     9 |   0.3819 |   0.3912  |   0.874  |    10.89  |
|    10 |   0.3912 |   0.3177  |   0.908  |    11.16  |

**11.0% → 90.8% test accuracy** in 10 rounds. Loss falls from 2.31 to
0.32. The single-round dips at rounds 5, 7, and 9 are expected on
non-IID data — FedAvg is not a contraction when client objectives
disagree.

Per-round wall time is dominated by client-side local training in
PyTorch, **not** by aggregation. The Rust orchestrator's contribution
to round time is ~10 ms at this model size (see `docs/benchmarks.md`
medium tier). If you want to know why a round takes 11 s, the answer is
"five clients each running an MNIST epoch on CPU," not "aggregation."

## What this proves and what it doesn't

**Proves:** the end-to-end pipeline (PyTorch local training →
state-dict flatten → PyO3 boundary → Rust FedAvg → state-dict
unflatten → eval) is correct. A real model genuinely trains.

**Does not prove:**

- Convergence at scale (50–100+ clients). The hermetic test uses 4,
  the MNIST demo uses 5. Non-IID FedAvg behaviour shifts with client
  count.
- Convergence on hard tasks (CIFAR-10, language modelling, etc.). MNIST
  is a forgiving baseline — a non-converging implementation would fail
  here, but converging here is necessary, not sufficient.
- Robustness with attacks active. Existing attack simulations (model
  poisoning, Sybil, Gaussian noise, label flipping) live in
  `vfl-core/src/security.rs` and have unit tests but no
  end-to-end-with-real-training convergence assertion yet.

## What's still mocked

The aggregation path and round loop are now real end-to-end. The
following surfaces still contain stand-ins that should be replaced
before performance claims are made about anything other than the
aggregation kernel:

- **Attacks** (`vfl-core/src/security.rs`) — `simulate_*` functions
  *do* mutate weights, but only as illustrative perturbations rather
  than implementations of published attack literature (e.g. real
  inner-product manipulation, real projected-gradient attacks). The
  unit tests assert "weights changed," not "attack achieves its
  documented effect."
- **Datasets** — `ExperimentConfig.dataset` is a free-form string. No
  real Hugging Face dataset loader is wired through the Rust core or
  the orchestrator config yet.
- **Storage** — `ExperimentConfig.storage` is a URI string with no
  resolver behind it.
- **Strategies** — `FedAvg`, `FedProx` (server-side identical to
  FedAvg by construction; the proximal term is client-side), and
  `FedMedian` are real. Other published strategies (Krum, Trimmed
  Mean, Bulyan, FedYogi, FedAdam) are not implemented.
- **Partitioning** — only McMahan-style shard partition lives in
  `velocity.training.non_iid_shard_partition`. Dirichlet-α partition
  and quantity-skew partition are missing.

These are tracked as a roadmap item: replace each with a real,
plug-and-play implementation, then add a convergence test that
exercises it from this page.

## Follow-ups

- [ ] CIFAR-10 demo (real ConvNet client model, longer training horizon)
- [ ] Convergence-under-attack: assert FedMedian survives a poisoned
  client where FedAvg is corrupted, measured rather than asserted by hand
- [ ] Dirichlet-α partition + a sweep over α to chart accuracy vs
  client heterogeneity
- [ ] Real Hugging Face dataset wiring so `dataset=` actually loads
  something
- [ ] Crowd-scale convergence (50+ clients) — paired with the
  crowd-scale benchmark tier so speed and convergence numbers come from
  the same setup
