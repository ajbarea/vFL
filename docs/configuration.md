# Configuration

Every knob on `VelocityServer` — what it means, what it defaults to, and when you need to change it.

## `VelocityServer` constructor

```python
VelocityServer(
    model_id: str,
    dataset: str,
    strategy: Strategy | None = None,   # defaults to FedAvg()
    storage: str = "local://checkpoints",
    layer_shapes: dict[str, int] | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | *required* | Hugging Face model identifier (e.g. `meta-llama/Llama-3-8B`) or a local path. |
| `dataset` | `str` | *required* | Dataset name or local path. Any HF Hub slug works. |
| `strategy` | [`Strategy`](strategies.md) | `FedAvg()` | Aggregation algorithm (a `FedAvg`, `FedProx`, `FedMedian`, `TrimmedMean`, `Krum`, `MultiKrum`, `Bulyan`, or `GeometricMedian` instance). |
| `storage` | `str` | `"local://checkpoints"` | Checkpoint storage URI. Supports `local://`, `hf-xet://`, and any fsspec-style scheme your environment can resolve. |
| `layer_shapes` | `dict[str, int] \| None` | small demo net | Maps layer name → parameter count. Must match the model being trained for real experiments. |

### Default `layer_shapes`

When you don't pass `layer_shapes`, VelocityFL uses a tiny two-layer demo net — enough to exercise aggregation end-to-end in tests:

```python
{
    "fc1.weight": 128,
    "fc1.bias":   16,
    "fc2.weight": 256,
    "fc2.bias":   10,
}
```

For a real model, pass the actual parameter-count map. A quick way to derive it from a PyTorch model:

```python
layer_shapes = {name: p.numel() for name, p in model.named_parameters()}
```

## Runtime fields (passed to `.run()`)

```python
server.run(min_clients: int = 1, rounds: int = 1)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `min_clients` | `int ≥ 1` | `1` | Minimum clients required per round. |
| `rounds` | `int ≥ 1` | `1` | Number of federated rounds to execute. |

## Attack fields (passed to `.simulate_attack()`)

```python
server.simulate_attack(
    attack_type: str,
    *,
    intensity: float = 0.1,
    count: int = 1,
)
```

| Field | Used by | Type | Default | Description |
|---|---|---|---|---|
| `attack_type` | all | `str` | *required* | `model_poisoning` \| `sybil_nodes` \| `gaussian_noise`. |
| `intensity` | `model_poisoning`, `gaussian_noise` | `float ≥ 0` | `0.1` | Magnitude of the perturbation. |
| `count` | `sybil_nodes` | `int ≥ 1` | `1` | Number of sybil clients to inject. |

> **Call order** — `simulate_attack` can be called **before or after** `run()`. When called before, attacks are queued and applied to the first round that executes.

> **Data-pipeline attacks** — for `label_flipping` (and other label/feature
> corruption primitives) see [`velocity.data_attacks`](attacks.md#data-pipeline-attacks-velocitydata_attacks).
> Those attacks live in the Python data layer because the Rust core never
> sees raw labels.

## Strategy parameters

Each strategy is a frozen dataclass; parameters live on the instance. Pass the instance itself to `VelocityServer(strategy=…)`.

| Strategy | Parameter | Default | Description |
|---|---|---|---|
| `FedAvg` | *(none)* | — | Sample-weighted mean (McMahan et al., AISTATS 2017). |
| `FedProx` | `mu: float` | `0.01` | Proximal-term coefficient applied client-side in `local_train`; aggregation kernel is FedAvg (Li et al., MLSys 2020). |
| `FedMedian` | *(none)* | — | Coordinate-wise median (Yin et al., ICML 2018). |
| `TrimmedMean` | `k: int` | *required* | Drop `k` smallest + `k` largest per coordinate, mean the rest. Requires `2k < n` (Yin et al., ICML 2018). |
| `Krum` | `f: int` | *required* | Pick the single closest client by Krum score. Requires `n ≥ 2f + 3` (Blanchard et al., NeurIPS 2017). |
| `MultiKrum` | `f: int`, `m: int \| None` | `m = n − f` when `None` | Average the `m` lowest-scoring Krum updates. `1 ≤ m ≤ n − f` (El Mhamdi et al., ICML 2018). |
| `Bulyan` | `f: int`, `m: int \| None` | `m = n − 2f` when `None` | Multi-Krum → trimmed-mean composition. Requires `n ≥ 4f + 3` (El Mhamdi et al., ICML 2018). |
| `GeometricMedian` | `eps: float`, `max_iter: int` | `eps=1e-6`, `max_iter=3` | RFA Weiszfeld iteration — sample-weighted, 1/2 breakdown point. Defaults match Pillutla et al., IEEE TSP 2022. |

See [Strategies](strategies.md) for when to use each.

## Environment

| Variable | Effect |
|---|---|
| `VFL_DB_PATH` | Overrides the experiment SQLite path. Default: `./.velocity/experiments.db`. |
| `VFL_MEMORY_DIR` | Overrides the researcher memory root. Default: `~/.velocity/memory/`. |
| `VFL_USER_ID` | Overrides the per-user scope for memory and runs. Default: current shell user. |
| `PREFECT_API_URL` | Honored by Prefect — if set, the flow wrappers report to your Prefect backend. |
| `HF_TOKEN` | Honored by the `huggingface_hub` client for private models / datasets. |
