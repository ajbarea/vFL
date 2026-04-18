"""Real FedAvg convergence demo on MNIST.

Five clients, each holding a non-IID slice of MNIST (≈two digit classes
per client, McMahan-style shard partition). Ten rounds of Federated
Averaging through the Rust orchestrator. Every round, the server evaluates
the aggregated global model on the held-out test set and reports real loss
+ real accuracy.

This is the proof that VelocityFL does federated learning, not just the
math inside one round of it.

Run::

    uv pip install 'velocity-fl[torch]'
    uv run maturin develop --release
    uv run python examples/mnist_fedavg.py

Expect test accuracy to climb from ~10% (random) to >90% over 10 rounds.
First run downloads MNIST (~12MB) into ``./data/``.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from velocity import _core
from velocity.training import (
    ClientData,
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    non_iid_shard_partition,
    state_dict_to_layers,
)

NUM_CLIENTS = 5
SHARDS_PER_CLIENT = 2  # 5 * 2 = 10 shards, one per digit class
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Nightly convergence floor: if this run doesn't clear it, something regressed.
# 0.85 leaves slack above the sub-convergent ~0.80 we saw in early iterations
# and below the ~0.92 we've observed on passing runs.
MIN_FINAL_ACC = 0.85


def make_model() -> nn.Module:
    """Small MLP — 784 -> 128 -> 64 -> 10. ~109K params."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def main() -> None:
    torch.manual_seed(SEED)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    DATA_ROOT.mkdir(exist_ok=True)
    train_set = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    client_subsets = non_iid_shard_partition(
        train_set, num_clients=NUM_CLIENTS, shards_per_client=SHARDS_PER_CLIENT, seed=SEED
    )
    clients = [
        ClientData(loader=DataLoader(s, batch_size=BATCH_SIZE, shuffle=True), num_samples=len(s))
        for s in client_subsets
    ]
    test_loader = DataLoader(test_set, batch_size=512)

    template = make_model()
    template_state = template.state_dict()

    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64",
        dataset="mnist",
        strategy=_core.Strategy.fed_avg(),
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    print(f"VelocityFL MNIST FedAvg demo — {NUM_CLIENTS} clients, non-IID, {ROUNDS} rounds")
    print(f"Per-client sample counts: {[c.num_samples for c in clients]}")
    print(f"{'round':>5} | {'pre-loss':>9} | {'post-loss':>9} | {'post-acc':>8} | {'sec':>6}")
    print("-" * 56)

    initial_eval = make_model()
    initial_eval.load_state_dict(template_state)
    init_loss, init_acc = evaluate(initial_eval, test_loader)
    print(f"{'init':>5} | {init_loss:>9.4f} | {'-':>9} | {init_acc:>8.3f} | {'-':>6}")

    for round_idx in range(1, ROUNDS + 1):
        round_start = time.perf_counter()

        global_state = layers_to_state_dict(orch.global_weights(), template_state)

        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, test_loader)

        client_updates = []
        for client in clients:
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_train(local_model, client.loader, epochs=LOCAL_EPOCHS, lr=LR)
            client_updates.append(
                _core.ClientUpdate(
                    num_samples=client.num_samples,
                    weights=state_dict_to_layers(local_model.state_dict()),
                )
            )

        orch.run_round(client_updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, test_loader)
        elapsed = time.perf_counter() - round_start

        print(
            f"{round_idx:>5} | {pre_loss:>9.4f} | {post_loss:>9.4f} | "
            f"{post_acc:>8.3f} | {elapsed:>6.2f}"
        )

    print()
    print(
        f"Initial accuracy: {init_acc:.3f}   ->   Final accuracy: {post_acc:.3f}   "
        f"(loss {init_loss:.4f} -> {post_loss:.4f})"
    )

    if post_acc < MIN_FINAL_ACC:
        raise SystemExit(
            f"FAIL: final accuracy {post_acc:.3f} below nightly floor {MIN_FINAL_ACC:.2f}"
        )
    print(f"PASS: final accuracy {post_acc:.3f} >= {MIN_FINAL_ACC:.2f}")


if __name__ == "__main__":
    main()
