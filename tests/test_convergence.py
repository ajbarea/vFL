"""End-to-end FedAvg convergence proof — hermetic, no network, CPU-only.

Builds a tiny separable classification task, partitions it non-IID across
four clients, then runs 8 real FedAvg rounds through the Rust orchestrator
with PyTorch on the client side. Asserts the run actually *converges*:
loss decreases monotonically (with slack) and final test accuracy clears
85%. This is the test that proves vFL does federated learning, not just
the math inside one round of it.

Two partitioning regimes are exercised — McMahan-style sharding
(``velocity.partition.shard``) and Dirichlet-alpha (``velocity.partition.dirichlet``)
— so the partitioner module is covered end-to-end alongside the aggregator.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402  — gated on torch import above
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402
from velocity import _core  # noqa: E402
from velocity.partition import dirichlet, shard  # noqa: E402
from velocity.training import (  # noqa: E402
    ClientData,
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

# ---------------------------------------------------------------------------
# Synthetic data — four well-separated 2D Gaussian blobs, one per class
# ---------------------------------------------------------------------------


class GaussianBlobs(Dataset):
    """4-class 2D Gaussian-blobs dataset with deterministic seeding.

    Trivially separable so a 2-layer MLP can solve it; non-IID partitioning
    is what makes the FL setup non-trivial.
    """

    CENTERS = torch.tensor([[3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])

    def __init__(self, samples_per_class: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        xs, ys = [], []
        for class_idx, center in enumerate(self.CENTERS):
            noise = torch.randn(samples_per_class, 2, generator=gen) * 0.6
            xs.append(center + noise)
            ys.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))
        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        perm = torch.randperm(len(self.y), generator=gen)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.targets = self.y

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 4))


# ---------------------------------------------------------------------------
# Shared FedAvg loop — run the rounds, return per-round (loss, accuracy)
# ---------------------------------------------------------------------------


def _run_fedavg(
    train_set: Dataset,
    test_set: Dataset,
    client_indices: Sequence[Sequence[int]],
    *,
    rounds: int,
    local_epochs: int,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Drive `rounds` of FedAvg on the given client partition; return (losses, accuracies)."""
    num_clients = len(client_indices)
    clients: list[ClientData] = [
        ClientData(
            loader=DataLoader(Subset(train_set, list(idx)), batch_size=32, shuffle=True),
            num_samples=len(idx),
        )
        for idx in client_indices
    ]
    test_loader = DataLoader(test_set, batch_size=128)

    template = make_model()
    template_state = template.state_dict()

    orch = _core.Orchestrator(
        model_id="hermetic-mlp",
        dataset="gaussian-blobs-4class",
        strategy=_core.Strategy.fed_avg(),
        storage="memory://",
        min_clients=num_clients,
        rounds=rounds,
        layer_shapes=layer_shapes(template_state),
    )
    # Seed the orchestrator with a real PyTorch init, not zeros — zeros at the
    # bias of a Linear-ReLU stack collapses every input to the same logits
    # and there is no useful gradient on round 1.
    orch.set_global_weights(state_dict_to_layers(template_state))

    losses: list[float] = []
    accuracies: list[float] = []

    for _ in range(rounds):
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        client_updates = []
        for client in clients:
            local_model = make_model()
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_train(local_model, client.loader, epochs=local_epochs, lr=lr)
            client_updates.append(
                _core.ClientUpdate(
                    num_samples=client.num_samples,
                    weights=state_dict_to_layers(local_model.state_dict()),
                )
            )

        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, test_loader)

        summary = orch.run_round(client_updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, test_loader)

        losses.append(post_loss)
        accuracies.append(post_acc)
        # Rust core must round-trip the caller-reported loss verbatim — no proxy.
        assert summary.global_loss == pytest.approx(pre_loss, rel=1e-6, abs=1e-6)

    return losses, accuracies


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_fedavg_converges_on_shard_partition() -> None:
    torch.manual_seed(0)

    train_set = GaussianBlobs(samples_per_class=400, seed=1)
    test_set = GaussianBlobs(samples_per_class=200, seed=2)

    labels = [int(t) for t in train_set.targets]
    client_indices = shard(labels, num_clients=4, shards_per_client=2, seed=42)

    losses, accuracies = _run_fedavg(train_set, test_set, client_indices, rounds=8, local_epochs=2)

    _assert_converges(losses, accuracies)


def test_fedavg_converges_on_dirichlet_partition() -> None:
    torch.manual_seed(0)

    train_set = GaussianBlobs(samples_per_class=400, seed=1)
    test_set = GaussianBlobs(samples_per_class=200, seed=2)

    labels = [int(t) for t in train_set.targets]
    # alpha=0.3 gives heavy but non-degenerate label skew across 4 clients —
    # some clients see mostly one or two classes, but every class is
    # represented across the federation.
    client_indices = dirichlet(labels, num_clients=4, alpha=0.3, seed=42)

    losses, accuracies = _run_fedavg(train_set, test_set, client_indices, rounds=8, local_epochs=2)

    _assert_converges(losses, accuracies)


# ---------------------------------------------------------------------------
# Convergence assertions
# ---------------------------------------------------------------------------


def _assert_converges(losses: Sequence[float], accuracies: Sequence[float]) -> None:
    _assert_loss_trend_down(losses)
    assert accuracies[-1] >= 0.85, (
        f"final test accuracy {accuracies[-1]:.3f} below 0.85 threshold; "
        f"trajectory: {[round(a, 3) for a in accuracies]}"
    )
    # `>=` rather than `>` — on this easy task, some partitions saturate at 1.0
    # after round one and simply stay there, which is valid convergence, not
    # stagnation. The halved-loss check above is the real "it kept learning"
    # guard.
    assert accuracies[-1] >= accuracies[0], (
        f"accuracy regressed over rounds: {accuracies[0]:.3f} -> {accuracies[-1]:.3f}"
    )


def _assert_loss_trend_down(values: Sequence[float]) -> None:
    """Loss should generally fall round-over-round.

    Allow modest non-monotonic wiggles (FedAvg on non-IID data is not a
    contraction) but require the overall trend: final < first by a clear
    margin, and no single round may regress more than 25%.
    """
    assert values[-1] < values[0] * 0.5, (
        f"loss did not roughly halve: {values[0]:.4f} -> {values[-1]:.4f}; full: {values}"
    )
    for prev, curr in pairwise(values):
        assert curr <= prev * 1.25, (
            f"loss regressed sharply: {prev:.4f} -> {curr:.4f}; full trajectory: {values}"
        )
