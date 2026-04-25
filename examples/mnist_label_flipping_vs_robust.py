"""Data-poisoning robustness head-to-head on MNIST.

Ten clients, two of them compromised by bijective label flipping. The two
compromised clients train honestly on their own data — but on the labels
their (adversarial) data pipeline serves them, which are a derangement of
the true labels. Their resulting weight updates point in directions the
honest clients never would, simulating a real-world data-poisoning
adversary as in Tolpegin et al. (ESORICS 2020).

The same ten client updates are fed into two orchestrators over ten rounds:

1. A FedAvg baseline — sample-weighted averaging treats the corrupted
   updates as equally valid and the global model degrades.
2. Multi-Krum with ``f = 2`` — the Krum score should detect the corrupted
   clients (their flat-space distance to the honest cluster is large) and
   exclude them from the aggregate.

The test is the *gap*: Multi-Krum's final accuracy must beat FedAvg's by
at least ``MIN_GAP``, and Multi-Krum must clear ``MIN_MULTIKRUM_ACC`` on
its own. This is the assertion that guards the data-pipeline attack
story (the new honest replacement for the old `LabelFlipping` no-op)
against silent regressions in either ``data_attacks`` or the robust
aggregator.

References:
    Biggio, Nelson, Laskov. *Poisoning Attacks against Support Vector
    Machines*. ICML 2012.
    https://icml.cc/2012/papers/880.pdf
        — Foundational data-poisoning paper; label flipping is its
          simplest realisation.

    Tolpegin, Truex, Gursoy, Liu. *Data Poisoning Attacks Against
    Federated Learning Systems*. ESORICS 2020.
    https://link.springer.com/chapter/10.1007/978-3-030-58951-6_24
        — FL-specific formulation; demonstrates substantial accuracy
          drop under label-flipping with only a small fraction of
          compromised participants.

Run::

    uv pip install 'velocity-fl[hf,torch]'
    uv run maturin develop --release
    uv run python examples/mnist_label_flipping_vs_robust.py
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn
from torchvision import transforms
from velocity import _core
from velocity.data_attacks import make_label_flip_callback
from velocity.datasets import load_federated
from velocity.training import (
    evaluate,
    layer_shapes,
    layers_to_state_dict,
    local_train,
    state_dict_to_layers,
)

NUM_CLIENTS = 10
NUM_COMPROMISED = 2
COMPROMISED_IDS = (0, 1)  # first two clients have flipped labels
NUM_CLASSES = 10
SHARDS_PER_CLIENT = 2
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0
ATTACK_SEED = 137  # separate seed for the label-flip permutation

# Convergence floors. Multi-Krum on this setup has cleared 0.85 in clean
# runs; FedAvg under the same label-flipping attack typically lands in
# 0.55-0.70 depending on which classes were flipped into which. 0.78
# leaves seed-variance slack for Multi-Krum; the 0.10 gap is well below
# the typical ~0.20 delta but tight enough to catch a real regression.
MIN_MULTIKRUM_ACC = 0.78
MIN_GAP = 0.10

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_updates(
    split,
    global_state: dict,
    compromised_ids: tuple[int, ...],
) -> list[_core.ClientUpdate]:
    """Run one round of local training. Compromised clients train against
    a label-flipped view of their own data; honest clients train normally."""
    flip_cb = make_label_flip_callback(num_classes=NUM_CLASSES, seed=ATTACK_SEED)
    updates: list[_core.ClientUpdate] = []
    for client_idx, client in enumerate(split.clients):
        local_model = make_model()
        local_model.load_state_dict(copy.deepcopy(global_state))
        label_attack = flip_cb if client_idx in compromised_ids else None
        local_train(
            local_model,
            client.loader,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            label_attack=label_attack,
        )
        updates.append(
            _core.ClientUpdate(
                num_samples=client.num_samples,
                weights=state_dict_to_layers(local_model.state_dict()),
            )
        )
    return updates


def run_experiment(split, strategy: _core.Strategy, template_state: dict, label: str) -> float:
    """Run `ROUNDS` FL rounds with `strategy`, returning final test accuracy."""
    orch = _core.Orchestrator(
        model_id="mnist-mlp-128-64",
        dataset="ylecun/mnist",
        strategy=strategy,
        storage="memory://",
        min_clients=NUM_CLIENTS,
        rounds=ROUNDS,
        layer_shapes=layer_shapes(template_state),
    )
    orch.set_global_weights(state_dict_to_layers(template_state))

    print(f"\n--- {label} ---")
    print(f"{'round':>5} | {'post-loss':>9} | {'post-acc':>8} | {'sec':>6}")

    for round_idx in range(1, ROUNDS + 1):
        round_start = time.perf_counter()
        global_state = layers_to_state_dict(orch.global_weights(), template_state)
        pre_eval = make_model()
        pre_eval.load_state_dict(global_state)
        pre_loss, _ = evaluate(pre_eval, split.test_loader)

        updates = make_updates(split, global_state, COMPROMISED_IDS)
        orch.run_round(updates, reported_loss=pre_loss)

        post_eval = make_model()
        post_eval.load_state_dict(layers_to_state_dict(orch.global_weights(), template_state))
        post_loss, post_acc = evaluate(post_eval, split.test_loader)
        elapsed = time.perf_counter() - round_start
        print(f"{round_idx:>5} | {post_loss:>9.4f} | {post_acc:>8.3f} | {elapsed:>6.2f}")

    return post_acc


def main() -> None:
    torch.manual_seed(SEED)

    split = load_federated(
        "ylecun/mnist",
        num_clients=NUM_CLIENTS,
        partition="shard",
        shards_per_client=SHARDS_PER_CLIENT,
        batch_size=BATCH_SIZE,
        seed=SEED,
        transform=MNIST_TRANSFORM,
    )

    template_state = make_model().state_dict()

    print(
        f"VelocityFL data-poisoning demo — {NUM_CLIENTS} clients "
        f"({NUM_COMPROMISED} label-flipped), {ROUNDS} rounds"
    )
    print(f"Per-client sample counts: {[c.num_samples for c in split.clients]}")

    fedavg_acc = run_experiment(split, _core.Strategy.fed_avg(), template_state, "FedAvg baseline")
    multikrum_acc = run_experiment(
        split,
        _core.Strategy.multi_krum(NUM_COMPROMISED),
        template_state,
        "Multi-Krum (f=2)",
    )

    gap = multikrum_acc - fedavg_acc
    print()
    print(f"FedAvg final accuracy (label-flip attack):     {fedavg_acc:.3f}")
    print(f"Multi-Krum final accuracy (label-flip attack): {multikrum_acc:.3f}")
    print(f"Gap (MK - FedAvg):                             {gap:+.3f}")

    failures: list[str] = []
    if multikrum_acc < MIN_MULTIKRUM_ACC:
        failures.append(
            f"Multi-Krum accuracy {multikrum_acc:.3f} below floor {MIN_MULTIKRUM_ACC:.2f}"
        )
    if gap < MIN_GAP:
        failures.append(f"defense gap {gap:+.3f} below required margin {MIN_GAP:+.2f}")

    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print(f"PASS: Multi-Krum cleared floor and out-performed FedAvg by {gap:+.3f}")


if __name__ == "__main__":
    main()
