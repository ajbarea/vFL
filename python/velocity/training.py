"""Real federated training utilities for VelocityFL.

The Rust core (`velocity._core.Orchestrator`) only sees flat layer weights —
it does not know about models, datasets, or losses. This module provides the
PyTorch-side glue that turns "I have N clients with local data" into a real
FedAvg run against the Rust aggregator, with honest per-round evaluation.

Torch is an optional dependency (``velocity-fl[torch]``); importing this
module without it raises a clear error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.training requires PyTorch. Install with: pip install 'velocity-fl[torch]'"
    ) from exc


__all__ = [
    "ClientData",
    "evaluate",
    "layers_to_state_dict",
    "local_train",
    "state_dict_to_layers",
]


@dataclass
class ClientData:
    """One client's view of the federation: their local training loader and sample count."""

    loader: DataLoader
    num_samples: int


def state_dict_to_layers(state_dict: dict[str, Tensor]) -> dict[str, list[float]]:
    """Flatten a torch ``state_dict`` into the Rust core's ``{name: [f32]}`` shape."""
    return {name: tensor.detach().cpu().flatten().tolist() for name, tensor in state_dict.items()}


def layers_to_state_dict(
    layers: dict[str, list[float]], reference: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Inverse of ``state_dict_to_layers``: reshape flat weights back to tensor shapes."""
    return {
        name: torch.tensor(layers[name], dtype=ref.dtype).reshape(ref.shape)
        for name, ref in reference.items()
    }


def layer_shapes(state_dict: dict[str, Tensor]) -> dict[str, int]:
    """Flat element count per layer — what `Orchestrator.__init__(layer_shapes=...)` wants."""
    return {name: int(tensor.numel()) for name, tensor in state_dict.items()}


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    loss_fn: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Run local SGD on one client's data, returning the trained model in-place."""
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    model.to(device)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` of ``model`` on ``loader``.

    Accuracy assumes a classification head; for regression-style models pass a
    custom ``loss_fn`` and ignore the second return value.
    """
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_samples += int(y.numel())
    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss / total_samples, total_correct / total_samples


def aggregated_loss(per_client: Iterable[tuple[float, int]]) -> float:
    """Sample-weighted mean of per-client losses — useful when a server-side
    eval loader isn't available and you only have client-reported losses."""
    total_loss = 0.0
    total_samples = 0
    for loss, n in per_client:
        total_loss += loss * n
        total_samples += n
    return total_loss / total_samples if total_samples > 0 else float("nan")
