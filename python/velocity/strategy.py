"""FL aggregation strategies exposed to Python users."""

from __future__ import annotations

from enum import Enum


class Strategy(str, Enum):
    """Supported federated learning aggregation strategies.

    These correspond 1-to-1 with the Rust-native strategy types in
    ``vfl_core``.  The :class:`~velocity.server.VelocityServer` translates
    them into the appropriate Rust objects at run-time.
    """

    FedAvg = "FedAvg"
    """Federated Averaging — weighted mean by number of local samples (default)."""

    FedProx = "FedProx"
    """FedAvg with a proximal regularisation term for heterogeneous clients."""

    FedMedian = "FedMedian"
    """Coordinate-wise median — robust against Byzantine / poisoned clients."""
