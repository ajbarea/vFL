"""velocity — Python interface for VelocityFL.

High-performance federated learning orchestration backed by a Rust engine.
"""

from velocity.server import VelocityServer
from velocity.strategy import (
    FedAvg,
    FedMedian,
    FedProx,
    Krum,
    MultiKrum,
    Strategy,
    parse_strategy,
)

__all__ = [
    "FedAvg",
    "FedMedian",
    "FedProx",
    "Krum",
    "MultiKrum",
    "Strategy",
    "VelocityServer",
    "parse_strategy",
]
__version__ = "0.1.0"
