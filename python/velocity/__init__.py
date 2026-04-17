"""velocity — Python interface for VelocityFL.

High-performance federated learning orchestration backed by a Rust engine.
"""

from velocity.server import VelocityServer
from velocity.strategy import Strategy

__all__ = ["Strategy", "VelocityServer"]
__version__ = "0.1.0"
