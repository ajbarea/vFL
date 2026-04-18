"""Type stubs for the PyO3-compiled `velocity._core` module.

These declarations let static analyzers (ty, pyright, mypy) reason about the
native extension without introspecting the compiled `.so`. Keep the stub
surface in sync with `vfl-core/src/lib.rs`.
"""

from typing import Any

class Strategy:
    @staticmethod
    def fed_avg() -> Strategy: ...
    @staticmethod
    def fed_prox(mu: float) -> Strategy: ...
    @staticmethod
    def fed_median() -> Strategy: ...

class ClientUpdate:
    num_samples: int
    weights: dict[str, list[float]]
    def __init__(self, num_samples: int, weights: dict[str, list[float]]) -> None: ...

class RoundSummary:
    round: int
    num_clients: int
    global_loss: float
    attack_results: str  # JSON-encoded list

class Orchestrator:
    def __init__(
        self,
        model_id: str,
        dataset: str,
        strategy: Strategy,
        storage: str,
        min_clients: int,
        rounds: int,
        layer_shapes: dict[str, int],
    ) -> None: ...
    def register_attack(
        self,
        attack_type: str,
        intensity: float = ...,
        count: int = ...,
        fraction: float = ...,
    ) -> None: ...
    def run_round(
        self,
        updates: list[ClientUpdate],
        reported_loss: float | None = ...,
    ) -> RoundSummary: ...
    def global_weights(self) -> dict[str, list[float]]: ...
    def set_global_weights(self, weights: dict[str, list[float]]) -> None: ...
    def history_json(self) -> str: ...

def aggregate(updates: list[ClientUpdate], strategy: Strategy) -> dict[str, list[float]]: ...
def apply_gaussian_noise(
    weights: dict[str, list[float]], std_dev: float
) -> tuple[dict[str, list[float]], str]: ...

# Catch-all for anything else exposed by the compiled module.
def __getattr__(name: str) -> Any: ...
