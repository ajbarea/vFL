"""Attack simulation helpers for resilience testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttackResult:
    """Summary of a simulated attack."""

    attack_type: str
    clients_affected: int
    severity: float
    description: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttackResult:
        return cls(
            attack_type=d["attack_type"],
            clients_affected=d["clients_affected"],
            severity=d["severity"],
            description=d["description"],
        )

    def __str__(self) -> str:
        return (
            f"[{self.attack_type}] {self.description} "
            f"(severity={self.severity:.3f}, clients={self.clients_affected})"
        )


# Round-level attack identifiers understood by the Rust engine. These
# operate on weights/client rosters during a round; data-pipeline attacks
# (label flipping etc.) live in :mod:`velocity.data_attacks` because the
# Rust core never sees raw labels or input features.
VALID_ATTACKS: frozenset[str] = frozenset({"model_poisoning", "sybil_nodes", "gaussian_noise"})
