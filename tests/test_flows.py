"""Tests for the Prefect FL flow wrapper."""

from __future__ import annotations

from typing import Any

from velocity.flows import federated_training_flow


class _FakeStrategy:
    value = "FedAvg"


class _FakeServer:
    """Minimal stand-in for VelocityServer; exposes only what flows.py reads."""

    def __init__(self, rounds: int = 2) -> None:
        self.model_id = "fake/model"
        self.rounds = rounds
        self.strategy = _FakeStrategy()
        self._round = 0

    def _run_single_round(self) -> dict[str, Any]:
        self._round += 1
        return {
            "round": self._round,
            "global_loss": 0.1 * self._round,
            "num_clients": 3,
            "attack_results": [],
        }


def test_federated_training_flow_runs_all_rounds():
    server = _FakeServer(rounds=3)
    summaries = federated_training_flow(server)
    assert len(summaries) == 3
    assert [s["round"] for s in summaries] == [1, 2, 3]
    assert summaries[-1]["num_clients"] == 3


def test_federated_training_flow_single_round():
    server = _FakeServer(rounds=1)
    summaries = federated_training_flow(server)
    assert len(summaries) == 1
    assert summaries[0]["round"] == 1
    assert summaries[0]["global_loss"] == 0.1
