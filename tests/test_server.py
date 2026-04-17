"""Tests for VelocityServer (pure-Python fallback path)."""

import pytest

from velocity.server import VelocityServer
from velocity.strategy import Strategy


@pytest.fixture
def small_server():
    """Return a VelocityServer with a tiny layer configuration."""
    return VelocityServer(
        model_id="test/model",
        dataset="test-dataset",
        strategy=Strategy.FedAvg,
        storage="local://test",
        layer_shapes={"fc1": 4, "fc2": 2},
    )


def test_server_defaults(small_server):
    assert small_server.model_id == "test/model"
    assert small_server.strategy == Strategy.FedAvg
    assert small_server.global_weights == {}
    assert small_server.history == []


def test_server_run_returns_summaries(small_server):
    summaries = small_server.run(min_clients=2, rounds=3)
    assert len(summaries) == 3


def test_summary_fields(small_server):
    summaries = small_server.run(min_clients=1, rounds=1)
    s = summaries[0]
    assert "round" in s
    assert "num_clients" in s
    assert "global_loss" in s
    assert "attack_results" in s
    assert s["round"] == 1


def test_global_weights_populated_after_run(small_server):
    small_server.run(min_clients=1, rounds=1)
    weights = small_server.global_weights
    assert "fc1" in weights
    assert len(weights["fc1"]) == 4


def test_history_grows_with_rounds(small_server):
    small_server.run(min_clients=1, rounds=4)
    assert len(small_server.history) == 4


def test_simulate_attack_raises_on_invalid_type(small_server):
    with pytest.raises(ValueError, match="Unknown attack type"):
        small_server.simulate_attack("nonexistent_attack")


def test_simulate_attack_queued_before_run(small_server):
    small_server.simulate_attack("model_poisoning", intensity=0.5)
    summaries = small_server.run(min_clients=1, rounds=1)
    # The attack result should appear in the first round
    assert len(summaries) == 1


def test_simulate_attack_all_valid_types(small_server):
    for attack in ["model_poisoning", "sybil_nodes", "gaussian_noise", "label_flipping"]:
        small_server.simulate_attack(attack)
    # All four registered attacks should not raise
    summaries = small_server.run(min_clients=1, rounds=1)
    assert summaries


def test_different_strategies():
    for strategy in Strategy:
        server = VelocityServer(
            model_id="m",
            dataset="d",
            strategy=strategy,
            layer_shapes={"w": 2},
        )
        summaries = server.run(min_clients=1, rounds=1)
        assert summaries[0]["round"] == 1


def test_default_layer_shapes():
    server = VelocityServer(model_id="m", dataset="d")
    assert server.layer_shapes  # not empty
