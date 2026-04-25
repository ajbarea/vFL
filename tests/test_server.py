"""Tests for VelocityServer (pure-Python fallback path)."""

import json

import pytest
from velocity.server import VelocityServer, _PurePythonOrchestrator
from velocity.strategy import FedAvg, FedMedian, FedProx


@pytest.fixture
def small_server():
    """Return a VelocityServer with a tiny layer configuration."""
    return VelocityServer(
        model_id="test/model",
        dataset="test-dataset",
        strategy=FedAvg(),
        storage="local://test",
        layer_shapes={"fc1": 4, "fc2": 2},
    )


def test_server_defaults(small_server):
    assert small_server.model_id == "test/model"
    assert small_server.strategy == FedAvg()
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
    for attack in ["model_poisoning", "sybil_nodes", "gaussian_noise"]:
        small_server.simulate_attack(attack)
    # All three round-level attacks should register without error.
    summaries = small_server.run(min_clients=1, rounds=1)
    assert summaries


@pytest.mark.parametrize("strategy", [FedAvg(), FedMedian(), FedProx()])
def test_different_strategies(strategy):
    # Parameter-free strategies only; Krum/MultiKrum need n >= 2f+3 clients
    # and have their own dedicated tests in test_strategy.py.
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


# ---------------------------------------------------------------------------
# _PurePythonOrchestrator — the fallback used when velocity._core is absent
# ---------------------------------------------------------------------------


@pytest.fixture
def pure_orc():
    return _PurePythonOrchestrator(
        model_id="t", min_clients=2, rounds=3, layer_shapes={"w": 4, "b": 1}
    )


def test_pure_python_orchestrator_run_round_returns_summary(pure_orc):
    summary = pure_orc.run_round(num_clients=3)
    assert summary["round"] == 1
    assert summary["num_clients"] == 3
    assert "global_loss" in summary
    assert summary["attack_results"] == []


def test_pure_python_orchestrator_increments_round_count(pure_orc):
    pure_orc.run_round(num_clients=2)
    pure_orc.run_round(num_clients=2)
    assert pure_orc.run_round(num_clients=2)["round"] == 3


def test_pure_python_orchestrator_updates_global_weights(pure_orc):
    initial = [list(v) for v in pure_orc.global_weights().values()]
    pure_orc.run_round(num_clients=3)
    updated = [list(v) for v in pure_orc.global_weights().values()]
    # With random clients, at least one layer's weights should have moved off zero
    assert updated != initial


def test_pure_python_orchestrator_register_and_consume_attack(pure_orc):
    pure_orc.register_attack(attack_type="gaussian_noise", std=0.1)
    summary = pure_orc.run_round(num_clients=2)
    assert summary["attack_results"] == [{"attack_type": "gaussian_noise"}]
    # Attack buffer should be consumed
    assert pure_orc.run_round(num_clients=2)["attack_results"] == []


def test_pure_python_orchestrator_history_accumulates(pure_orc):
    pure_orc.run_round(num_clients=2)
    pure_orc.run_round(num_clients=2)
    history = json.loads(pure_orc.history_json())
    assert len(history) == 2
    assert [h["round"] for h in history] == [1, 2]
