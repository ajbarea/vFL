"""Tests for the Strategy enum."""

from velocity.strategy import Strategy


def test_strategy_values():
    assert Strategy.FedAvg.value == "FedAvg"
    assert Strategy.FedProx.value == "FedProx"
    assert Strategy.FedMedian.value == "FedMedian"


def test_strategy_is_string_enum():
    assert Strategy.FedAvg == "FedAvg"
    assert Strategy.FedProx == "FedProx"


def test_strategy_from_value():
    assert Strategy("FedAvg") is Strategy.FedAvg


def test_all_strategies_present():
    names = {s.value for s in Strategy}
    assert "FedAvg" in names
    assert "FedProx" in names
    assert "FedMedian" in names
