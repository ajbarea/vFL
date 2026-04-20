"""Tests for the Strategy sum type (frozen dataclasses)."""

from dataclasses import FrozenInstanceError

import pytest
from velocity.strategy import (
    ALL_STRATEGIES,
    FedAvg,
    FedMedian,
    FedProx,
    Krum,
    MultiKrum,
    parse_strategy,
    strategy_name,
)


def test_all_strategies_tuple_covers_sum_type():
    names = {cls.__name__ for cls in ALL_STRATEGIES}
    assert names == {"FedAvg", "FedProx", "FedMedian", "Krum", "MultiKrum"}


def test_parameter_free_strategies_equal_and_hashable():
    # Frozen dataclasses compare by value, not identity.
    assert FedAvg() == FedAvg()
    assert FedMedian() == FedMedian()
    assert hash(FedAvg()) == hash(FedAvg())


def test_parameterised_strategies_compare_by_field():
    assert FedProx() == FedProx(mu=0.01)
    assert FedProx(mu=0.5) != FedProx(mu=0.1)
    assert Krum(f=2) == Krum(f=2)
    assert Krum(f=2) != Krum(f=3)
    assert MultiKrum(f=2) == MultiKrum(f=2, m=None)
    assert MultiKrum(f=2, m=5) != MultiKrum(f=2, m=6)


def test_frozen_prevents_mutation():
    s = Krum(f=2)
    with pytest.raises(FrozenInstanceError):
        s.f = 3  # type: ignore[misc]


def test_strategy_name_returns_class_name():
    assert strategy_name(FedAvg()) == "FedAvg"
    assert strategy_name(FedProx(mu=0.1)) == "FedProx"
    assert strategy_name(Krum(f=1)) == "Krum"
    assert strategy_name(MultiKrum(f=1, m=3)) == "MultiKrum"


def test_parse_strategy_string_forms():
    assert parse_strategy("FedAvg") == FedAvg()
    assert parse_strategy("FedMedian") == FedMedian()
    assert parse_strategy("FedProx") == FedProx()
    # Case-insensitive + whitespace tolerant
    assert parse_strategy("  fedavg  ") == FedAvg()


def test_parse_strategy_dict_forms():
    assert parse_strategy({"type": "FedAvg"}) == FedAvg()
    assert parse_strategy({"type": "FedProx", "mu": 0.25}) == FedProx(mu=0.25)
    assert parse_strategy({"type": "Krum", "f": 2}) == Krum(f=2)
    assert parse_strategy({"type": "MultiKrum", "f": 1, "m": 3}) == MultiKrum(f=1, m=3)
    assert parse_strategy({"type": "MultiKrum", "f": 1}) == MultiKrum(f=1, m=None)


def test_parse_strategy_passthrough():
    # Instances round-trip unchanged.
    for s in (FedAvg(), FedProx(mu=0.05), FedMedian(), Krum(f=1), MultiKrum(f=1, m=2)):
        assert parse_strategy(s) == s


def test_parse_strategy_errors():
    with pytest.raises(ValueError, match="unknown strategy"):
        parse_strategy("FedNope")
    with pytest.raises(ValueError, match="requires parameters"):
        parse_strategy("Krum")  # f is required
    with pytest.raises(ValueError, match="unknown parameter"):
        parse_strategy({"type": "Krum", "f": 2, "bogus": 1})
