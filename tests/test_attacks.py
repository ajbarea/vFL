"""Tests for AttackResult and attack validation helpers."""

from dataclasses import FrozenInstanceError

import pytest
from velocity.attacks import VALID_ATTACKS, AttackResult


def test_attack_result_from_dict():
    data = {
        "attack_type": "model_poisoning",
        "clients_affected": 2,
        "severity": 0.15,
        "description": "Some poisoning",
    }
    result = AttackResult.from_dict(data)
    assert result.attack_type == "model_poisoning"
    assert result.clients_affected == 2
    assert abs(result.severity - 0.15) < 1e-9
    assert result.description == "Some poisoning"


def test_attack_result_str():
    result = AttackResult(
        attack_type="sybil_nodes",
        clients_affected=5,
        severity=1.0,
        description="5 Byzantine clients injected",
    )
    s = str(result)
    assert "sybil_nodes" in s
    assert "5 Byzantine clients injected" in s


def test_valid_attacks_contains_expected():
    # Round-level attacks only — label_flipping is a data-pipeline attack
    # and lives in velocity.data_attacks (DATA_ATTACK_TYPES).
    assert "model_poisoning" in VALID_ATTACKS
    assert "sybil_nodes" in VALID_ATTACKS
    assert "gaussian_noise" in VALID_ATTACKS
    assert "label_flipping" not in VALID_ATTACKS
    assert len(VALID_ATTACKS) == 3


def test_attack_result_is_frozen():
    result = AttackResult("model_poisoning", 1, 0.1, "desc")
    with pytest.raises(FrozenInstanceError):
        result.severity = 0.9  # type: ignore[misc]
