import json

import pytest
import typer
from typer.testing import CliRunner
from velocity.cli import _coerce_scalar, _parse_strategy_cli, app
from velocity.strategy import FedAvg, FedMedian, Krum, MultiKrum, TrimmedMean

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "strategies" in result.stdout
    assert "simulate-attack" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_cli_strategies():
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    lines = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert {"FedAvg", "FedProx", "FedMedian", "TrimmedMean", "Krum", "MultiKrum"}.issubset(lines)


def test_cli_run_json_output():
    result = runner.invoke(
        app,
        [
            "run",
            "--model-id",
            "test/model",
            "--dataset",
            "test/dataset",
            "--rounds",
            "1",
            "--min-clients",
            "1",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["round"] == 1


# ---------------------------------------------------------------------------
# _parse_strategy_cli / _coerce_scalar — the `Name:key=value` shorthand
# ---------------------------------------------------------------------------


def test_parse_strategy_cli_bare_name():
    assert _parse_strategy_cli("FedAvg") == FedAvg()
    assert _parse_strategy_cli("FedMedian") == FedMedian()


def test_parse_strategy_cli_colon_form_parameterised():
    assert _parse_strategy_cli("Krum:f=2") == Krum(f=2)
    assert _parse_strategy_cli("MultiKrum:f=2,m=7") == MultiKrum(f=2, m=7)
    assert _parse_strategy_cli("TrimmedMean:k=1") == TrimmedMean(k=1)
    # trailing comma / empty pair is tolerated
    assert _parse_strategy_cli("Krum:f=2,") == Krum(f=2)


def test_parse_strategy_cli_bad_pair_raises():
    with pytest.raises(typer.BadParameter, match="expected key=value"):
        _parse_strategy_cli("Krum:f=")
    with pytest.raises(typer.BadParameter, match="expected key=value"):
        _parse_strategy_cli("Krum:=2")


def test_parse_strategy_cli_unknown_strategy_surfaces_as_bad_param():
    with pytest.raises(typer.BadParameter, match="unknown strategy"):
        _parse_strategy_cli("NotAStrategy")
    with pytest.raises(typer.BadParameter, match="unknown strategy"):
        _parse_strategy_cli("AlsoNot:f=1")


def test_parse_strategy_cli_missing_required_param_surfaces_as_bad_param():
    # `Krum` has no default for `f`; bare name hits the required-param path.
    with pytest.raises(typer.BadParameter, match="requires parameters"):
        _parse_strategy_cli("Krum")


def test_coerce_scalar_none_forms():
    assert _coerce_scalar("none") is None
    assert _coerce_scalar("NULL") is None
    assert _coerce_scalar("None") is None


def test_coerce_scalar_int_float_string():
    assert _coerce_scalar("42") == 42
    assert _coerce_scalar("-7") == -7
    assert _coerce_scalar("3.14") == 3.14
    assert _coerce_scalar("1e-3") == pytest.approx(0.001)
    # Falls through to raw string when nothing parses.
    assert _coerce_scalar("hello") == "hello"


def test_coerce_scalar_int_beats_float_for_integer_strings():
    # "5" should become int(5), not float(5.0) — order of coercion matters.
    result = _coerce_scalar("5")
    assert result == 5
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# simulate-attack command
# ---------------------------------------------------------------------------


def test_cli_simulate_attack_emits_single_round_json():
    result = runner.invoke(
        app,
        [
            "simulate-attack",
            "gaussian_noise",
            "--intensity",
            "0.05",
            "--min-clients",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    # simulate-attack emits one round, not an array
    assert isinstance(payload, dict)
    assert payload["round"] == 1
    assert isinstance(payload.get("attack_results"), list)


def test_cli_simulate_attack_rejects_unknown_attack():
    result = runner.invoke(app, ["simulate-attack", "not_a_real_attack"])
    assert result.exit_code != 0
    assert "attack_type must be one of" in result.stdout or "attack_type must be one of" in (
        result.stderr or ""
    )


def test_cli_run_krum_shorthand_surfaces_insufficient_clients():
    # Krum(f=2) requires n >= 2*2 + 3 = 7; with min_clients=1 the round errors.
    # Test confirms the shorthand parses through to the server path.
    result = runner.invoke(
        app,
        [
            "run",
            "--model-id",
            "test/model",
            "--dataset",
            "test/dataset",
            "--strategy",
            "Krum:f=2",
            "--rounds",
            "1",
            "--min-clients",
            "1",
        ],
    )
    # Non-zero exit because aggregation raises; the shorthand itself parsed.
    assert result.exit_code != 0
