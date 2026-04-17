import json

from typer.testing import CliRunner

from velocity.cli import app


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
    assert {"FedAvg", "FedProx", "FedMedian"}.issubset(lines)


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
