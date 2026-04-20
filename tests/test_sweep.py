"""Tests for velocity.sweep — the parallel strategy x attack matrix runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from velocity.cli import app
from velocity.strategy import FedAvg, FedMedian, FedProx, Krum, MultiKrum, parse_strategy
from velocity.sweep import (
    AttackSpec,
    RunResult,
    RunSpec,
    SweepResult,
    _coerce_sweep_scalar,
    _parse_sweep_strategy,
    _run_one,
    capture_manifest,
    load_config,
    render_comparison,
    run_sweep,
    specs_from_cli,
)


def test_parse_strategy_case_insensitive():
    assert parse_strategy("fedavg") == FedAvg()
    assert parse_strategy("FedMedian") == FedMedian()
    assert parse_strategy("  FEDPROX  ") == FedProx()


def test_parse_strategy_unknown_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        parse_strategy("FedNope")


def test_parse_strategy_dict_form():
    assert parse_strategy({"type": "Krum", "f": 2}) == Krum(f=2)
    assert parse_strategy({"type": "MultiKrum", "f": 1, "m": 3}) == MultiKrum(f=1, m=3)
    # m defaults to None when omitted
    assert parse_strategy({"type": "MultiKrum", "f": 1}) == MultiKrum(f=1, m=None)


def test_parse_strategy_passthrough_and_errors():
    # Idempotent: a strategy instance round-trips.
    assert parse_strategy(FedAvg()) == FedAvg()
    # Missing required param surfaces a typed error.
    with pytest.raises(ValueError, match="requires parameters"):
        parse_strategy("Krum")
    # Unknown parameter name.
    with pytest.raises(ValueError, match="unknown parameter"):
        parse_strategy({"type": "Krum", "f": 2, "bogus": 1})


def test_specs_from_cli_includes_baseline_per_strategy():
    specs = specs_from_cli(
        strategies=["FedAvg", "FedMedian"],
        attacks=["model_poisoning"],
        rounds=1,
        min_clients=1,
    )
    # 2 strategies x (baseline + 1 attack) = 4
    assert len(specs) == 4
    names = [s.name for s in specs]
    assert "fedavg-baseline" in names
    assert "fedavg-model_poisoning" in names
    assert "fedmedian-baseline" in names
    assert "fedmedian-model_poisoning" in names


def test_specs_from_cli_no_attacks_still_runs_baselines():
    specs = specs_from_cli(
        strategies=["FedAvg"],
        attacks=[],
        rounds=1,
        min_clients=1,
    )
    assert len(specs) == 1
    assert specs[0].attacks == []


def test_specs_from_cli_rejects_empty_strategies():
    with pytest.raises(ValueError, match="at least one strategy"):
        specs_from_cli(strategies=[], attacks=[], rounds=1, min_clients=1)


def test_specs_from_cli_seed_increments_per_run():
    specs = specs_from_cli(
        strategies=["FedAvg", "FedMedian"],
        attacks=[],
        rounds=1,
        min_clients=1,
        seed=10,
    )
    seeds = [s.seed for s in specs]
    assert seeds == [10, 11]


def test_load_config_merges_shared_and_runs(tmp_path: Path):
    toml = tmp_path / "exp.toml"
    toml.write_text(
        """
[shared]
model_id    = "demo/m"
dataset     = "demo/d"
rounds      = 2
min_clients = 1

[[runs]]
name     = "a"
strategy = "FedAvg"

[[runs]]
name     = "b"
strategy = "FedMedian"
[[runs.attacks]]
type      = "model_poisoning"
intensity = 0.3
"""
    )
    specs = load_config(toml)
    assert len(specs) == 2
    assert specs[0].name == "a"
    assert specs[0].strategy == FedAvg()
    assert specs[0].rounds == 2
    assert specs[1].attacks[0].type == "model_poisoning"
    assert specs[1].attacks[0].intensity == 0.3


def test_load_config_empty_runs_raises(tmp_path: Path):
    toml = tmp_path / "empty.toml"
    toml.write_text('[shared]\nmodel_id = "x"\n')
    with pytest.raises(ValueError, match="no \\[\\[runs\\]\\]"):
        load_config(toml)


def test_attack_spec_passes_unknown_fields_through():
    spec = AttackSpec(type="model_poisoning", intensity=0.5, custom_param="foo")  # type: ignore[call-arg]
    # extra="allow" means custom_param is kept on the model
    assert spec.model_dump()["custom_param"] == "foo"


def test_run_spec_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RunSpec(
            name="x",
            strategy=FedAvg(),
            bogus_field="nope",  # type: ignore[call-arg]
        )


def test_capture_manifest_has_expected_keys():
    m = capture_manifest()
    assert "vfl_version" in m
    assert "timestamp" in m
    assert "host" in m and "cpu_count" in m["host"]
    assert "git" in m  # branch/commit may be None outside a git repo, but key must exist


def test_run_sweep_end_to_end_writes_artifacts(tmp_path: Path):
    specs = specs_from_cli(
        strategies=["FedAvg", "FedMedian"],
        attacks=[],
        rounds=1,
        min_clients=1,
    )
    result = run_sweep(specs, out_dir=tmp_path, parallel=2)
    assert isinstance(result, SweepResult)
    assert len(result.runs) == 2
    assert all(r.succeeded for r in result.runs)
    # Spec order preserved (not completion order)
    assert [r.spec.name for r in result.runs] == [s.name for s in specs]
    # Artifacts present
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
    for s in specs:
        assert (tmp_path / s.name / "config.json").exists()
        assert (tmp_path / s.name / "rounds.csv").exists()
        assert (tmp_path / s.name / "summary.json").exists()


def test_run_sweep_rejects_empty_specs(tmp_path: Path):
    with pytest.raises(ValueError, match="no specs"):
        run_sweep([], out_dir=tmp_path)


def test_render_comparison_shows_nan_when_loss_missing(tmp_path: Path):
    # Without torch, the Rust orchestrator reports NaN for global_loss —
    # render_comparison should surface that honestly rather than inventing a winner.
    specs = specs_from_cli(
        strategies=["FedAvg"],
        attacks=[],
        rounds=1,
        min_clients=1,
    )
    result = run_sweep(specs, out_dir=tmp_path, parallel=1)
    md = render_comparison(result)
    assert "# Sweep:" in md
    assert "| Run |" in md
    assert "NaN" in md
    assert "Lowest final loss" not in md  # no real metric ⇒ no winner line


def test_render_comparison_picks_lowest_loss_winner():
    def _run(name: str, strategy, loss: float) -> RunResult:
        return RunResult(
            spec=RunSpec(name=name, strategy=strategy, rounds=1, min_clients=1),
            rounds=[],
            final_loss=loss,
            mean_loss=loss,
            elapsed_seconds=1.0,
        )

    result = SweepResult(
        runs=[
            _run("a", FedAvg(), 0.5),
            _run("b", FedMedian(), 0.1),  # winner
            _run("c", FedProx(), 0.3),
        ],
        total_elapsed=2.0,
        serial_elapsed=6.0,
        parallel=3,
        out_dir="/tmp/fake",
    )
    md = render_comparison(result)
    assert "Lowest final loss" in md
    assert "`b`" in md  # winner name in backticks


def test_comparison_json_round_trips(tmp_path: Path):
    specs = specs_from_cli(
        strategies=["FedAvg"],
        attacks=[],
        rounds=1,
        min_clients=1,
    )
    run_sweep(specs, out_dir=tmp_path, parallel=1)
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert data["parallel"] == 1
    assert len(data["runs"]) == 1
    assert data["runs"][0]["spec"]["name"] == "fedavg-baseline"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

runner = CliRunner()


def test_cli_sweep_adhoc_writes_artifacts(tmp_path: Path):
    out = tmp_path / "sweep"
    result = runner.invoke(
        app,
        [
            "sweep",
            "--strategies",
            "FedAvg,FedMedian",
            "--rounds",
            "1",
            "--min-clients",
            "1",
            "--out",
            str(out),
            "--parallel",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "comparison.md").exists()
    assert (out / "comparison.json").exists()


def test_cli_sweep_config_file(tmp_path: Path):
    toml = tmp_path / "exp.toml"
    toml.write_text(
        """
[shared]
model_id    = "demo/m"
dataset     = "demo/d"
rounds      = 1
min_clients = 1

[[runs]]
name     = "a"
strategy = "FedAvg"
"""
    )
    out = tmp_path / "sweep"
    result = runner.invoke(app, ["sweep", str(toml), "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / "a" / "summary.json").exists()


def test_cli_sweep_rejects_empty_invocation():
    result = runner.invoke(app, ["sweep"])
    assert result.exit_code != 0


def test_cli_sweep_rejects_unknown_attack(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "sweep",
            "--strategies",
            "FedAvg",
            "--attacks",
            "not_a_real_attack",
            "--rounds",
            "1",
            "--min-clients",
            "1",
            "--out",
            str(tmp_path / "sweep"),
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _parse_sweep_strategy / _coerce_sweep_scalar — sweep-CLI shorthand internals
# ---------------------------------------------------------------------------


def test_parse_sweep_strategy_bare_name():
    assert _parse_sweep_strategy("FedAvg") == FedAvg()
    assert _parse_sweep_strategy("FedMedian") == FedMedian()


def test_parse_sweep_strategy_colon_form():
    assert _parse_sweep_strategy("Krum:f=2") == Krum(f=2)
    assert _parse_sweep_strategy("MultiKrum:f=1,m=3") == MultiKrum(f=1, m=3)
    # Trailing comma / empty pair is tolerated
    assert _parse_sweep_strategy("Krum:f=2,") == Krum(f=2)


def test_parse_sweep_strategy_colon_form_with_none():
    # `m=none` should coerce to None, making MultiKrum's default-at-aggregation-time path explicit
    assert _parse_sweep_strategy("MultiKrum:f=1,m=none") == MultiKrum(f=1, m=None)


def test_coerce_sweep_scalar_none_forms():
    assert _coerce_sweep_scalar("none") is None
    assert _coerce_sweep_scalar("NULL") is None


def test_coerce_sweep_scalar_int_float_string():
    assert _coerce_sweep_scalar("42") == 42
    assert isinstance(_coerce_sweep_scalar("42"), int)
    assert _coerce_sweep_scalar("3.14") == 3.14
    assert _coerce_sweep_scalar("hello") == "hello"


# ---------------------------------------------------------------------------
# Strategy serialisation round-trip — exercises _serialize_strategy
# ---------------------------------------------------------------------------


def test_run_spec_strategy_serialises_with_fields():
    # model_dump invokes _serialize_strategy; the result must feed straight
    # back into model_validate (same shape parse_strategy accepts).
    spec = RunSpec(name="k", strategy=Krum(f=2), rounds=1, min_clients=1)
    dumped = spec.model_dump(mode="json")
    assert dumped["strategy"] == {"type": "Krum", "f": 2}
    # Round-trip
    rehydrated = RunSpec.model_validate(dumped)
    assert rehydrated.strategy == Krum(f=2)


def test_run_spec_multikrum_serialises_all_fields():
    spec = RunSpec(name="mk", strategy=MultiKrum(f=1, m=3), rounds=1, min_clients=1)
    dumped = spec.model_dump(mode="json")
    assert dumped["strategy"] == {"type": "MultiKrum", "f": 1, "m": 3}


def test_run_spec_fedavg_serialises_with_no_params():
    # Parameter-free dataclasses still emit {"type": ...} cleanly.
    spec = RunSpec(name="b", strategy=FedAvg(), rounds=1, min_clients=1)
    assert spec.model_dump(mode="json")["strategy"] == {"type": "FedAvg"}


# ---------------------------------------------------------------------------
# _run_one — worker body, called directly (not via subprocess) for coverage
# ---------------------------------------------------------------------------


def test_run_one_happy_path_returns_serialisable_dict():
    spec = RunSpec(name="direct", strategy=FedAvg(), rounds=1, min_clients=1)
    result = _run_one(spec.model_dump(mode="json"))
    assert result["spec"]["name"] == "direct"
    assert isinstance(result["rounds"], list)
    assert result["error"] is None
    assert "final_loss" in result and "mean_loss" in result
    # Re-validates cleanly — round-trip through the pydantic schema.
    rehydrated = RunResult.model_validate(result)
    assert rehydrated.succeeded


def test_run_one_captures_aggregation_error_as_string():
    # Krum(f=2) requires n >= 7; with min_clients=1 the Rust kernel raises,
    # and _run_one must catch it and surface an error string rather than crash.
    spec = RunSpec(name="err", strategy=Krum(f=2), rounds=1, min_clients=1)
    result = _run_one(spec.model_dump(mode="json"))
    assert result["error"] is not None
    assert isinstance(result["error"], str)
    # final_loss / mean_loss fall back to NaN when no rounds succeeded
    assert result["final_loss"] != result["final_loss"]  # NaN != NaN
    assert result["mean_loss"] != result["mean_loss"]
