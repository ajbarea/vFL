"""velocity.sweep — parallel strategy x attack matrix runs with auto-comparison.

Fan out multiple FL experiments across a process pool, collect results, and
write a single comparison report. Each worker owns its own Rust orchestrator,
so there's no shared state or locking — parallelism is linear in CPU count.

The CLI wrapper lives in ``velocity.cli``; this module is importable for
agent / programmatic use.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from velocity import __version__
from velocity.strategy import Strategy, parse_strategy, strategy_name


class AttackSpec(BaseModel):
    """One attack applied to a run. Unknown fields pass through to the server."""

    model_config = ConfigDict(extra="allow")

    type: str
    intensity: float = 0.1
    count: int = 1
    fraction: float = 0.1


class RunSpec(BaseModel):
    """A single experiment — strategy, dataset/model, rounds, optional attacks."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    strategy: Strategy
    model_id: str = "demo/model"
    dataset: str = "demo/dataset"
    storage: str = "local://checkpoints"
    rounds: int = Field(default=1, ge=1)
    min_clients: int = Field(default=1, ge=1)
    seed: int = 0
    attacks: list[AttackSpec] = Field(default_factory=list)

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, value: Any) -> Strategy:
        # Pydantic serialises strategies as ``{"type": "Krum", "f": 2}`` below,
        # so the same shape round-trips through model_dump → model_validate.
        return parse_strategy(value)

    @field_serializer("strategy")
    def _serialize_strategy(self, value: Strategy) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": strategy_name(value)}
        for f in fields(type(value)):
            payload[f.name] = getattr(value, f.name)
        return payload


class RunResult(BaseModel):
    """Outcome of a single run — summaries, timing, error if any."""

    model_config = ConfigDict(extra="forbid")

    spec: RunSpec
    rounds: list[dict[str, Any]]
    final_loss: float
    mean_loss: float
    elapsed_seconds: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class SweepResult(BaseModel):
    """All runs plus wall-clock and parallelism metadata."""

    model_config = ConfigDict(extra="forbid")

    runs: list[RunResult]
    total_elapsed: float
    serial_elapsed: float
    parallel: int
    out_dir: str


def load_config(path: str | Path) -> list[RunSpec]:
    """Parse a TOML experiment file into an ordered list of :class:`RunSpec`.

    TOML shape::

        [shared]
        model_id = "demo/model"
        rounds   = 10

        [[runs]]
        name     = "fedavg-baseline"
        strategy = "FedAvg"

        [[runs]]
        name     = "fedmedian-poisoned"
        strategy = "FedMedian"
        [[runs.attacks]]
        type      = "model_poisoning"
        intensity = 0.5
    """
    data: dict[str, Any] = tomllib.loads(Path(path).read_text())
    shared: dict[str, Any] = data.get("shared", {})
    runs: list[dict[str, Any]] = data.get("runs", [])
    if not runs:
        raise ValueError(f"no [[runs]] defined in {path}")
    return [RunSpec.model_validate({**shared, **run}) for run in runs]


def specs_from_cli(
    *,
    strategies: list[str],
    attacks: list[str],
    rounds: int,
    min_clients: int,
    seed: int = 0,
    model_id: str = "demo/model",
    dataset: str = "demo/dataset",
) -> list[RunSpec]:
    """Build a strategy x attack matrix from CLI flag values.

    Always includes a no-attack baseline per strategy so the researcher can
    see strategy performance with and without each attack. Strategy strings
    follow :func:`velocity.strategy.parse_strategy` — ``"FedAvg"`` for
    defaults, ``"Krum:f=2"`` for parameterised ones.
    """
    if not strategies:
        raise ValueError("at least one strategy is required")
    resolved = [_parse_sweep_strategy(s) for s in strategies]
    attack_specs: list[AttackSpec | None] = [None, *(AttackSpec(type=a) for a in attacks)]
    specs: list[RunSpec] = []
    for strategy in resolved:
        for attack in attack_specs:
            parts: list[str] = [strategy_name(strategy).lower()]
            if attack is not None:
                parts.append(attack.type)
            else:
                parts.append("baseline")
            specs.append(
                RunSpec(
                    name="-".join(parts),
                    strategy=strategy,
                    model_id=model_id,
                    dataset=dataset,
                    rounds=rounds,
                    min_clients=min_clients,
                    seed=seed + len(specs),
                    attacks=[attack] if attack is not None else [],
                )
            )
    return specs


def _parse_sweep_strategy(value: str) -> Strategy:
    """Parse a sweep-CLI strategy string (same syntax as the main CLI).

    Accepts ``FedAvg`` or ``Krum:f=2,m=3`` — delegates to
    :func:`parse_strategy` after splitting the colon form.
    """
    if ":" in value:
        name, _, rest = value.partition(":")
        params: dict[str, Any] = {}
        for pair in rest.split(","):
            if not pair:
                continue
            k, _, v = pair.partition("=")
            params[k.strip()] = _coerce_sweep_scalar(v.strip())
        return parse_strategy({"type": name, **params})
    return parse_strategy(value)


def _coerce_sweep_scalar(raw: str) -> Any:
    if raw.lower() in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _run_one(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Worker body — runs in a subprocess.

    Takes/returns plain dicts to keep pickling straightforward; re-validates
    with pydantic inside the worker so any schema drift fails cleanly.
    """
    spec = RunSpec.model_validate(spec_dict)

    import random

    random.seed(spec.seed)

    from velocity.server import VelocityServer

    start = time.perf_counter()
    error: str | None = None
    summaries: list[dict[str, Any]] = []
    try:
        server = VelocityServer(
            model_id=spec.model_id,
            dataset=spec.dataset,
            strategy=spec.strategy,
            storage=spec.storage,
        )
        for attack in spec.attacks:
            server.simulate_attack(
                attack.type,
                intensity=attack.intensity,
                count=attack.count,
                fraction=attack.fraction,
            )
        summaries = server.run(min_clients=spec.min_clients, rounds=spec.rounds)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - start

    losses = [s["global_loss"] for s in summaries if not _is_nan(s.get("global_loss"))]
    final_loss = losses[-1] if losses else float("nan")
    mean_loss = sum(losses) / len(losses) if losses else float("nan")

    return RunResult(
        spec=spec,
        rounds=summaries,
        final_loss=final_loss,
        mean_loss=mean_loss,
        elapsed_seconds=elapsed,
        error=error,
    ).model_dump(mode="json")


def _write_run_artifacts(out_dir: Path, result: RunResult) -> None:
    run_dir = out_dir / result.spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(result.spec.model_dump_json(indent=2))

    with (run_dir / "rounds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "num_clients", "global_loss", "attack_results"])
        for r in result.rounds:
            writer.writerow(
                [
                    r.get("round", ""),
                    r.get("num_clients", ""),
                    r.get("global_loss", ""),
                    json.dumps(r.get("attack_results", [])),
                ]
            )

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "final_loss": result.final_loss,
                "mean_loss": result.mean_loss,
                "elapsed_seconds": result.elapsed_seconds,
                "num_rounds": len(result.rounds),
                "error": result.error,
            },
            indent=2,
        )
    )


def capture_manifest() -> dict[str, Any]:
    """Reproducibility manifest — git state, host, Python/vFL version."""

    def _try_run(cmd: list[str]) -> str | None:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    dirty_out = _try_run(["git", "status", "--porcelain"])
    return {
        "vfl_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "git": {
            "branch": _try_run(["git", "branch", "--show-current"]),
            "commit": _try_run(["git", "rev-parse", "HEAD"]),
            "dirty": bool(dirty_out) if dirty_out is not None else None,
        },
    }


def run_sweep(
    specs: list[RunSpec],
    *,
    out_dir: Path,
    parallel: int | None = None,
) -> SweepResult:
    """Fan out `specs` across a process pool, collect results, write artifacts.

    Returns a :class:`SweepResult` containing every run ordered as the specs
    were provided (not as they completed). Also writes
    ``{out_dir}/{manifest,comparison}.{json,md}`` and per-run directories.
    """
    if not specs:
        raise ValueError("no specs to run")

    parallel = parallel or min(os.cpu_count() or 1, len(specs))
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "manifest.json").write_text(json.dumps(capture_manifest(), indent=2))

    start = time.perf_counter()
    results: list[RunResult] = []

    spec_dicts = [s.model_dump(mode="json") for s in specs]

    with ProcessPoolExecutor(
        max_workers=parallel,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [pool.submit(_run_one, d) for d in spec_dicts]
        for future in as_completed(futures):
            result = RunResult.model_validate(future.result())
            _write_run_artifacts(out_dir, result)
            results.append(result)

    spec_order = {s.name: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: spec_order[r.spec.name])

    total_elapsed = time.perf_counter() - start
    sweep_result = SweepResult(
        runs=results,
        total_elapsed=total_elapsed,
        serial_elapsed=sum(r.elapsed_seconds for r in results),
        parallel=parallel,
        out_dir=str(out_dir),
    )

    (out_dir / "comparison.json").write_text(sweep_result.model_dump_json(indent=2))
    (out_dir / "comparison.md").write_text(render_comparison(sweep_result))

    return sweep_result


def render_comparison(result: SweepResult) -> str:
    """Human-readable markdown report of a sweep."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    speedup = (
        f"{result.serial_elapsed / result.total_elapsed:.1f}x"
        if result.total_elapsed > 0
        else "n/a"
    )

    lines: list[str] = [
        f"# Sweep: {ts}",
        "",
        (
            f"{len(result.runs)} runs, {result.parallel} parallel, "
            f"total wall {result.total_elapsed:.1f}s "
            f"(serial would be {result.serial_elapsed:.1f}s — {speedup} speedup)"
        ),
        "",
        "| Run | Strategy | Attack | Final loss | Mean loss | Elapsed | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in result.runs:
        attack = r.spec.attacks[0].type if r.spec.attacks else "—"
        status = "ok" if r.succeeded else f"ERROR: {r.error}"
        final = f"{r.final_loss:.4f}" if not _is_nan(r.final_loss) else "NaN"
        mean = f"{r.mean_loss:.4f}" if not _is_nan(r.mean_loss) else "NaN"
        lines.append(
            f"| {r.spec.name} | {strategy_name(r.spec.strategy)} | {attack} | "
            f"{final} | {mean} | {r.elapsed_seconds:.1f}s | {status} |"
        )

    succeeded = [r for r in result.runs if r.succeeded and not _is_nan(r.final_loss)]
    if succeeded:
        winner = min(succeeded, key=lambda r: r.final_loss)
        lines.append("")
        lines.append(f"**Lowest final loss:** `{winner.spec.name}` ({winner.final_loss:.4f})")

    return "\n".join(lines) + "\n"
