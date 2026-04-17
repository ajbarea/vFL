"""VelocityFL command-line interface."""

from __future__ import annotations

import json
from typing import Optional

import typer

from velocity import __version__
from velocity.server import VelocityServer
from velocity.strategy import Strategy

app = typer.Typer(
    name="velocity",
    help="VelocityFL CLI — run federated experiments and inspect capabilities.",
    no_args_is_help=True,
)


def _parse_strategy(value: str) -> Strategy:
    normalized = value.strip().lower()
    mapping = {
        "fedavg": Strategy.FedAvg,
        "fedprox": Strategy.FedProx,
        "fedmedian": Strategy.FedMedian,
    }
    strategy = mapping.get(normalized)
    if strategy is None:
        raise typer.BadParameter("strategy must be one of: FedAvg, FedProx, FedMedian")
    return strategy


@app.command()
def version() -> None:
    """Show VelocityFL version."""
    typer.echo(__version__)


@app.command()
def strategies() -> None:
    """List available aggregation strategies."""
    for strategy in Strategy:
        typer.echo(strategy.value)


@app.command()
def run(
    model_id: str = typer.Option(..., help="Model identifier."),
    dataset: str = typer.Option(..., help="Dataset identifier."),
    strategy: str = typer.Option("FedAvg", help="FedAvg, FedProx, or FedMedian."),
    storage: str = typer.Option("local://checkpoints", help="Storage URI."),
    min_clients: int = typer.Option(1, min=1, help="Minimum number of clients."),
    rounds: int = typer.Option(1, min=1, help="Number of FL rounds."),
) -> None:
    """Run a federated learning experiment and print round summaries as JSON."""
    server = VelocityServer(
        model_id=model_id,
        dataset=dataset,
        strategy=_parse_strategy(strategy),
        storage=storage,
    )
    summaries = server.run(min_clients=min_clients, rounds=rounds)
    typer.echo(json.dumps(summaries))


@app.command("simulate-attack")
def simulate_attack(
    attack_type: str = typer.Argument(..., help="Attack name."),
    model_id: str = typer.Option("demo/model", help="Model identifier."),
    dataset: str = typer.Option("demo/dataset", help="Dataset identifier."),
    strategy: str = typer.Option("FedAvg", help="FedAvg, FedProx, or FedMedian."),
    min_clients: int = typer.Option(1, min=1, help="Minimum number of clients."),
    intensity: float = typer.Option(0.1, min=0.0, help="Attack intensity."),
    count: int = typer.Option(1, min=1, help="Sybil node count."),
    fraction: float = typer.Option(0.1, min=0.0, max=1.0, help="Label-flipping fraction."),
) -> None:
    """Register an attack and run one round to observe impact."""
    server = VelocityServer(
        model_id=model_id,
        dataset=dataset,
        strategy=_parse_strategy(strategy),
    )
    server.simulate_attack(
        attack_type,
        intensity=intensity,
        count=count,
        fraction=fraction,
    )
    summaries = server.run(min_clients=min_clients, rounds=1)
    typer.echo(json.dumps(summaries[0]))
