"""FL aggregation strategies.

Each strategy is a frozen dataclass carrying its own parameters (if any).
``Strategy`` is a union type alias for hinting parameters that accept any of
them. Instantiate the dataclass you want::

    from velocity import FedAvg, FedProx, Krum, MultiKrum, VelocityServer

    server = VelocityServer(..., strategy=Krum(f=2))
    server = VelocityServer(..., strategy=MultiKrum(f=2, m=7))

CLI / TOML consumers that need to accept a user-supplied string pass it
through :func:`parse_strategy`: ``"FedAvg"`` for default instances,
``{"type": "Krum", "f": 2}`` for parameterised ones.

Matches the Flower 2026 strategy-object pattern. Migrated from a string-
backed ``Enum`` in v0.1.0 — callers that previously wrote
``Strategy.FedAvg`` now write ``FedAvg()``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Union, cast


@dataclass(frozen=True)
class FedAvg:
    """Federated Averaging — weighted mean by number of local samples."""


@dataclass(frozen=True)
class FedProx:
    """FedAvg with a proximal regularisation term.

    ``mu`` controls how strongly each client is pulled back toward the global
    model during local training. Does not change the aggregation kernel —
    FedProx and FedAvg produce identical server-side weights; the client's
    objective is where ``mu`` lives.
    """

    mu: float = 0.01


@dataclass(frozen=True)
class FedMedian:
    """Coordinate-wise median — tolerates up to ~(n-1)/2 Byzantine clients."""


@dataclass(frozen=True)
class Krum:
    """Byzantine-robust selection (Blanchard et al. 2017, arXiv:1703.02757).

    Picks the single client whose sum of ``n - f - 2`` smallest squared
    distances to other clients is minimal. Requires ``n >= 2*f + 3``.
    """

    f: int


@dataclass(frozen=True)
class MultiKrum:
    """Multi-Krum (El Mhamdi et al. 2018) — averages top-``m`` by Krum score.

    ``m = None`` resolves to ``n - f`` at aggregation time ("largest
    non-Byzantine group" interpretation). Requires ``n >= 2*f + 3`` and
    ``1 <= m <= n - f``. ``MultiKrum(f, m=1)`` reduces to :class:`Krum`.
    """

    f: int
    m: int | None = None


Strategy = Union[FedAvg, FedProx, FedMedian, Krum, MultiKrum]
"""Union of every aggregation strategy — use in type hints and isinstance checks."""


ALL_STRATEGIES: tuple[type[Strategy], ...] = (FedAvg, FedProx, FedMedian, Krum, MultiKrum)
"""Every concrete strategy class, in stable display order."""

_NAME_TO_CLASS: dict[str, type[Strategy]] = {cls.__name__: cls for cls in ALL_STRATEGIES}


def strategy_name(strategy: Strategy) -> str:
    """Class name of a strategy instance, e.g. ``"Krum"`` for ``Krum(f=2)``."""
    return type(strategy).__name__


def parse_strategy(value: str | dict[str, Any] | Strategy) -> Strategy:
    """Coerce a user-supplied value into a strategy instance.

    Accepts three shapes:

    * A strategy instance — returned as-is (no copy).
    * A string like ``"FedAvg"`` or ``"krum"`` — returns a default-constructed
      instance. Raises :class:`ValueError` if the strategy requires parameters
      (e.g. ``"Krum"`` has no default for ``f``).
    * A mapping like ``{"type": "Krum", "f": 2}`` — the ``type`` / ``name``
      key selects the class, remaining keys populate its fields.

    Lookup is case-insensitive on the class name.
    """
    if isinstance(value, ALL_STRATEGIES):
        return value  # type: ignore[return-value]

    if isinstance(value, str):
        cls = _lookup(value)
        try:
            # Parameter-free strategies construct cleanly; parameterised ones
            # raise TypeError on missing args — that path is caught below to
            # produce the friendlier ValueError with the required-field list.
            return cls()  # ty: ignore[missing-argument]
        except TypeError as exc:
            required = [f.name for f in fields(cls) if f.default is MISSING]
            raise ValueError(
                f"strategy {cls.__name__!r} requires parameters "
                f"{required}; pass a dict like "
                f"{{'type': {cls.__name__!r}, ...}}"
            ) from exc

    if isinstance(value, dict):
        value_dict = cast(dict[str, Any], value)
        kind = value_dict.get("type") or value_dict.get("name")
        if not isinstance(kind, str):
            raise ValueError("strategy dict must have a string 'type' (or 'name') key")
        cls = _lookup(kind)
        params: dict[str, Any] = {
            k: v for k, v in value_dict.items() if k not in {"type", "name"}
        }
        field_names = {f.name for f in fields(cls)}
        unknown = set(params) - field_names
        if unknown:
            raise ValueError(f"unknown parameter(s) for {cls.__name__}: {sorted(unknown)}")
        return cls(**params)

    raise TypeError(
        f"strategy must be a Strategy instance, str, or dict — got {type(value).__name__}"
    )


def _lookup(name: str) -> type[Strategy]:
    normalized = name.strip()
    for cname, cls in _NAME_TO_CLASS.items():
        if cname.lower() == normalized.lower():
            return cls
    valid = ", ".join(_NAME_TO_CLASS)
    raise ValueError(f"unknown strategy {name!r}. Valid: {valid}")
