"""Property-based tests for the aggregation kernel.

These don't check specific numeric outputs; they check *algebraic invariants*
that must hold for any valid input. Hypothesis generates hundreds of inputs
per run; a single counterexample breaks the test and is auto-minimized.

The invariants are strategy-agnostic on purpose — they define what an
aggregator *is*, independent of which one we chose to implement.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from velocity import _core

# ---------------------------------------------------------------------------
# Strategies — one source of truth for generating aggregation inputs.
#
# Kept intentionally small (<=4 layers, <=8 coords, <=6 clients) so a full
# Hypothesis cycle still fits in a ~3-second budget. Weight magnitudes are
# bounded so sums don't overflow f32.
# ---------------------------------------------------------------------------

_LAYER_NAMES = st.sampled_from(["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
_COORD_VALUE = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32
)
_SAMPLE_COUNT = st.integers(min_value=1, max_value=1000)


@st.composite
def _layer_shapes(draw: st.DrawFn) -> dict[str, int]:
    names = draw(st.lists(_LAYER_NAMES, min_size=1, max_size=4, unique=True))
    return {name: draw(st.integers(min_value=1, max_value=8)) for name in names}


@st.composite
def _client_updates(draw: st.DrawFn, n_clients: int) -> list[_core.ClientUpdate]:
    """A list of `n_clients` updates that all share the same layer shapes."""
    shapes = draw(_layer_shapes())
    return [
        _core.ClientUpdate(
            num_samples=draw(_SAMPLE_COUNT),
            weights={
                name: draw(st.lists(_COORD_VALUE, min_size=size, max_size=size))
                for name, size in shapes.items()
            },
        )
        for _ in range(n_clients)
    ]


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# FedAvg invariants
# ---------------------------------------------------------------------------

_SETTINGS = settings(
    max_examples=30,
    deadline=None,  # PyO3 call + small allocations add jitter; don't fail on slow examples
    suppress_health_check=[HealthCheck.too_slow],
)


@given(update=_client_updates(n_clients=1))
@_SETTINGS
def test_fedavg_singleton_returns_input(update: list[_core.ClientUpdate]) -> None:
    """FedAvg of one update must return that update verbatim (per-coord)."""
    result = _core.aggregate(update, _core.Strategy.fed_avg())
    for name, values in update[0].weights.items():
        assert name in result
        assert len(result[name]) == len(values)
        for got, want in zip(result[name], values):
            assert _close(got, want), f"{name}: {got} != {want}"


@given(n=st.integers(min_value=2, max_value=6), template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedavg_identical_updates_is_identity(n: int, template: list[_core.ClientUpdate]) -> None:
    """FedAvg of N copies of U == U. (Rules out any bogus bias in the accumulator.)"""
    copies = [
        _core.ClientUpdate(num_samples=template[0].num_samples, weights=dict(template[0].weights))
        for _ in range(n)
    ]
    result = _core.aggregate(copies, _core.Strategy.fed_avg())
    for name, values in template[0].weights.items():
        for got, want in zip(result[name], values):
            assert _close(got, want)


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedavg_matches_weighted_mean_reference(updates: list[_core.ClientUpdate]) -> None:
    """Rust FedAvg must match the textbook weighted mean computed in pure Python."""
    total = sum(u.num_samples for u in updates)
    result = _core.aggregate(updates, _core.Strategy.fed_avg())
    for name in updates[0].weights:
        for i in range(len(updates[0].weights[name])):
            expected = sum(u.weights[name][i] * (u.num_samples / total) for u in updates)
            assert _close(result[name][i], expected, tol=1e-3)


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedavg_preserves_layer_shape(updates: list[_core.ClientUpdate]) -> None:
    """Aggregation must not rename or resize layers."""
    result = _core.aggregate(updates, _core.Strategy.fed_avg())
    assert result.keys() == updates[0].weights.keys()
    for name, values in updates[0].weights.items():
        assert len(result[name]) == len(values)


# ---------------------------------------------------------------------------
# FedMedian invariants
# ---------------------------------------------------------------------------


@given(update=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_singleton_returns_input(update: list[_core.ClientUpdate]) -> None:
    result = _core.aggregate(update, _core.Strategy.fed_median())
    for name, values in update[0].weights.items():
        for got, want in zip(result[name], values):
            assert _close(got, want)


@given(n=st.integers(min_value=2, max_value=5), template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_identical_updates_is_identity(
    n: int, template: list[_core.ClientUpdate]
) -> None:
    """FedMedian of N copies of U must equal U coordinate-wise."""
    copies = [
        _core.ClientUpdate(num_samples=template[0].num_samples, weights=dict(template[0].weights))
        for _ in range(n)
    ]
    result = _core.aggregate(copies, _core.Strategy.fed_median())
    for name, values in template[0].weights.items():
        for got, want in zip(result[name], values):
            assert _close(got, want)


@given(template=_client_updates(n_clients=1))
@_SETTINGS
def test_fedmedian_resists_one_extreme_outlier(
    template: list[_core.ClientUpdate],
) -> None:
    """4 honest clients with weights U, 1 attacker with weights U+100 ⇒ median ≈ U.

    This is the Byzantine-robustness claim for FedMedian: a single client with
    arbitrarily large values cannot move the coordinate-wise median when the
    majority is honest. FedAvg would shift proportionally; median doesn't.
    """
    base_weights = dict(template[0].weights)
    honest = [_core.ClientUpdate(num_samples=100, weights=dict(base_weights)) for _ in range(4)]
    attacker_weights = {name: [v + 100.0 for v in vs] for name, vs in base_weights.items()}
    attacker = _core.ClientUpdate(num_samples=100, weights=attacker_weights)

    result = _core.aggregate([*honest, attacker], _core.Strategy.fed_median())
    for name, values in base_weights.items():
        for got, want in zip(result[name], values):
            assert _close(got, want), f"median moved under 1 outlier: {got} vs {want}"


# ---------------------------------------------------------------------------
# FedProx invariants — same aggregation kernel as FedAvg with μ metadata
# ---------------------------------------------------------------------------


@given(updates=_client_updates(n_clients=3))
@_SETTINGS
def test_fedprox_matches_fedavg_output(updates: list[_core.ClientUpdate]) -> None:
    """FedProx and FedAvg must produce the same aggregated weights.

    μ is consumed during *local* training (proximal regularizer on the client);
    it's not a server-side aggregation knob. The kernel is weighted mean either
    way — this test pins that invariant.
    """
    fedavg = _core.aggregate(updates, _core.Strategy.fed_avg())
    fedprox = _core.aggregate(updates, _core.Strategy.fed_prox(0.01))
    for name in fedavg:
        for a, p in zip(fedavg[name], fedprox[name]):
            assert _close(a, p)


# ---------------------------------------------------------------------------
# Shape-mismatch failure mode — not algebra, but a contract the kernel must enforce
# ---------------------------------------------------------------------------


def test_aggregate_rejects_mismatched_layer_sizes() -> None:
    u1 = _core.ClientUpdate(num_samples=10, weights={"l": [1.0, 2.0, 3.0]})
    u2 = _core.ClientUpdate(num_samples=10, weights={"l": [1.0, 2.0]})
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary surfaces plain exceptions
        _core.aggregate([u1, u2], _core.Strategy.fed_avg())


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(Exception):  # noqa: B017 — PyO3 boundary
        _core.aggregate([], _core.Strategy.fed_avg())
