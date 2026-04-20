"""NumPy reference implementations of Krum and Multi-Krum.

These are independent oracles we compare the Rust kernel against. Kept in
`tests/` so they never ship in the installed package — they're test fixtures,
not a second implementation to maintain.

Algorithms match:
  Krum       — Blanchard et al. 2017, "Machine Learning with Adversaries"
  Multi-Krum — El Mhamdi et al. 2018, "The Hidden Vulnerability of Distributed
               Learning in Byzantium"

Both require `n >= 2*f + 3` where n is the number of updates and f is the
assumed Byzantine count. Both flatten updates into a single vector for the
distance computation, then return the original per-layer shapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np


class ClientWeights(TypedDict):
    num_samples: int
    weights: Mapping[str, Sequence[float]]


def _flatten(weights: Mapping[str, Sequence[float]], order: Sequence[str]) -> np.ndarray:
    """Concatenate per-layer weights into one f32 vector, using a fixed layer order."""
    return np.concatenate([np.asarray(weights[name], dtype=np.float32) for name in order])


def _krum_scores(flat: np.ndarray, f: int) -> np.ndarray:
    """Per-client Krum score: sum of the (n-f-2) smallest squared distances to others.

    `flat` has shape (n, d). Returns a (n,) array of scores.
    """
    n = flat.shape[0]
    # Pairwise squared distances in f64 to bound accumulation error, then cast
    # back — the Rust kernel does the same.
    diff = flat[:, None, :].astype(np.float64) - flat[None, :, :].astype(np.float64)
    dist_sq = np.sum(diff * diff, axis=-1)  # (n, n), zeros on diagonal

    k = n - f - 2
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Drop self-distance (always zero but excluded by the paper's definition),
        # then sum the k smallest of the remaining n-1.
        others = np.concatenate([dist_sq[i, :i], dist_sq[i, i + 1 :]])
        others.sort()
        scores[i] = others[:k].sum()
    return scores


def krum_reference(
    updates: Sequence[ClientWeights],
    f: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """NumPy Krum: return the single-winner's weights plus its index.

    Raises ValueError if n < 2f + 3.
    """
    n = len(updates)
    if n < 2 * f + 3:
        raise ValueError(f"Krum requires n >= 2f+3; got n={n}, f={f}")

    order = list(updates[0]["weights"].keys())
    flat = np.stack([_flatten(u["weights"], order) for u in updates])
    scores = _krum_scores(flat, f)
    winner = int(np.argmin(scores))

    # Winner's per-layer weights, as f32 arrays (matches the Rust output dtype).
    weights = {
        name: np.asarray(updates[winner]["weights"][name], dtype=np.float32) for name in order
    }
    return weights, [winner]


def multi_krum_reference(
    updates: Sequence[ClientWeights],
    f: int,
    m: int | None = None,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """NumPy Multi-Krum: return the uniform mean of the m lowest-scoring updates.

    `m` defaults to `n - f` when None. Selected indices are returned sorted
    ascending so Python and Rust can compare them directly.

    Raises ValueError if n < 2f + 3 or m is out of [1, n-f].
    """
    n = len(updates)
    if n < 2 * f + 3:
        raise ValueError(f"MultiKrum requires n >= 2f+3; got n={n}, f={f}")

    m_eff = m if m is not None else n - f
    if m_eff < 1 or m_eff > n - f:
        raise ValueError(f"MultiKrum m must be in [1, n-f={n - f}]; got {m_eff}")

    order = list(updates[0]["weights"].keys())
    flat = np.stack([_flatten(u["weights"], order) for u in updates])
    scores = _krum_scores(flat, f)

    # np.argpartition gives us the m smallest in O(n), then sort the selection
    # so the client-id list is deterministic under ties.
    selected = np.argpartition(scores, m_eff - 1)[:m_eff]
    selected.sort()

    # Uniform mean per layer (Multi-Krum is *not* sample-weighted — deliberate:
    # Byzantine clients can lie about num_samples to amplify their pull).
    weights = {}
    for name in order:
        stacked = np.stack(
            [np.asarray(updates[i]["weights"][name], dtype=np.float32) for i in selected]
        )
        weights[name] = stacked.mean(axis=0).astype(np.float32)
    return weights, selected.tolist()
