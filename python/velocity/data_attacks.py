"""Data-pipeline poisoning attacks for federated learning.

These attacks operate on labels in a client's data loader *before* training,
distinct from :mod:`velocity.attacks` (the Rust orchestrator's round-level
attacks: model_poisoning, sybil_nodes, gaussian_noise). The split is
deliberate — data attacks live in the client's data, round attacks live in
the server's view of weights/clients. They compose, but they're not the
same primitive.

Use :func:`make_label_flip_callback` to obtain a closure suitable for
``local_train(..., label_attack=callback)``: the closure rewrites the
labels of every minibatch the compromised client trains on, simulating a
worker whose data has been mislabeled at rest.

References:
    Biggio, Nelson, Laskov. *Poisoning Attacks against Support Vector
    Machines*. ICML 2012, pp. 1467-1474.
    https://icml.cc/2012/papers/880.pdf
        — Originator of the label-flipping data-poisoning primitive.

    Tolpegin, Truex, Gursoy, Liu. *Data Poisoning Attacks Against Federated
    Learning Systems*. ESORICS 2020.
    https://link.springer.com/chapter/10.1007/978-3-030-58951-6_24
        — Modern formulation in the FL setting; targeted vs. untargeted
          flips; demonstrates substantial accuracy drop with only a small
          fraction of malicious participants.
"""

from __future__ import annotations

from collections.abc import Callable

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.data_attacks requires PyTorch. Install with: pip install 'velocity-fl[torch]'"
    ) from exc

__all__ = [
    "DATA_ATTACK_TYPES",
    "apply_label_flipping",
    "apply_targeted_label_flipping",
    "make_label_flip_callback",
]

DATA_ATTACK_TYPES: frozenset[str] = frozenset({"label_flipping", "targeted_label_flipping"})


def apply_label_flipping(
    labels: Tensor,
    num_classes: int,
    *,
    seed: int | None = None,
) -> Tensor:
    """Bijectively permute every label so no class maps to itself.

    Generates a derangement of ``range(num_classes)`` (a permutation with
    no fixed points), then applies it elementwise to ``labels``. The
    enforcement loop fixes any incidental self-map by swapping with the
    next slot — for ``num_classes >= 2`` this always yields a valid
    derangement. Deterministic given ``seed``.

    This is the *untargeted bijective* flavor of label-flipping — the
    worst-case data-poisoning primitive when the attacker has no specific
    misclassification goal, only generic damage.

    Raises:
        ValueError: if ``num_classes < 2`` (no derangement exists).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2; got {num_classes}")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    perm = torch.randperm(num_classes, generator=gen).tolist()
    for i in range(num_classes):
        if perm[i] == i:
            j = (i + 1) % num_classes
            perm[i], perm[j] = perm[j], perm[i]

    perm_tensor = torch.tensor(perm, dtype=labels.dtype, device=labels.device)
    return perm_tensor[labels]


def apply_targeted_label_flipping(
    labels: Tensor,
    *,
    source_class: int,
    target_class: int,
    flip_ratio: float = 1.0,
    seed: int | None = None,
) -> Tensor:
    """Flip a fraction of ``source_class`` labels to ``target_class``.

    The classic targeted-misclassification primitive — e.g. ``9 -> 1`` to
    induce a specific confusion while leaving every other class intact.
    With ``flip_ratio == 1.0`` all source-class samples are flipped;
    smaller ratios sample uniformly without replacement (deterministic
    under ``seed``). When the input contains no source-class samples the
    output equals the input.

    Raises:
        ValueError: if ``flip_ratio`` is not in ``[0, 1]``.
    """
    if not 0.0 <= flip_ratio <= 1.0:
        raise ValueError(f"flip_ratio must be in [0, 1]; got {flip_ratio}")

    modified = labels.clone()
    source_mask = labels == source_class

    if flip_ratio < 1.0:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        num_source = int(source_mask.sum().item())
        num_to_flip = int(num_source * flip_ratio) if num_source > 0 else 0
        if num_to_flip > 0:
            indices = torch.where(source_mask)[0]
            chosen = indices[torch.randperm(len(indices), generator=gen)[:num_to_flip]]
            new_mask = torch.zeros_like(labels, dtype=torch.bool)
            new_mask[chosen] = True
            source_mask = new_mask
        else:
            source_mask = torch.zeros_like(labels, dtype=torch.bool)

    modified[source_mask] = target_class
    return modified


def make_label_flip_callback(
    num_classes: int,
    *,
    targeted: bool = False,
    source_class: int | None = None,
    target_class: int | None = None,
    flip_ratio: float = 1.0,
    seed: int = 0,
) -> Callable[[Tensor], Tensor]:
    """Return a closure suitable for ``local_train(label_attack=...)``.

    For a compromised client the attacker has corrupted the data labels
    *at rest*, so every minibatch sees the same flip rule. We pre-compute
    the rule once (the derangement permutation, or the source/target
    pair) and the returned closure just applies it per-batch with no
    further randomness.

    With ``targeted=False`` returns an untargeted bijective flipper;
    ``targeted=True`` requires both ``source_class`` and ``target_class``
    and returns a targeted flipper using ``flip_ratio``.

    Raises:
        ValueError: if ``targeted=True`` but ``source_class`` or
            ``target_class`` is missing.
    """
    if targeted:
        if source_class is None or target_class is None:
            raise ValueError("targeted=True requires both source_class and target_class")
        sc, tc, ratio, s = source_class, target_class, flip_ratio, seed

        def _targeted(labels: Tensor) -> Tensor:
            return apply_targeted_label_flipping(
                labels,
                source_class=sc,
                target_class=tc,
                flip_ratio=ratio,
                seed=s,
            )

        return _targeted

    # Untargeted: pre-generate a derangement so every batch sees the same
    # permutation. This matches the "labels corrupted at rest" semantics —
    # no per-batch randomness, no leakage between clients via the seed.
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(num_classes, generator=gen).tolist()
    for i in range(num_classes):
        if perm[i] == i:
            j = (i + 1) % num_classes
            perm[i], perm[j] = perm[j], perm[i]
    perm_t = torch.tensor(perm)

    def _bijective(labels: Tensor) -> Tensor:
        p = perm_t.to(dtype=labels.dtype, device=labels.device)
        return p[labels]

    return _bijective
