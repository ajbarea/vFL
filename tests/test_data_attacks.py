"""Unit tests for the data-pipeline attack primitives.

Covers the contract :mod:`velocity.data_attacks` advertises:

* ``apply_label_flipping`` is a derangement of the label space — every
  class maps to a different class.
* ``apply_targeted_label_flipping`` flips exactly ``flip_ratio`` of the
  source-class samples, deterministic under ``seed``.
* ``make_label_flip_callback`` returns a closure with the "labels
  corrupted at rest" semantics — every batch sees the same flip rule,
  no per-batch randomness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from velocity.data_attacks import (  # noqa: E402
    DATA_ATTACK_TYPES,
    apply_label_flipping,
    apply_targeted_label_flipping,
    make_label_flip_callback,
)

# --------------------------------------------------------------------- types


def test_data_attack_types_contents():
    assert frozenset({"label_flipping", "targeted_label_flipping"}) == DATA_ATTACK_TYPES


# ----------------------------------------------------------- bijective flip


def test_apply_label_flipping_is_a_derangement():
    # Every class must map to a different class — no fixed points.
    labels = torch.arange(10)
    flipped = apply_label_flipping(labels, num_classes=10, seed=0)
    assert not torch.any(flipped == labels), (
        "apply_label_flipping left at least one class fixed: "
        f"labels={labels.tolist()}, flipped={flipped.tolist()}"
    )


def test_apply_label_flipping_preserves_shape_and_dtype():
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
    flipped = apply_label_flipping(labels, num_classes=5, seed=0)
    assert flipped.shape == labels.shape
    assert flipped.dtype == labels.dtype


def test_apply_label_flipping_is_deterministic_under_seed():
    labels = torch.arange(10).repeat(3)
    a = apply_label_flipping(labels, num_classes=10, seed=42)
    b = apply_label_flipping(labels, num_classes=10, seed=42)
    assert torch.equal(a, b)


def test_apply_label_flipping_is_bijective_function_of_class():
    # Same source class → same target class everywhere in the tensor
    # (the derangement is a function on the label space, not per-sample).
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    flipped = apply_label_flipping(labels, num_classes=3, seed=7)
    for c in range(3):
        targets = flipped[labels == c]
        if len(targets) > 0:
            assert torch.all(targets == targets[0]), (
                f"Class {c} mapped to multiple targets: {targets.tolist()}"
            )


def test_apply_label_flipping_handles_minimum_classes():
    # num_classes=2 → only valid derangement is the swap (0↔1).
    labels = torch.tensor([0, 1, 0, 1])
    flipped = apply_label_flipping(labels, num_classes=2, seed=0)
    assert torch.equal(flipped, torch.tensor([1, 0, 1, 0]))


def test_apply_label_flipping_rejects_too_few_classes():
    labels = torch.tensor([0, 0, 0])
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        apply_label_flipping(labels, num_classes=1, seed=0)


# ------------------------------------------------------------ targeted flip


def test_apply_targeted_label_flipping_full_ratio_flips_all_source():
    labels = torch.tensor([7, 7, 1, 7, 2, 7])
    flipped = apply_targeted_label_flipping(labels, source_class=7, target_class=1, flip_ratio=1.0)
    # All 7s become 1s; non-7s untouched.
    assert torch.equal(flipped, torch.tensor([1, 1, 1, 1, 2, 1]))


def test_apply_targeted_label_flipping_partial_ratio_count_correct():
    # 10 source-class samples, flip_ratio=0.4 → exactly 4 flipped.
    labels = torch.tensor([5] * 10 + [3] * 5)
    flipped = apply_targeted_label_flipping(
        labels, source_class=5, target_class=3, flip_ratio=0.4, seed=0
    )
    num_flipped = int((flipped[:10] == 3).sum().item())
    assert num_flipped == 4
    # Non-source samples untouched.
    assert torch.equal(flipped[10:], torch.tensor([3, 3, 3, 3, 3]))


def test_apply_targeted_label_flipping_zero_ratio_is_identity():
    labels = torch.tensor([5, 5, 5, 5])
    flipped = apply_targeted_label_flipping(labels, source_class=5, target_class=3, flip_ratio=0.0)
    assert torch.equal(flipped, labels)


def test_apply_targeted_label_flipping_no_source_in_batch_is_identity():
    labels = torch.tensor([1, 2, 3, 4])
    flipped = apply_targeted_label_flipping(labels, source_class=9, target_class=0, flip_ratio=1.0)
    assert torch.equal(flipped, labels)


def test_apply_targeted_label_flipping_rejects_bad_ratio():
    labels = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError, match="flip_ratio must be in"):
        apply_targeted_label_flipping(labels, source_class=1, target_class=2, flip_ratio=1.5)
    with pytest.raises(ValueError, match="flip_ratio must be in"):
        apply_targeted_label_flipping(labels, source_class=1, target_class=2, flip_ratio=-0.1)


# ------------------------------------------------------------- callback API


def test_make_label_flip_callback_untargeted_consistent_across_batches():
    # The "labels corrupted at rest" semantics: every batch the compromised
    # client trains on must see the same permutation.
    cb = make_label_flip_callback(num_classes=10, seed=0)
    batch1 = torch.tensor([3, 5, 7, 1])
    batch2 = torch.tensor([3, 5, 7, 1])
    assert torch.equal(cb(batch1), cb(batch2))


def test_make_label_flip_callback_untargeted_different_seeds_diverge():
    cb_a = make_label_flip_callback(num_classes=10, seed=0)
    cb_b = make_label_flip_callback(num_classes=10, seed=1)
    labels = torch.arange(10)
    # With high probability (true for these seeds) the two derangements
    # differ. If not, the test is meaningless — assert at least one diff.
    diff = (cb_a(labels) != cb_b(labels)).any()
    assert bool(diff)


def test_make_label_flip_callback_targeted_round_trip():
    cb = make_label_flip_callback(
        num_classes=10,
        targeted=True,
        source_class=9,
        target_class=1,
        flip_ratio=1.0,
        seed=0,
    )
    labels = torch.tensor([9, 0, 9, 5, 9])
    out = cb(labels)
    assert torch.equal(out, torch.tensor([1, 0, 1, 5, 1]))


def test_make_label_flip_callback_targeted_requires_classes():
    with pytest.raises(ValueError, match="targeted=True requires"):
        make_label_flip_callback(num_classes=10, targeted=True)


def test_callback_preserves_dtype_and_device():
    cb = make_label_flip_callback(num_classes=10, seed=0)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    out = cb(labels)
    assert out.dtype == labels.dtype
    assert out.device == labels.device
