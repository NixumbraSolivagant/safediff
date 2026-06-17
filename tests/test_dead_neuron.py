"""Tests for safediff.dead_neuron."""

from __future__ import annotations

import numpy as np

from safediff.dead_neuron import dead_count, dead_ratio, rank_by_deadness


def test_dead_ratio_full() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    assert dead_ratio(arr, eps=1e-6) == 1.0


def test_dead_ratio_empty() -> None:
    arr = np.zeros((0,), dtype=np.float32)
    assert dead_ratio(arr, eps=1e-6) == 0.0


def test_dead_ratio_half() -> None:
    arr = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert dead_ratio(arr, eps=1e-6) == 0.5


def test_dead_count() -> None:
    arr = np.array([0.0, 0.5, 1.0, 1e-9], dtype=np.float32)
    dead, total = dead_count(arr, eps=1e-6)
    assert dead == 2  # the zeros; 1e-9 is below 1e-6 threshold actually... wait
    # 0.0 and 1e-9 are both < 1e-6 -> 2 dead
    assert total == 4


def test_rank_by_deadness_sorts_descending() -> None:
    deltas = {
        "a": np.zeros((10,), dtype=np.float32),  # 100% dead
        "b": np.ones((10,), dtype=np.float32),  # 0% dead
        "c": np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),  # 25% dead
    }
    ranked = rank_by_deadness(deltas, eps=1e-6)
    assert [name for name, *_ in ranked] == ["a", "c", "b"]


def test_dead_eps_respected() -> None:
    arr = np.array([1e-9, 1e-3], dtype=np.float32)
    # With eps=1e-6, the first value is "dead"; with eps=1e-12, it isn't.
    assert dead_ratio(arr, eps=1e-6) == 0.5
    assert dead_ratio(arr, eps=1e-12) == 0.0
