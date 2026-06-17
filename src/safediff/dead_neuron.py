"""Dead-neuron detection helpers.

A "dead" parameter is one whose absolute delta between two checkpoints is
below ``eps``. This module exposes a small functional API that the analyzer
delegates to, plus helpers for ranking layers by deadness.
"""

from __future__ import annotations

import numpy as np


def dead_ratio(delta: np.ndarray, eps: float = 1e-6) -> float:
    """Return the fraction of elements with ``|delta| < eps``.

    ``delta`` is flattened internally, so this works for tensors of any rank.
    """
    if delta.size == 0:
        return 0.0
    return float(np.mean(np.abs(delta.ravel()) < eps))


def dead_count(delta: np.ndarray, eps: float = 1e-6) -> tuple[int, int]:
    """Return ``(num_dead, total)`` for ``delta``."""
    total = int(delta.size)
    if total == 0:
        return 0, 0
    num_dead = int(np.count_nonzero(np.abs(delta.ravel()) < eps))
    return num_dead, total


def rank_by_deadness(
    deltas: dict[str, np.ndarray], eps: float = 1e-6
) -> list[tuple[str, float, int, int]]:
    """Rank tensors by descending dead-ration, returning ``(name, ratio, dead, total)``."""
    out = []
    for name, arr in deltas.items():
        dead, total = dead_count(arr, eps=eps)
        ratio = dead / total if total else 0.0
        out.append((name, ratio, dead, total))
    out.sort(key=lambda t: t[1], reverse=True)
    return out
