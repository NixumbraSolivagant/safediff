"""Core weight-diff analysis.

Given two ``dict[str, np.ndarray]`` state-dicts, compute per-tensor statistics
and flag anomalous layers. Pure numpy — no torch / GPU dependency at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

# Statistic definitions. Centralized so visualizer and JSON exporter stay in sync.
STAT_FIELDS = ("max_abs", "l2_norm", "mean", "std")


@dataclass
class LayerStat:
    """Per-tensor diff statistics. Mutable so the analyzer can flag anomalies in place."""

    name: str
    shape: tuple[int, ...]
    numel: int
    max_abs: float
    l2_norm: float
    mean: float
    std: float
    is_anomaly: bool = False
    is_dead_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "max_abs": self.max_abs,
            "l2_norm": self.l2_norm,
            "mean": self.mean,
            "std": self.std,
            "is_anomaly": self.is_anomaly,
            "dead_fraction": self.is_dead_fraction,
        }


@dataclass
class DiffReport:
    """Full diff report for a pair of checkpoints."""

    common: list[LayerStat] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    total_params_a: int = 0
    total_params_b: int = 0

    @property
    def anomalies(self) -> list[LayerStat]:
        return [s for s in self.common if s.is_anomaly]

    def filter(self, pattern: str | None) -> DiffReport:
        """Return a new report keeping only layers whose name matches ``pattern``.

        Glob-style: ``*`` is a wildcard, ``.`` is literal.
        """
        if not pattern:
            return self
        import fnmatch

        self.common = [s for s in self.common if fnmatch.fnmatch(s.name, pattern)]
        return self

    def head(self, n: int) -> DiffReport:
        self.common = self.common[:n]
        return self


def _stats_for_delta(name: str, delta: np.ndarray, dead_eps: float) -> LayerStat:
    abs_delta = np.abs(delta)
    numel = int(delta.size)
    return LayerStat(
        name=name,
        shape=tuple(delta.shape),
        numel=numel,
        max_abs=float(abs_delta.max()) if numel else 0.0,
        l2_norm=float(np.linalg.norm(delta)) if numel else 0.0,
        mean=float(delta.mean()) if numel else 0.0,
        std=float(delta.std()) if numel else 0.0,
        is_dead_fraction=float(np.mean(abs_delta < dead_eps)) if numel else 0.0,
    )


def _flag_anomalies(
    stats: list[LayerStat], anomaly_threshold: float
) -> list[LayerStat]:
    """Mark layers whose L2 norm exceeds the global median by ``anomaly_threshold``x
    or whose ``max_abs`` exceeds 1.0 in fp32 (a heuristic for numerical blowup).
    """
    if not stats:
        return stats
    l2s = np.array([s.l2_norm for s in stats], dtype=np.float64)
    # Median is robust to the very outliers we are trying to find.
    median = float(np.median(l2s)) if l2s.size else 0.0
    upper = median * anomaly_threshold
    for s in stats:
        s.is_anomaly = bool(
            (upper > 0.0 and s.l2_norm > upper) or s.max_abs > 1.0
        )
    return stats


def analyze(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    *,
    dead_eps: float = 1e-6,
    anomaly_threshold: float = 10.0,
) -> DiffReport:
    """Compute a ``DiffReport`` comparing state-dicts ``a`` and ``b``."""
    report = DiffReport(
        total_params_a=int(sum(v.size for v in a.values())),
        total_params_b=int(sum(v.size for v in b.values())),
    )
    common_keys: Iterable[str] = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    report.only_in_a = only_a
    report.only_in_b = only_b

    for key in common_keys:
        arr_a = np.asarray(a[key])
        arr_b = np.asarray(b[key])
        if arr_a.shape != arr_b.shape:
            # Shape mismatch is itself an anomaly worth surfacing.
            stat = LayerStat(
                name=key,
                shape=arr_b.shape,
                numel=int(arr_b.size),
                max_abs=float("inf"),
                l2_norm=float("inf"),
                mean=float("nan"),
                std=float("nan"),
                is_anomaly=True,
                is_dead_fraction=0.0,
            )
        else:
            delta = arr_b - arr_a
            stat = _stats_for_delta(key, delta, dead_eps=dead_eps)
        report.common.append(stat)

    # Sort by L2 norm descending — the "biggest movers" float to the top.
    report.common.sort(key=lambda s: s.l2_norm, reverse=True)
    _flag_anomalies(report.common, anomaly_threshold=anomaly_threshold)
    return report
