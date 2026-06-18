"""Model Sanity Checker — static audit of a single checkpoint file.

Before loading a model into a GPU (or distributing it), this module scans it for
numerical problems that waste VRAM, cause NaN cascades, or trip up quantisation
pipelines:

1. **NaN / Inf**         — every tensor is searched for float('nan') or float('inf').
2. **Extreme outliers**   — values beyond ``k`` standard deviations from the mean.
3. **Near-zero layers**  — layers where >90% of weights are within eps of zero.
4. **Frozen layers**     — layers that are identical to the previous layer (copy-paste bug).
5. **Outlier distribution** — per-layer histogram summary for manual inspection.

All checks are pure numpy and fully offline — no GPU, no training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np


# ------------------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------------------

@dataclass
class OutlierReport:
    """Outlier statistics for a single tensor."""

    name: str
    shape: tuple[int, ...]
    numel: int
    has_nan: bool = False
    has_inf: bool = False
    has_neg_inf: bool = False
    has_pos_inf: bool = False
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0
    outlier_count: int = 0
    outlier_fraction: float = 0.0
    outlier_sigma: float = 5.0
    near_zero_fraction: float = 0.0
    is_unsafe: bool = False  # True if any serious issue found

    def severity(self) -> str:
        if self.has_nan or self.has_inf:
            return "critical"
        if self.outlier_fraction > 0.01:
            return "high"
        if self.outlier_fraction > 0:
            return "medium"
        if self.near_zero_fraction > 0.95:
            return "warning"
        return "ok"


@dataclass
class AuditReport:
    """Aggregated report for an entire checkpoint."""

    path: Path
    total_layers: int = 0
    total_params: int = 0
    nan_layers: list[OutlierReport] = field(default_factory=list)
    inf_layers: list[OutlierReport] = field(default_factory=list)
    outlier_layers: list[OutlierReport] = field(default_factory=list)
    near_zero_layers: list[OutlierReport] = field(default_factory=list)
    frozen_layers: list[tuple[str, str]] = field(default_factory=list)
    all_reports: list[OutlierReport] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return len(self.nan_layers) + len(self.inf_layers)

    @property
    def warning_count(self) -> int:
        return len(self.outlier_layers) + len(self.near_zero_layers)

    @property
    def is_healthy(self) -> bool:
        return self.critical_count == 0 and self.warning_count == 0

    def summary(self) -> str:
        parts = []
        if self.critical_count > 0:
            parts.append(f"[red]{self.critical_count} critical[/red]")
        if self.outlier_layers:
            parts.append(f"[yellow]{len(self.outlier_layers)} outlier[/yellow]")
        if self.near_zero_layers:
            parts.append(f"[yellow]{len(self.near_zero_layers)} near-zero[/yellow]")
        if self.frozen_layers:
            parts.append(f"[yellow]{len(self.frozen_layers)} frozen[/yellow]")
        if not parts:
            parts.append("[green]all clear[/green]")
        return ", ".join(parts)


# ------------------------------------------------------------------------------------------
# Core audit logic
# ------------------------------------------------------------------------------------------

def _inspect(arr: np.ndarray, name: str, outlier_sigma: float, near_zero_eps: float) -> OutlierReport:
    """Analyze a single tensor and return an OutlierReport.

    Outlier detection uses a hybrid heuristic to defeat the masking effect:

    1. If std is well-defined (not blown up by outliers), flag values > ``outlier_sigma``σ.
    2. **Also** flag values that are > ``outlier_sigma`` × MAD from the median — this
       is robust even when the std is inflated by a small cluster of extreme values.
    3. Either signal flags the value as an outlier (union).
    """
    numel = int(arr.size)
    if numel == 0:
        return OutlierReport(name=name, shape=arr.shape, numel=0)

    flat = arr.ravel()
    min_val = float(flat.min())
    max_val = float(flat.max())
    mean_val = float(flat.mean())
    std_val = float(flat.std())

    has_nan = bool(np.any(np.isnan(flat)))
    has_pos_inf = bool(np.any(np.isposinf(flat)))
    has_neg_inf = bool(np.any(np.isneginf(flat)))
    has_inf = has_pos_inf or has_neg_inf

    # Outlier detection — std-based
    if std_val > 1e-12:
        lower = mean_val - outlier_sigma * std_val
        upper = mean_val + outlier_sigma * std_val
        outlier_mask_std = (flat < lower) | (flat > upper)
    else:
        outlier_mask_std = np.zeros(flat.shape, dtype=bool)

    # Outlier detection — MAD-based (robust to the masking effect)
    finite = flat[np.isfinite(flat)]
    if finite.size > 0:
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        if mad > 1e-12:
            mad_lower = med - outlier_sigma * 1.4826 * mad
            mad_upper = med + outlier_sigma * 1.4826 * mad
            outlier_mask_mad = (flat < mad_lower) | (flat > mad_upper)
        else:
            outlier_mask_mad = np.zeros(flat.shape, dtype=bool)
    else:
        outlier_mask_mad = np.zeros(flat.shape, dtype=bool)

    outlier_mask = outlier_mask_std | outlier_mask_mad
    outlier_count = int(np.count_nonzero(outlier_mask))
    outlier_fraction = outlier_count / numel

    # Near-zero: values extremely close to zero (wasted parameters).
    near_zero_count = int(np.count_nonzero(np.abs(flat) < near_zero_eps))
    near_zero_fraction = near_zero_count / numel

    is_unsafe = has_nan or has_inf or outlier_fraction > 0.01

    return OutlierReport(
        name=name,
        shape=tuple(arr.shape),
        numel=numel,
        has_nan=has_nan,
        has_inf=has_inf,
        has_pos_inf=has_pos_inf,
        has_neg_inf=has_neg_inf,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        std_val=std_val,
        outlier_count=outlier_count,
        outlier_fraction=outlier_fraction,
        outlier_sigma=outlier_sigma,
        near_zero_fraction=near_zero_fraction,
        is_unsafe=is_unsafe,
    )


def _find_frozen_layers(
    tensors: dict[str, np.ndarray], rtol: float = 1e-5, atol: float = 1e-8
) -> list[tuple[str, str]]:
    """Find pairs of layers that are (almost) identical — likely a copy-paste bug."""
    names = sorted(tensors.keys())
    frozen: list[tuple[str, str]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            a = tensors[name_a]
            b = tensors[name_b]
            if a.shape != b.shape:
                continue
            if np.allclose(a, b, rtol=rtol, atol=atol):
                frozen.append((name_a, name_b))
    # Avoid flooding with O(n²) comparisons for large models: stop after 50 pairs.
    return frozen[: 50]


def audit(
    tensors: dict[str, np.ndarray],
    *,
    outlier_sigma: float = 5.0,
    near_zero_eps: float = 1e-6,
) -> AuditReport:
    """Run a full static audit on a checkpoint's tensors.

    Args:
        tensors: Output of ``load_tensors`` — ``dict[str, np.ndarray]``.
        outlier_sigma: Flag values beyond this many standard deviations as outliers.
        near_zero_eps: Flag a layer as near-zero if >90% of values fall within this.

    Returns:
        An ``AuditReport`` with categorized findings.
    """
    report = AuditReport(
        path=Path("<unknown>"),
        total_layers=len(tensors),
        total_params=int(sum(v.size for v in tensors.values())),
    )

    all_reports: list[OutlierReport] = []
    for name, arr in tensors.items():
        r = _inspect(arr, name, outlier_sigma, near_zero_eps)
        all_reports.append(r)

        if r.has_nan:
            report.nan_layers.append(r)
        if r.has_inf:
            report.inf_layers.append(r)
        if r.outlier_fraction > 0:
            report.outlier_layers.append(r)
        if r.near_zero_fraction > 0.90:
            report.near_zero_layers.append(r)

    report.all_reports = all_reports
    report.frozen_layers = _find_frozen_layers(tensors)

    # Sort outlier layers by outlier fraction descending.
    report.outlier_layers.sort(key=lambda r: r.outlier_fraction, reverse=True)
    report.near_zero_layers.sort(key=lambda r: r.near_zero_fraction, reverse=True)

    return report
