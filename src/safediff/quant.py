"""Quantisation pre-flight scanner.

This module is the SDK entry point for engineers who need to run a LLM through
a quantisation pipeline (AWQ, GPTQ, INT4, INT8, llama.cpp, vLLM, etc.) and
want to know **before** starting the slow quantisation process:

1. Which layers are likely to produce large quantization error.
2. Whether ``per-tensor`` or ``per-channel`` quantization is appropriate.
3. Which layers should be skipped entirely.
4. An overall "quantizability score" for the checkpoint.

The public API::

    from safediff.quant import analyze, suggest_scheme, QuantReport

    report = analyze("model.safetensors")
    for layer in report.layers:
        print(layer.name, layer.suggested_scheme, layer.error_estimate_8bit)

All checks are pure numpy — no GPU, no PyTorch at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# ------------------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------------------

SCHEME_LITERAL = Literal["per-tensor", "per-channel", "skip"]


@dataclass
class QuantScheme:
    """Quantisation scheme suggested for one tensor."""

    name: str
    shape: tuple[int, ...]
    bits: int
    suggested_scheme: SCHEME_LITERAL
    clip_min: float
    clip_max: float
    clip_ratio: float  # fraction of elements clipped on either side
    scale: float  # quantization scale
    zero_point: float  # zero-point for asymmetric (0.0 for symmetric)
    outlier_ratio: float  # fraction of elements treated as outliers
    error_estimate: float  # relative MSE vs. original (lower = better)
    reason: str  # human-readable justification

    def quality_score(self) -> float:
        """Return a 0-100 quality score (higher = better quantizability)."""
        score = 100.0
        if self.outlier_ratio > 0.05:
            score -= 40
        elif self.outlier_ratio > 0.01:
            score -= 20
        elif self.outlier_ratio > 0.001:
            score -= 5
        if self.clip_ratio > 0.01:
            score -= 15 * min(1.0, self.clip_ratio / 0.05)
        if self.error_estimate > 0.1:
            score -= 5 * min(1.0, self.error_estimate)
        return max(0.0, score)


@dataclass
class QuantLayerStat:
    """Full quantisation statistics for one named tensor."""

    name: str
    shape: tuple[int, ...]
    numel: int
    # Stats for each requested bit width
    schemes_4bit: QuantScheme | None = None
    schemes_8bit: QuantScheme | None = None
    # Chosen scheme at the default (lowest) bit width
    recommended: QuantScheme | None = None
    # Overall score at the lowest bit width (0-100)
    health_score: float = 100.0


@dataclass
class QuantReport:
    """Aggregated quantisation report for an entire checkpoint."""

    path: Path
    total_layers: int = 0
    total_params: int = 0
    layers: list[QuantLayerStat] = field(default_factory=list)
    # Aggregated summary
    healthy_count: int = 0  # score >= 80
    warning_count: int = 0  # score 50-80
    danger_count: int = 0  # score < 50
    skip_count: int = 0
    per_channel_count: int = 0
    worst_offender: str = ""
    overall_score: float = 100.0  # weighted by param count

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        parts = []
        if self.danger_count:
            parts.append(f"[red]{self.danger_count} dangerous[/red]")
        if self.warning_count:
            parts.append(f"[yellow]{self.warning_count} caution[/yellow]")
        if self.healthy_count:
            parts.append(f"[green]{self.healthy_count} healthy[/green]")
        if self.skip_count:
            parts.append(f"[dim]{self.skip_count} skip[/dim]")
        return ", ".join(parts) if parts else "[green]all clear[/green]"


# ------------------------------------------------------------------------------------------
# Core algorithm helpers
# ------------------------------------------------------------------------------------------

def _median_abs_deviation(values: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD) for a 1-D array.  Uses the 1.4826 scale factor."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return med, 1.4826 * mad


def _outlier_fraction(
    arr: np.ndarray,
    clip_min: float,
    clip_max: float,
    outlier_sigma: float = 5.0,
) -> tuple[float, float, float]:
    """Count elements outside the MAD-based safe range.

    Returns (outlier_fraction, mad_lower, mad_upper).
    Uses MAD so that extreme outliers cannot mask each other.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, float("-inf"), float("inf")

    med, scaled_mad = _median_abs_deviation(finite)
    if scaled_mad < 1e-12:
        # All values identical (or near-zero). Treat any non-zero value as outlier.
        if np.abs(finite).max() > 1e-12:
            return 1.0, float("-inf"), float("inf")
        return 0.0, med, med

    mad_lower = med - outlier_sigma * scaled_mad
    mad_upper = med + outlier_sigma * scaled_mad

    # Union of the hard clip range and the MAD-based safe range.
    # An outlier is one that is OUTSIDE both.
    effective_lower = max(clip_min, mad_lower)
    effective_upper = min(clip_max, mad_upper)
    outlier_mask = (arr < effective_lower) | (arr > effective_upper)
    return float(np.mean(outlier_mask)), mad_lower, mad_upper


def _symmetric_scale(arr: np.ndarray, bits: int) -> tuple[float, float, float]:
    """Symmetric per-tensor quantization: clip to [-max_abs, max_abs], scale = max_abs / (2^(bits-1) - 1).

    Returns (scale, clip_min, clip_max).
    """
    max_abs = float(np.abs(arr).max())
    if max_abs < 1e-30:
        return 1.0, 0.0, 0.0
    qmax = float(2 ** (bits - 1) - 1)
    scale = max_abs / qmax
    return scale, -max_abs, max_abs


def _asymmetric_scale(
    arr: np.ndarray, bits: int
) -> tuple[float, float, float, float]:
    """Asymmetric per-tensor quantization with zero-point.

    Returns (scale, zero_point, clip_min, clip_max).
    """
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-30:
        return 1.0, 0.0, mn, mx
    qmax = float(2**bits - 1)
    scale = (mx - mn) / qmax
    zero_point = int(round(-mn / scale))
    zero_point = max(0, min(int(qmax), zero_point))
    clip_min = -zero_point * scale
    clip_max = (qmax - zero_point) * scale
    return scale, float(zero_point), clip_min, clip_max


def _relative_mse(
    original: np.ndarray,
    clip_min: float,
    clip_max: float,
    scale: float,
    zero_point: float,
) -> float:
    """Compute relative MSE: MSE(orig, dequant(quant(orig))) / var(orig)."""
    flat = original.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0

    # Clip
    clipped = np.clip(finite, clip_min, clip_max)
    # Quantize
    if scale < 1e-30:
        return 0.0
    quantized = np.round((clipped - zero_point) / scale)
    # Dequantize
    dequantized = quantized * scale + zero_point

    mse = float(np.mean((finite - dequantized) ** 2))
    var = float(np.var(finite))
    if var < 1e-30:
        return 0.0
    return mse / var


def _per_channel_scheme(
    arr: np.ndarray,
    bits: int,
    channel_axis: int,
    outlier_sigma: float,
    scheme_name: str,
) -> QuantScheme | None:
    """Per-channel quantization: one scale per output channel (axis=0 for linear weights)."""
    shape = arr.shape
    if len(shape) < 2:
        return None

    n_channels = shape[channel_axis]
    if n_channels == 0:
        return None

    total_params = arr.size
    per_channel_params = total_params // n_channels

    # Estimate: one scale + one zero_point per channel
    estimated_overhead = n_channels * (1 + 1) * 4 / (total_params * 4 / bits)
    if estimated_overhead > 0.25 and bits == 4:
        return None  # overhead too high for per-channel at 4-bit (skip rather than degrade)

    qmax = float(2 ** (bits - 1) - 1) if scheme_name == "per-channel-sym" else float(2**bits - 1)

    # Aggregate stats across channels
    total_clip_ratio = 0.0
    total_outlier = 0.0
    total_error = 0.0
    weighted_error = 0.0
    scales: list[float] = []
    clip_min_overall = float("inf")
    clip_max_overall = float("-inf")

    for ch in range(n_channels):
        idx: list[slice | int] = [slice(None)] * len(shape)
        idx[channel_axis] = ch
        channel_data = arr[tuple(idx)]

        max_abs = float(np.abs(channel_data).max())
        if max_abs < 1e-30:
            scales.append(1.0)
            continue

        scale = max_abs / qmax
        scales.append(scale)
        clip_lo = -max_abs
        clip_hi = max_abs

        # Outlier fraction for this channel
        frac, _, _ = _outlier_fraction(channel_data, clip_lo, clip_hi, outlier_sigma)
        total_outlier += frac * per_channel_params

        # Clip ratio
        n_clipped = int(np.sum((channel_data < clip_lo) | (channel_data > clip_hi)))
        total_clip_ratio += n_clipped / total_params

        # Relative MSE
        mse = _relative_mse(channel_data, clip_lo, clip_hi, scale, 0.0)
        total_error += mse * per_channel_params
        weighted_error += mse * per_channel_params

        clip_min_overall = min(clip_min_overall, clip_lo)
        clip_max_overall = max(clip_max_overall, clip_hi)

    outlier_ratio = total_outlier / total_params
    clip_ratio = total_clip_ratio
    avg_scale = np.mean(scales) if scales else 1.0
    avg_error = total_error / n_channels

    # Pick representative scale (median) for the report
    repr_scale = float(np.median(scales)) if scales else 1.0
    return QuantScheme(
        name="<per-channel summary>",
        shape=shape,
        bits=bits,
        suggested_scheme="per-channel",
        clip_min=clip_min_overall,
        clip_max=clip_max_overall,
        clip_ratio=clip_ratio,
        scale=repr_scale,
        zero_point=0.0,
        outlier_ratio=outlier_ratio,
        error_estimate=avg_error,
        reason=(
            f"per-channel symmetric; "
            f"outlier_ratio={outlier_ratio:.4f}, "
            f"avg_error={avg_error:.4f}"
        ),
    )


# ------------------------------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------------------------------

def suggest_scheme(
    arr: np.ndarray,
    name: str,
    target_bits: int | list[int] = [4, 8],
    outlier_sigma: float = 5.0,
) -> dict[int, QuantScheme]:
    """Suggest quantization schemes for a single tensor.

    Args:
        arr: The weight tensor (numpy array).
        name: Layer name, for reporting.
        target_bits: Bit widths to evaluate. Default [4, 8].
        outlier_sigma: MAD threshold for outlier detection. Default 5.0 (NIST standard).

    Returns:
        A dict {bits: QuantScheme} for each requested bit width.
        The caller can inspect both to choose the right scheme.
    """
    shape = tuple(arr.shape)
    numel = int(arr.size)

    if numel == 0:
        return {
            b: QuantScheme(
                name=name,
                shape=shape,
                bits=b,
                suggested_scheme="skip",
                clip_min=0.0,
                clip_max=0.0,
                clip_ratio=0.0,
                scale=1.0,
                zero_point=0.0,
                outlier_ratio=0.0,
                error_estimate=0.0,
                reason="empty tensor",
            )
            for b in (target_bits if isinstance(target_bits, list) else [target_bits])
        }

    if isinstance(target_bits, int):
        target_bits = [target_bits]
    results: dict[int, QuantScheme] = {}

    for bits in target_bits:
        # --- Symmetric per-tensor (INT4/INT8 standard) ---
        scale, clip_min, clip_max = _symmetric_scale(arr, bits)
        outlier_ratio, _, _ = _outlier_fraction(arr, clip_min, clip_max, outlier_sigma)
        clip_count = int(np.sum((arr < clip_min) | (arr > clip_max)))
        clip_ratio = clip_count / numel
        rel_mse = _relative_mse(arr, clip_min, clip_max, scale, 0.0)

        # --- Determine scheme ---
        suggested: SCHEME_LITERAL
        reason: str

        if outlier_ratio <= 0.001 and clip_ratio <= 0.001:
            suggested = "per-tensor"
            reason = (
                f"clean distribution; outlier_ratio={outlier_ratio:.4f}, "
                f"clip_ratio={clip_ratio:.4f}, rel_MSE={rel_mse:.4f}"
            )
        elif outlier_ratio <= 0.01 and clip_ratio <= 0.005:
            suggested = "per-tensor"
            reason = (
                f"acceptable; outlier_ratio={outlier_ratio:.4f}, "
                f"clip_ratio={clip_ratio:.4f}, rel_MSE={rel_mse:.4f}"
            )
        elif outlier_ratio > 0.05:
            # Try per-channel
            if len(shape) >= 2:
                pc = _per_channel_scheme(arr, bits, channel_axis=0, outlier_sigma=outlier_sigma, scheme_name="per-channel-sym")
                if pc is not None and pc.outlier_ratio < 0.01:
                    results[bits] = pc
                    continue
            suggested = "skip"
            reason = (
                f"outlier_ratio={outlier_ratio:.4f} > 5%%; "
                f"per-channel not feasible or still too high; skip this layer for {bits}bit"
            )
        elif outlier_ratio > 0.001:
            # Moderate outliers — per-tensor may still be acceptable
            suggested = "per-tensor"
            reason = (
                f"moderate outliers; outlier_ratio={outlier_ratio:.4f}, "
                f"clip_ratio={clip_ratio:.4f}, rel_MSE={rel_mse:.4f}; "
                f"consider per-channel if accuracy degrades"
            )
        else:
            suggested = "per-tensor"
            reason = f"rel_MSE={rel_mse:.4f}"

        results[bits] = QuantScheme(
            name=name,
            shape=shape,
            bits=bits,
            suggested_scheme=suggested,
            clip_min=clip_min,
            clip_max=clip_max,
            clip_ratio=clip_ratio,
            scale=scale,
            zero_point=0.0,
            outlier_ratio=outlier_ratio,
            error_estimate=rel_mse,
            reason=reason,
        )

    return results


def analyze(
    path: str | Path,
    *,
    outlier_sigma: float = 5.0,
) -> QuantReport:
    """Run quantisation pre-flight analysis on a checkpoint file.

    Args:
        path: Path to a ``.safetensors``, ``.pt``, ``.pth``, or ``.bin`` checkpoint.
        outlier_sigma: MAD-based outlier threshold. 5.0 is the NIST standard.

    Returns:
        A ``QuantReport`` with per-layer ``QuantLayerStat`` objects,
        sorted by decreasing danger (worst offenders first).
    """
    from safediff.loader import load_tensors

    p = Path(path)
    tensors = load_tensors(str(p))

    layers: list[QuantLayerStat] = []
    total_params = int(sum(v.size for v in tensors.values()))
    weighted_score_sum = 0.0

    for name, arr in tensors.items():
        schemes = suggest_scheme(arr, name, target_bits=[4, 8], outlier_sigma=outlier_sigma)
        s4 = schemes.get(4)
        s8 = schemes.get(8)

        # Recommended: prefer 4-bit if it's healthy enough, otherwise 8-bit
        if s4 is not None and s4.quality_score() >= 60.0:
            recommended = s4
        elif s8 is not None:
            recommended = s8
        elif s4 is not None:
            recommended = s4
        else:
            recommended = None

        health = recommended.quality_score() if recommended else 0.0

        stat = QuantLayerStat(
            name=name,
            shape=tuple(arr.shape),
            numel=int(arr.size),
            schemes_4bit=s4,
            schemes_8bit=s8,
            recommended=recommended,
            health_score=health,
        )
        layers.append(stat)

        # Weighted by param count
        weighted_score_sum += health * arr.size

    # Sort: worst offenders first
    layers.sort(key=lambda s: s.health_score)

    # Summary counts
    healthy_count = sum(1 for s in layers if s.health_score >= 80)
    warning_count = sum(1 for s in layers if 50 <= s.health_score < 80)
    danger_count = sum(1 for s in layers if s.health_score < 50)
    skip_count = sum(
        1 for s in layers
        if s.recommended is not None and s.recommended.suggested_scheme == "skip"
    )
    per_channel_count = sum(
        1 for s in layers
        if s.recommended is not None and s.recommended.suggested_scheme == "per-channel"
    )
    worst = layers[0].name if layers else ""

    overall_score = (weighted_score_sum / total_params) if total_params > 0 else 100.0

    return QuantReport(
        path=p,
        total_layers=len(layers),
        total_params=total_params,
        layers=layers,
        healthy_count=healthy_count,
        warning_count=warning_count,
        danger_count=danger_count,
        skip_count=skip_count,
        per_channel_count=per_channel_count,
        worst_offender=worst,
        overall_score=overall_score,
    )
