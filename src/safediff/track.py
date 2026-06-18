"""Learning Dynamics Tracker — trace weight evolution across a sequence of checkpoints.

Given a directory of checkpoints, the tracker:
1. Discovers and sorts them chronologically (by mtime, filename pattern, or embedded step).
2. Treats the first checkpoint as baseline.
3. For every other checkpoint, computes per-layer delta statistics from the baseline
   and from the immediately preceding checkpoint.
4. Automatically detects which layer first started diverging (anomalous drift before
   the overall loss spike became visible).
5. Renders a per-layer time-series table and a terminal trend chart.

All analysis is pure numpy — no torch / GPU dependency at runtime.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np


# ------------------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    """Metadata for one checkpoint in the tracking sequence."""

    path: Path
    step: int  # chronological index (0 = oldest, N-1 = newest)
    label: str  # display label, e.g. "epoch_10" or "step_500"
    mtime: float  # modification timestamp for sorting fallback
    _has_step: bool = field(default=True, repr=False)  # private: was step extracted?


@dataclass
class LayerSnapshot:
    """Diff statistics for a single layer at a single checkpoint."""

    checkpoint_step: int
    l2_norm: float
    max_abs: float
    mean: float
    std: float
    dead_fraction: float
    # Delta from the immediately previous checkpoint (not baseline)
    incremental_l2: float
    incremental_max_abs: float


@dataclass
class LayerSeries:
    """Full time-series of statistics for one named layer."""

    name: str
    shape: tuple[int, ...]
    snapshots: list[LayerSnapshot] = field(default_factory=list)

    def l2_values(self) -> list[float]:
        return [s.l2_norm for s in self.snapshots]

    def incr_l2_values(self) -> list[float]:
        return [s.incremental_l2 for s in self.snapshots]

    def incr_max_values(self) -> list[float]:
        return [s.incremental_max_abs for s in self.snapshots]

    def dead_values(self) -> list[float]:
        return [s.dead_fraction for s in self.snapshots]


# ------------------------------------------------------------------------------------------
# Divergence detection — modified Z-score using Median Absolute Deviation (MAD)
# https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
# ------------------------------------------------------------------------------------------

def modified_zscore(value: float, values: list[float]) -> float:
    """Modified z-score using Median Absolute Deviation. Robust to outliers.

    Uses the Iglewicz–Hoaglin recommendation of MAD = median(|x - median(x)|)
    with the 1.4826 scale factor so the result is comparable to a standard
    z-score under a normal distribution. Returns 0.0 when MAD is effectively
    zero (i.e. almost all values are identical) — this means "no drift
    detectable from this distribution".
    """
    if len(values) < 2:
        return 0.0
    med = float(np.median(values))
    deviations = [abs(v - med) for v in values]
    mad = float(np.median(deviations))
    if mad < 1e-12:
        # All values identical — but if this one differs, it's an extreme outlier.
        if abs(value - med) > 1e-12:
            return float("inf")
        return 0.0
    return abs(value - med) / (1.4826 * mad)


@dataclass
class DivergenceAlert:
    """A layer flagged as diverging, with evidence."""

    layer_name: str
    first_drift_step: int  # checkpoint step when divergence was first detected
    first_drift_incr_l2: float  # its incremental L2 at that step
    modified_zscore: float  # how extreme this drift was
    l2_trend: list[float]  # full incremental L2 history for context


# ------------------------------------------------------------------------------------------
# Checkpoint sorting utilities
# ------------------------------------------------------------------------------------------

# Common checkpoint filename patterns, in priority order.
_STEP_PATTERNS = [
    (re.compile(r"(?:epoch[-_]?)(\d+)"), r"epoch\1"),
    (re.compile(r"(?:step[-_]?)(\d+)"), r"step\1"),
    (re.compile(r"(?:ckpt[-_]?)(\d+)"), r"ckpt\1"),
    (re.compile(r"(?:checkpoint[-_]?)(\d+)"), r"ckpt\1"),
    (re.compile(r"(\d{4,})"), r"\1"),  # 5+ digit numbers often are global steps
]


def _extract_step_from_name(name: str) -> int | None:
    """Try to extract a numeric step from a filename like 'epoch_10.safetensors'."""
    for pattern, _ in _STEP_PATTERNS:
        m = pattern.search(name)
        if m:
            return int(m.group(1))
    return None


def _make_label(path: Path, step: int) -> str:
    # Prefer a human-readable tag extracted from the filename.
    for pattern, template in _STEP_PATTERNS:
        m = pattern.search(path.stem)
        if m:
            tag = template.replace(r"\1", m.group(1))
            return tag
    return f"step_{step}"


def discover_checkpoints(
    directory: Path,
    extensions: tuple[str, ...] = (".safetensors", ".pt", ".pth", ".bin"),
) -> list[CheckpointInfo]:
    """Find all checkpoint files in ``directory`` and return them sorted chronologically.

    Sorting priority:
    1. Numeric step extracted from filename (e.g. epoch_10 → 10)
    2. Modification time (mtime) as tiebreaker
    3. Lexicographic path as final tiebreaker

    Raises ``FileNotFoundError`` if the directory doesn't exist.
    Raises ``ValueError`` if fewer than 2 checkpoints are found.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    raw: list[CheckpointInfo] = []
    for ext in extensions:
        for p in directory.glob(f"*{ext}"):
            extracted = _extract_step_from_name(p.name)
            raw.append(
                CheckpointInfo(
                    path=p,
                    step=extracted if extracted is not None else 0,
                    label=_make_label(p, extracted),
                    mtime=p.stat().st_mtime,
                    _has_step=extracted is not None,
                )
            )

    if not raw:
        raise ValueError(
            f"No checkpoint files found in {directory}. "
            f"Supported: {extensions}"
        )

    # Sort: extracted steps first (ascending), then unknown-order files by mtime + name.
    raw.sort(key=lambda c: (not c._has_step, c.step, c.mtime, c.path.name))

    # Re-assign sequential step indices to files that had no extracted number,
    # preserving their relative order.
    unknown_idx = 0
    for c in raw:
        if not c._has_step:
            c.step = unknown_idx
            c.label = _make_label(c.path, unknown_idx)
            unknown_idx += 1

    # Clean up private attribute before returning
    for c in raw:
        del c._has_step  # type: ignore[attr-defined]

    return raw


# ------------------------------------------------------------------------------------------
# Core tracking logic
# ------------------------------------------------------------------------------------------

def _compute_snapshot(
    delta: np.ndarray, prev_delta: np.ndarray | None, dead_eps: float
) -> tuple[float, float, float, float, float, float, float]:
    """Compute all scalar statistics for one delta array."""
    abs_delta = np.abs(delta)
    numel = delta.size
    if numel == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    max_abs = float(abs_delta.max())
    l2_norm = float(np.linalg.norm(delta))
    mean = float(delta.mean())
    std = float(delta.std())
    dead_fraction = float(np.mean(abs_delta < dead_eps))

    if prev_delta is not None:
        incr_l2 = float(np.linalg.norm(delta - prev_delta))
        incr_max = float(np.abs(delta - prev_delta).max())
    else:
        incr_l2 = l2_norm
        incr_max = max_abs

    return l2_norm, max_abs, mean, std, dead_fraction, incr_l2, incr_max


def track(
    checkpoints: list[CheckpointInfo],
    loader,  # callable: Path -> dict[str, np.ndarray]
    dead_eps: float = 1e-6,
    anomaly_threshold: float = 3.5,
) -> tuple[dict[str, LayerSeries], list[DivergenceAlert]]:
    """Build per-layer time-series from a chronologically ordered list of checkpoints.

    Args:
        checkpoints: Output of ``discover_checkpoints``, already sorted.
        loader: A callable that takes a ``Path`` and returns ``dict[str, np.ndarray]``.
               The CLI passes ``load_tensors`` from the loader module.
        dead_eps: Threshold for "dead parameter" classification.
        anomaly_threshold: Modified z-score threshold for divergence detection.
                          3.5 is the NIST recommended value.

    Returns:
        (layer_series, alerts) where:
        - layer_series: {layer_name: LayerSeries}
        - alerts: list of DivergenceAlert, sorted by first_drift_step
    """
    if len(checkpoints) < 2:
        raise ValueError(
            "At least 2 checkpoints are required to track dynamics. "
            f"Got {len(checkpoints)} in {checkpoints[0].path.parent}"
        )

    # Load all checkpoints into memory (they're numpy arrays; for huge models
    # a streaming pass could be added later).
    checkpoint_data: list[dict[str, np.ndarray]] = []
    for ckpt in checkpoints:
        checkpoint_data.append(loader(ckpt.path))

    # Collect the union of all layer names across checkpoints.
    all_layer_names: set[str] = set()
    for data in checkpoint_data:
        all_layer_names.update(data.keys())

    # Baseline: first checkpoint.
    base = checkpoint_data[0]

    # previous delta (for incremental stats)
    prev_delta_map: dict[str, np.ndarray | None] = {name: None for name in all_layer_names}

    # Accumulate series for every layer.
    layer_series: dict[str, LayerSeries] = {
        name: LayerSeries(name=name, shape=base[name].shape if name in base else ())
        for name in all_layer_names
    }

    # Incremental L2 history per layer (for divergence detection).
    incr_l2_history: dict[str, list[float]] = defaultdict(list)

    for step_idx, data in enumerate(checkpoint_data):
        for name in all_layer_names:
            if name not in data:
                continue

            arr = data[name]
            base_arr = base[name]
            shape = arr.shape

            # Ensure same shape for delta computation.
            if arr.shape != base_arr.shape:
                snapshot = LayerSnapshot(
                    checkpoint_step=step_idx,
                    l2_norm=float("inf"),
                    max_abs=float("inf"),
                    mean=float("nan"),
                    std=float("nan"),
                    dead_fraction=0.0,
                    incremental_l2=float("inf"),
                    incremental_max_abs=float("inf"),
                )
                layer_series[name].snapshots.append(snapshot)
                layer_series[name].shape = shape
                continue

            delta = arr - base_arr
            prev_delta = prev_delta_map.get(name)
            incr_delta = (delta - prev_delta) if prev_delta is not None else delta

            (
                l2_norm, max_abs, mean, std, dead_fraction, incr_l2, incr_max
            ) = _compute_snapshot(delta, prev_delta, dead_eps)

            snapshot = LayerSnapshot(
                checkpoint_step=step_idx,
                l2_norm=l2_norm,
                max_abs=max_abs,
                mean=mean,
                std=std,
                dead_fraction=dead_fraction,
                incremental_l2=incr_l2,
                incremental_max_abs=incr_max,
            )
            layer_series[name].snapshots.append(snapshot)
            incr_l2_history[name].append(incr_l2)
            prev_delta_map[name] = delta

    # --- Divergence detection: modified Z-score on incremental L2 ---
    alerts: list[DivergenceAlert] = []
    for name, history in incr_l2_history.items():
        if len(history) < 3:  # Need at least 3 points for MAD-based detection
            continue
        for i, value in enumerate(history):
            zscore = modified_zscore(value, history[:i] if i > 0 else history)
            if zscore > anomaly_threshold:
                alerts.append(
                    DivergenceAlert(
                        layer_name=name,
                        first_drift_step=i,
                        first_drift_incr_l2=value,
                        modified_zscore=zscore,
                        l2_trend=history,
                    )
                )
                break  # Only flag the first time a layer diverges

    # Sort by first drift step ascending (earliest drift first).
    alerts.sort(key=lambda a: a.first_drift_step)
    return layer_series, alerts


# ------------------------------------------------------------------------------------------
# Helpers for visualizer
# ------------------------------------------------------------------------------------------

def top_layers_by_drift(
    layer_series: dict[str, LayerSeries], metric: str = "cumulative_l2", top: int = 20
) -> list[tuple[str, float, list[float]]]:
    """Return top-N layers by the requested metric.

    Args:
        metric: "cumulative_l2" (L2 from baseline) or "incremental_l2" (per-step drift)
    """
    results = []
    for name, series in layer_series.items():
        if not series.snapshots:
            continue
        if metric == "cumulative_l2":
            final_value = series.snapshots[-1].l2_norm
        else:
            # Peak incremental L2 across all steps
            final_value = max(s.incremental_l2 for s in series.snapshots)
        results.append((name, final_value, series.l2_values()))
    results.sort(key=lambda t: t[1], reverse=True)
    return results[:top]


def normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list to [0, 1]. Returns [0]*len if all identical."""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-12:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ------------------------------------------------------------------------------------------
# Public API for external integrations (e.g. the TrainerTracker class)
# ------------------------------------------------------------------------------------------

def compute_delta(
    current: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    prev_delta: dict[str, np.ndarray] | None,
    dead_eps: float = 1e-6,
) -> tuple[dict[str, LayerSnapshot], dict[str, np.ndarray]]:
    """Compute per-layer incremental deltas from a baseline.

    This is the core delta engine extracted from ``track()`` for use by
    integrations that manage their own state and don't need CLI I/O.

    Args:
        current: Current layer weights as ``{name: np.ndarray}``.
        baseline: Baseline (reference) layer weights.  Values must have the same
            shape as those in ``current``.
        prev_delta: Previous delta map from the last call, or ``None`` for the
            first call.  Use the returned new delta map on the next call.
        dead_eps: Threshold for "dead parameter" classification.

    Returns:
        (snapshots, new_delta_map) where:
        - snapshots: {name: LayerSnapshot} for this step
        - new_delta_map: {name: np.ndarray} — pass as ``prev_delta`` on next call
    """
    if prev_delta is None:
        prev_delta = {name: None for name in baseline}

    snapshots: dict[str, LayerSnapshot] = {}
    new_delta_map: dict[str, np.ndarray] = {}

    for name in baseline:
        if name not in current:
            continue

        arr = current[name]
        base_arr = baseline[name]

        if arr.shape != base_arr.shape:
            snapshot = LayerSnapshot(
                checkpoint_step=-1,
                l2_norm=float("inf"),
                max_abs=float("inf"),
                mean=float("nan"),
                std=float("nan"),
                dead_fraction=0.0,
                incremental_l2=float("inf"),
                incremental_max_abs=float("inf"),
            )
            snapshots[name] = snapshot
            new_delta_map[name] = base_arr  # cannot compute delta
            continue

        delta = arr - base_arr
        prev = prev_delta.get(name)
        incr_delta = (delta - prev) if prev is not None else delta

        abs_delta = np.abs(delta)
        abs_incr = np.abs(incr_delta)
        numel = delta.size

        l2_norm = float(np.linalg.norm(delta)) if numel > 0 else 0.0
        max_abs = float(abs_delta.max()) if numel > 0 else 0.0
        mean_val = float(delta.mean()) if numel > 0 else 0.0
        std_val = float(delta.std()) if numel > 0 else 0.0
        dead_fraction = float(np.mean(abs_delta < dead_eps)) if numel > 0 else 0.0
        incr_l2 = float(np.linalg.norm(incr_delta)) if numel > 0 else 0.0
        incr_max = float(abs_incr.max()) if numel > 0 else 0.0

        snapshots[name] = LayerSnapshot(
            checkpoint_step=-1,
            l2_norm=l2_norm,
            max_abs=max_abs,
            mean=mean_val,
            std=std_val,
            dead_fraction=dead_fraction,
            incremental_l2=incr_l2,
            incremental_max_abs=incr_max,
        )
        new_delta_map[name] = delta

    return snapshots, new_delta_map


def anomaly_score(
    history: list[float],
    value: float,
    threshold: float = 3.5,
) -> float:
    """Return the modified Z-score for ``value`` against ``history``.

    Wrapper around ``modified_zscore`` for discoverability in the public API.
    Returns the Z-score; the caller decides whether to act on it.

    Args:
        history: Prior values (must have at least 2 elements).
        value: New observation to score.
        threshold: Threshold above which the value is considered anomalous.
                   Defaults to 3.5 (NIST recommended).

    Returns:
        Modified Z-score.  Values > ``threshold`` indicate divergence.
    """
    return modified_zscore(value, history)
