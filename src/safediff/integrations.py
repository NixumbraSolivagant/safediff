"""Training-loop integration: lightweight, zero-IO weight tracker.

This module exposes a ``Tracker`` class that can be embedded inside a training loop.
Instead of checkpoint files on disk, it receives numpy state dicts directly,
computes incremental deltas in-memory, and optionally pushes per-layer drift metrics
to a logger (wandb, tensorboard, or any callable).

Usage::

    import safediff as sd

    tracker = sd.Tracker(
        baseline_state_dict={k: v.cpu().numpy() for k, v in model.state_dict().items()},
        step_id=0,
        logger=wandb.run,          # optional
        anomaly_threshold=3.5,
        log_every=100,
    )

    # Inside training loop:
    for step in range(1, 10001):
        model.train_step()  # your training logic
        alert = tracker.update(
            {k: v.cpu().numpy() for k, v in model.state_dict().items()},
            step_id=step,
        )
        if alert:
            print(f"Drift detected in {alert.layer_name} at step {step}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from safediff.track import (
    CheckpointInfo,
    DivergenceAlert,
    LayerSeries,
    LayerSnapshot,
    modified_zscore,
)


@dataclass
class _InMemorySnapshot:
    """Lightweight snapshot stored in-memory by the Tracker."""
    step_id: int
    snapshots: dict[str, LayerSnapshot]


class Tracker:
    """Embeddable per-layer weight drift tracker.

    Call ``update(state_dict, step_id)`` once per training step (or at whatever
    frequency you want to observe).  The Tracker:

    1. Computes per-layer incremental L2 deltas in memory (zero disk IO).
    2. Maintains a per-layer incremental-L2 history.
    3. Detects divergence via MAD-based modified Z-score.
    4. Optionally logs per-layer metrics to ``logger`` at ``log_every`` intervals.

    Args:
        baseline_state_dict: The reference state dict.  All future deltas are
            relative to this.  Values must be numpy arrays.
        step_id: The initial step number corresponding to ``baseline_state_dict``.
        logger: Any object with a ``.log(dict, step=int)`` method
            (wandb.Run, tensorboard SummaryWriter, etc.) or a plain callable.
            If ``None``, logging is disabled.
        anomaly_threshold: Modified Z-score threshold for divergence detection.
            3.5 is the NIST recommended value.
        log_every: Push metrics to ``logger`` every N calls to ``update()``.
            Set to 0 to disable.
        dead_eps: Threshold for "dead parameter" classification.
    """

    def __init__(
        self,
        baseline_state_dict: dict[str, np.ndarray],
        step_id: int,
        logger: Any | None = None,
        anomaly_threshold: float = 3.5,
        log_every: int = 100,
        dead_eps: float = 1e-6,
    ) -> None:
        self._baseline: dict[str, np.ndarray] = dict(baseline_state_dict)
        self._anomaly_threshold = anomaly_threshold
        self._log_every = log_every
        self._dead_eps = dead_eps
        self._logger = logger

        # Per-layer incremental-L2 history for MAD-based anomaly detection.
        self._incr_l2_history: dict[str, list[float]] = {
            name: [] for name in baseline_state_dict
        }

        # Previous delta map (for incremental delta computation).
        self._prev_delta: dict[str, np.ndarray | None] = {
            name: None for name in baseline_state_dict
        }

        # Layer series (cumulative snapshots).
        self._layer_series: dict[str, LayerSeries] = {
            name: LayerSeries(name=name, shape=tuple(arr.shape))
            for name, arr in baseline_state_dict.items()
        }

        # Baseline is stored in snapshots; we also add a zero-snapshot to layer_series
        # so that series.snapshots covers every recorded step (baseline + every update).
        for name in baseline_state_dict:
            self._layer_series[name].snapshots.append(
                LayerSnapshot(
                    checkpoint_step=step_id,
                    l2_norm=0.0,
                    max_abs=0.0,
                    mean=0.0,
                    std=0.0,
                    dead_fraction=1.0,
                    incremental_l2=0.0,
                    incremental_max_abs=0.0,
                )
            )

        # In-memory snapshots of every recorded step (including baseline).
        self._snapshots: list[_InMemorySnapshot] = []
        self._snapshots.append(
            _InMemorySnapshot(
                step_id=step_id,
                snapshots={
                    name: LayerSnapshot(
                        checkpoint_step=step_id,
                        l2_norm=0.0,
                        max_abs=0.0,
                        mean=0.0,
                        std=0.0,
                        dead_fraction=1.0,
                        incremental_l2=0.0,
                        incremental_max_abs=0.0,
                    )
                    for name, arr in baseline_state_dict.items()
                },
            )
        )

        # Counter for log_every gating.
        self._update_count = 0

    def update(
        self,
        state_dict: dict[str, np.ndarray],
        step_id: int,
    ) -> list[DivergenceAlert]:
        """Update the tracker with the current model weights.

        Args:
            state_dict: Current model weights as ``{name: np.ndarray}``.
            step_id: Current training step / epoch.

        Returns:
            List of new ``DivergenceAlert`` instances since the last call.
            Empty list if no new divergences were detected.
        """
        new_alerts: list[DivergenceAlert] = []
        current_snapshots: dict[str, LayerSnapshot] = {}

        for name in self._baseline:
            if name not in state_dict:
                continue

            arr = state_dict[name]
            base_arr = self._baseline[name]

            if arr.shape != base_arr.shape:
                snapshot = LayerSnapshot(
                    checkpoint_step=step_id,
                    l2_norm=float("inf"),
                    max_abs=float("inf"),
                    mean=float("nan"),
                    std=float("nan"),
                    dead_fraction=0.0,
                    incremental_l2=float("inf"),
                    incremental_max_abs=float("inf"),
                )
                current_snapshots[name] = snapshot
                self._layer_series[name].snapshots.append(snapshot)
                continue

            delta = arr - base_arr
            prev_delta = self._prev_delta.get(name)

            if prev_delta is not None:
                incr_delta = delta - prev_delta
            else:
                incr_delta = delta

            abs_delta = np.abs(delta)
            abs_incr = np.abs(incr_delta)
            numel = delta.size

            l2_norm = float(np.linalg.norm(delta)) if numel > 0 else 0.0
            max_abs = float(abs_delta.max()) if numel > 0 else 0.0
            mean_val = float(delta.mean()) if numel > 0 else 0.0
            std_val = float(delta.std()) if numel > 0 else 0.0
            dead_fraction = float(np.mean(abs_delta < self._dead_eps)) if numel > 0 else 0.0
            incr_l2 = float(np.linalg.norm(incr_delta)) if numel > 0 else 0.0
            incr_max = float(abs_incr.max()) if numel > 0 else 0.0

            snapshot = LayerSnapshot(
                checkpoint_step=step_id,
                l2_norm=l2_norm,
                max_abs=max_abs,
                mean=mean_val,
                std=std_val,
                dead_fraction=dead_fraction,
                incremental_l2=incr_l2,
                incremental_max_abs=incr_max,
            )
            current_snapshots[name] = snapshot
            self._layer_series[name].snapshots.append(snapshot)

            # Update incremental L2 history and detect divergence.
            self._incr_l2_history[name].append(incr_l2)
            history = self._incr_l2_history[name]

            if len(history) >= 3:
                # Only check the new value against prior history (exclude it from the set).
                zscore = modified_zscore(incr_l2, history[:-1])
                if zscore > self._anomaly_threshold:
                    new_alerts.append(
                        DivergenceAlert(
                            layer_name=name,
                            first_drift_step=step_id,
                            first_drift_incr_l2=incr_l2,
                            modified_zscore=zscore,
                            l2_trend=list(history),
                        )
                    )

            self._prev_delta[name] = delta

        # Store snapshot.
        self._snapshots.append(_InMemorySnapshot(step_id=step_id, snapshots=current_snapshots))

        # Throttled logging.
        self._update_count += 1
        if self._logger is not None and self._log_every > 0 and self._update_count % self._log_every == 0:
            self._push_to_logger(step_id)

        return new_alerts

    def _push_to_logger(self, step_id: int) -> None:
        """Push per-layer drift metrics to the configured logger."""
        metrics: dict[str, float] = {}
        for name, history in self._incr_l2_history.items():
            if history:
                metrics[f"drift/incr_l2/{name}"] = history[-1]
        if not metrics:
            return

        logger = self._logger
        try:
            if hasattr(logger, "log"):
                # wandb, tensorboard, etc.
                logger.log(metrics, step=step_id)
            elif callable(logger):
                logger(metrics, step=step_id)
        except Exception:
            # Logging should never interrupt training.
            pass

    def checkpoints(self) -> list[CheckpointInfo]:
        """Return the recorded snapshots as ``CheckpointInfo`` objects for debug CLI use.

        These are purely in-memory; no files are read or written.
        """
        return [
            CheckpointInfo(
                path=Path(f"<memory:step={s.step_id}>"),
                step=s.step_id,
                label=f"step_{s.step_id}",
                mtime=float(s.step_id),
            )
            for s in self._snapshots
        ]

    def layer_series(self) -> dict[str, LayerSeries]:
        """Return the accumulated per-layer time series."""
        return self._layer_series

    def new_divergences(self) -> list[DivergenceAlert]:
        """Return all divergence alerts recorded so far.

        Uses the same logic as ``update()``: each new value is scored against
        the prior history only (not including itself), so the first outlier
        in a sequence is the one that gets alerted.
        """
        alerts: list[DivergenceAlert] = []
        for name, history in self._incr_l2_history.items():
            if len(history) < 3:
                continue
            for i, value in enumerate(history):
                # Score against only prior history.
                prior = history[:i] if i > 0 else history
                if len(prior) < 2:
                    continue
                zscore = modified_zscore(value, prior)
                if zscore > self._anomaly_threshold:
                    alerts.append(
                        DivergenceAlert(
                            layer_name=name,
                            first_drift_step=self._snapshots[i].step_id if i < len(self._snapshots) else i,
                            first_drift_incr_l2=value,
                            modified_zscore=zscore,
                            l2_trend=list(history),
                        )
                    )
                    break  # Only the first outlier triggers
        return alerts
