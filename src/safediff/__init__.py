"""safediff — Checkpoint static audit and learning dynamics tracking for PyTorch models."""

from __future__ import annotations

from safediff.analyzer import DiffReport, LayerStat, analyze
from safediff.audit import AuditReport, OutlierReport, audit
from safediff.loader import load_tensors
from safediff.track import (
    CheckpointInfo,
    DivergenceAlert,
    LayerSeries,
    LayerSnapshot,
    discover_checkpoints,
    modified_zscore,
    track,
    top_layers_by_drift,
)

__version__ = "0.2.0"
__all__ = [
    "__version__",
    # loader
    "load_tensors",
    # analyzer
    "analyze",
    "DiffReport",
    "LayerStat",
    # audit
    "audit",
    "AuditReport",
    "OutlierReport",
    # track
    "track",
    "discover_checkpoints",
    "CheckpointInfo",
    "LayerSeries",
    "LayerSnapshot",
    "DivergenceAlert",
    "top_layers_by_drift",
    "modified_zscore",
]
