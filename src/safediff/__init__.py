"""safediff — Checkpoint audit and learning dynamics tracking for PyTorch models.

The package exposes two entry points:

1. **Python SDK** (recommended)::

       from safediff import Tracker, quant_analyze

       # In training loop
       tracker = Tracker(baseline_state_dict, step_id=0, logger=wandb.run)
       alert = tracker.update(model_weights, step_id=step)

       # Quant pre-flight
       report = quant_analyze("model.safetensors")
       print(report.summary_line())

2. **CLI**::

       safediff quant model.safetensors
       safediff track ./checkpoints
       safediff audit model.safetensors
       safediff compare A.safetensors B.safetensors
"""

from __future__ import annotations

from safediff.analyzer import DiffReport, LayerStat, analyze
from safediff.audit import AuditReport, OutlierReport, audit
from safediff.integrations import Tracker
from safediff.loader import load_tensors
from safediff.quant import (
    QuantLayerStat,
    QuantReport,
    QuantScheme,
    analyze as quant_analyze,
    suggest_scheme,
)
from safediff.track import (
    CheckpointInfo,
    DivergenceAlert,
    LayerSeries,
    LayerSnapshot,
    anomaly_score,
    compute_delta,
    discover_checkpoints,
    modified_zscore,
    top_layers_by_drift,
    track,
)

__version__ = "0.3.0"
__all__ = [
    "__version__",
    # loader
    "load_tensors",
    # analyzer (compare command)
    "analyze",
    "DiffReport",
    "LayerStat",
    # audit (audit command)
    "audit",
    "AuditReport",
    "OutlierReport",
    # quant (quant command / SDK)
    "quant_analyze",
    "suggest_scheme",
    "QuantReport",
    "QuantScheme",
    "QuantLayerStat",
    # track (track command)
    "track",
    "discover_checkpoints",
    "CheckpointInfo",
    "LayerSeries",
    "LayerSnapshot",
    "DivergenceAlert",
    "top_layers_by_drift",
    "modified_zscore",
    # training-loop integration
    "Tracker",
    "compute_delta",
    "anomaly_score",
]
