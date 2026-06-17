"""Weight file loaders.

Supports:
* ``.safetensors`` (always available, the recommended path)
* ``.pt`` / ``.pth`` / ``.bin`` (requires optional ``torch`` extra)

All loaders return a flat ``dict[str, np.ndarray]`` of tensor name -> array.
Non-tensor metadata keys (optimizer state, training step, etc.) are filtered
out so the analyzer only ever sees weight matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

# Keys that frequently appear in PyTorch checkpoints but are NOT weight tensors.
_NON_TENSOR_KEYS = {
    "step",
    "global_step",
    "epoch",
    "num_training_steps",
    "scheduler_state_dict",
    "optimizer_state_dict",
    "args",
    "config",
    "model_config",
    "non_persistent_buffers_set",
    "state_dict",
    "pytorch-lightning_version",
    "callbacks",
    "lr_schedulers",
    "amp_scaler",
    "loops",
}


class UnsupportedFormatError(ValueError):
    """Raised when the file extension is not recognized by any loader."""


class TorchNotInstalledError(ImportError):
    """Raised when a torch checkpoint is opened without the torch extra."""


def _coerce_to_numpy(value: Any) -> np.ndarray | None:
    """Convert a tensor-like object to ``np.ndarray``; return ``None`` for non-tensors."""
    if isinstance(value, np.ndarray):
        return value
    # Lazy torch import — only happens if a torch checkpoint is opened.
    try:
        import torch  # type: ignore
    except ImportError:
        # If torch is not installed, the caller will not be in a torch path,
        # so silently skip the conversion.
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return None


def _filter_tensors(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Drop metadata keys and convert the rest to numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key in _NON_TENSOR_KEYS:
            continue
        if key.startswith("_"):
            continue
        arr = _coerce_to_numpy(value)
        if arr is None:
            continue
        out[key] = arr
    return out


def _load_safetensors(path: str) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    tensors: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _load_torch(path: str) -> dict[str, np.ndarray]:
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise TorchNotInstalledError(
            f"Loading {Path(path).name} requires PyTorch. "
            f"Install it with: pip install 'safediff[torch]'"
        ) from exc
    import torch  # type: ignore

    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Allow nested state_dict layout: {"state_dict": {...}} or {"model": {...}}.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        raw = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        raw = obj["model"]
    elif isinstance(obj, dict):
        raw = obj
    else:
        raise UnsupportedFormatError(
            f"{path}: expected a state-dict-like dict, got {type(obj).__name__}"
        )
    return _filter_tensors(raw)


_LOADERS: dict[str, Callable[[str], dict[str, np.ndarray]]] = {
    ".safetensors": _load_safetensors,
    ".pt": _load_torch,
    ".pth": _load_torch,
    ".bin": _load_torch,
}


def load_tensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load every tensor from ``path`` into a ``dict[str, np.ndarray]``.

    The choice of loader is dispatched on file extension. Metadata keys
    (optimizer state, training step, ...) are stripped automatically.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    suffix = p.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise UnsupportedFormatError(
            f"Unsupported checkpoint format: {suffix!r}. "
            f"Supported: {sorted(_LOADERS)}"
        )
    return loader(str(p))
