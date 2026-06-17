"""Built-in demo: fabricate two checkpoints from a tiny GPT-2.

The demo path is intentionally self-contained:
1. Try to download ``sshleifer/tiny-gpt2`` (~5MB) — this is the realistic path.
2. Fall back to fabricating a tiny random transformer if huggingface_hub is
   unavailable or the network is offline. The fabricated data still exercises
   every analyzer code path and produces a visually rich output.

The fabricated path also lets the test suite run without network access.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from safediff.analyzer import DiffReport, analyze
from safediff.loader import _filter_tensors

DEMO_REPO = "sshleifer/tiny-gpt2"


def _fabricate_state_dicts(rng: np.random.Generator) -> tuple[dict, dict]:
    """Synthesize two tiny ``GPT-2``-shaped state-dicts for offline demos."""
    shapes = {
        "transformer.wte.weight": (50, 32),
        "transformer.wpe.weight": (32, 32),
        "transformer.h.0.ln_1.weight": (32,),
        "transformer.h.0.ln_1.bias": (32,),
        "transformer.h.0.attn.c_attn.weight": (32, 96),
        "transformer.h.0.attn.c_attn.bias": (96,),
        "transformer.h.0.attn.c_proj.weight": (32, 32),
        "transformer.h.0.attn.c_proj.bias": (32,),
        "transformer.h.0.ln_2.weight": (32,),
        "transformer.h.0.ln_2.bias": (32,),
        "transformer.h.0.mlp.c_fc.weight": (32, 128),
        "transformer.h.0.mlp.c_fc.bias": (128,),
        "transformer.h.0.mlp.c_proj.weight": (128, 32),
        "transformer.h.0.mlp.c_proj.bias": (32,),
        "transformer.ln_f.weight": (32,),
        "transformer.ln_f.bias": (32,),
        "lm_head.weight": (50, 32),
    }
    # Build a "normal" checkpoint: small Gaussian weights, layer-norm ones set to 1.
    a = {}
    for name, shape in shapes.items():
        if name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name == "transformer.ln_f.weight":
            a[name] = np.ones(shape, dtype=np.float32)
        elif name.endswith(".bias") and "ln_" in name:
            a[name] = np.zeros(shape, dtype=np.float32)
        else:
            a[name] = np.asarray(rng.standard_normal(shape) * 0.02, dtype=np.float32)

    # Build checkpoint B: copy A, then inject a simulated anomaly (one layer
    # has its weights blown up 1000x) and zero out 30% of one layer's parameters
    # to make them "dead".
    b = {k: v.copy() for k, v in a.items()}
    rng.shuffle(list(b.keys()))
    explode_target = list(b.keys())[0]
    b[explode_target] = b[explode_target] * 1000.0

    # Pick another layer and zero out part of it to demonstrate dead neurons.
    dead_target = "transformer.h.0.mlp.c_fc.weight"
    if dead_target in b:
        flat = b[dead_target].ravel().copy()
        n_dead = int(flat.size * 0.30)
        idx = rng.choice(flat.size, size=n_dead, replace=False)
        flat[idx] = a[dead_target].ravel()[idx]  # keep delta == 0 for those
        b[dead_target] = flat.reshape(b[dead_target].shape)
    return a, b


def _try_download_demo() -> tuple[dict, dict] | None:
    """Best-effort fetch of ``sshleifer/tiny-gpt2`` from the HF Hub.

    Returns ``None`` if huggingface_hub is missing or the network is down.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None

    try:
        from safetensors.torch import load_file  # type: ignore
    except ImportError:
        load_file = None  # type: ignore

    try:
        path = hf_hub_download(
            repo_id=DEMO_REPO,
            filename="model.safetensors",
            cache_dir=os.environ.get("HF_HOME"),
        )
    except Exception:
        return None

    if load_file is None:
        # Fall back to a numpy-only loader: safetensors.safe_open works without torch.
        from safetensors import safe_open

        tensors_a: dict = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                tensors_a[k] = f.get_tensor(k)
    else:
        tensors_a = load_file(path)
    tensors_a = _filter_tensors(tensors_a)

    # Simulate "epoch 11" by adding small Gaussian noise to A and inflating one
    # tensor to make the diff visually interesting.
    rng = np.random.default_rng(42)
    tensors_b = {k: v.copy() for k, v in tensors_a.items()}
    if tensors_b:
        target = sorted(tensors_b)[0]
        tensors_b[target] = tensors_b[target] * 100.0
    for k in tensors_b:
        tensors_b[k] = tensors_b[k] + rng.standard_normal(tensors_b[k].shape).astype(
            tensors_b[k].dtype
        ) * (np.abs(tensors_a[k]).mean() + 1e-6) * 0.05
    return tensors_a, tensors_b


def run_demo() -> tuple[DiffReport, dict[str, np.ndarray]]:
    """Return a ``DiffReport`` and the underlying deltas for the demo."""
    fetched = _try_download_demo()
    if fetched is not None:
        a, b = fetched
    else:
        rng = np.random.default_rng(0)
        a, b = _fabricate_state_dicts(rng)

    report = analyze(a, b, dead_eps=1e-6, anomaly_threshold=8.0)
    deltas = {k: b[k] - a[k] for k in set(a) & set(b)}
    return report, deltas


def demo_to_disk(out_dir: str | Path) -> tuple[Path, Path]:
    """Materialize two ``.safetensors`` files for the demo, return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    a, b = _fabricate_state_dicts(rng)
    a_path = out / "demo_a.safetensors"
    b_path = out / "demo_b.safetensors"
    save_file(a, str(a_path))
    save_file(b, str(b_path))
    return a_path, b_path
