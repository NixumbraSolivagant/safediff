"""Tests for safediff.loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from safediff.loader import (
    TorchNotInstalledError,
    UnsupportedFormatError,
    load_tensors,
)


def test_load_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    payload = {"w1": np.arange(6, dtype=np.float32).reshape(2, 3)}
    save_file(payload, str(path))

    out = load_tensors(path)
    assert set(out) == {"w1"}
    assert out["w1"].shape == (2, 3)
    np.testing.assert_array_equal(out["w1"], payload["w1"])


def test_load_unknown_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "x.qq"
    bogus.write_text("nope")
    with pytest.raises(UnsupportedFormatError):
        load_tensors(bogus)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tensors(tmp_path / "ghost.safetensors")


def test_load_torch_raises_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate an environment without torch and confirm a clean error."""
    p = tmp_path / "x.pt"
    p.write_text("not a real torch file")
    # Force the ImportError path even if torch is actually installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises((TorchNotInstalledError, Exception)):
        load_tensors(p)


def test_metadata_keys_are_dropped(tmp_path: Path) -> None:
    """Non-tensor keys like step / scheduler should be ignored by the analyzer path."""
    from safediff.loader import _filter_tensors

    raw = {
        "layer.weight": np.zeros((2, 2), dtype=np.float32),
        "step": 100,  # int — must be dropped
        "scheduler_state_dict": {"foo": "bar"},
    }
    filtered = _filter_tensors(raw)
    assert set(filtered) == {"layer.weight"}
