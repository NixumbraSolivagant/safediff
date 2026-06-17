"""Tests for safediff.visualizer."""

from __future__ import annotations

import io
import json

import numpy as np
from rich.console import Console

from safediff.analyzer import analyze
from safediff.visualizer import render_json, render_report, sparkline


def test_sparkline_uses_block_glyphs() -> None:
    arr = np.random.default_rng(0).standard_normal(200)
    out = sparkline(arr, width=20)
    assert len(out) == 20
    assert all(c in " ▂▃▄▅▆▇█" for c in out)


def test_sparkline_empty_input() -> None:
    assert sparkline(np.array([]), width=10) == ""


def test_sparkline_constant_input_is_valid_glyphs() -> None:
    arr = np.zeros(100, dtype=np.float32)
    out = sparkline(arr, width=10)
    # All values equal — every character is either a block (peak bin) or blank
    # (other bins). No garbage should leak in.
    assert len(out) == 10
    assert all(c in " ▂▃▄▅▆▇█" for c in out)
    assert "█" in out  # at least one bin is at peak


def test_render_report_smoke() -> None:
    a = {"w1": np.zeros((2, 2), dtype=np.float32), "w2": np.zeros((2,), dtype=np.float32)}
    b = {
        "w1": np.full((2, 2), 0.1, dtype=np.float32),
        "w2": np.full((2,), 0.5, dtype=np.float32),
    }
    report = analyze(a, b)
    deltas = {k: b[k] - a[k] for k in set(a) & set(b)}

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)
    render_report(report, console=console, sparkline_width=20, deltas=deltas)
    text = buf.getvalue()
    assert "safediff" in text
    assert "w1" in text
    assert "w2" in text


def test_render_report_no_common_layers() -> None:
    a = {"x": np.zeros((2,))}
    b = {"y": np.zeros((2,))}
    report = analyze(a, b)
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)
    render_report(report, console=console)
    assert "No common layers" in buf.getvalue()


def test_render_json_is_valid() -> None:
    a = {"w": np.zeros((2, 2), dtype=np.float32)}
    b = {"w": np.full((2, 2), 0.5, dtype=np.float32)}
    report = analyze(a, b)
    payload = json.loads(render_json(report))
    assert "layers" in payload
    assert payload["layers"][0]["max_abs"] == 0.5
