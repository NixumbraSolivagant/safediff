"""Tests for safediff.analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from safediff.analyzer import LayerStat, analyze


def test_basic_stats() -> None:
    a = {"w": np.zeros((4, 4), dtype=np.float32)}
    b = {"w": np.ones((4, 4), dtype=np.float32)}
    report = analyze(a, b, dead_eps=1e-6)
    assert len(report.common) == 1
    s = report.common[0]
    assert s.max_abs == pytest.approx(1.0)
    assert s.l2_norm == pytest.approx(4.0)  # 16 * 1^2
    assert s.mean == pytest.approx(1.0)
    assert s.is_dead_fraction == pytest.approx(0.0)


def test_zero_delta_is_fully_dead() -> None:
    a = {"w": np.full((3, 3), 0.5, dtype=np.float32)}
    b = {"w": a["w"].copy()}
    report = analyze(a, b, dead_eps=1e-6)
    assert report.common[0].is_dead_fraction == pytest.approx(1.0)
    assert report.common[0].l2_norm == pytest.approx(0.0)


def test_sorted_by_l2_descending() -> None:
    a = {f"w{i}": np.zeros((2, 2), dtype=np.float32) for i in range(3)}
    b = {
        "w0": np.full((2, 2), 1.0, dtype=np.float32),  # L2 = 2
        "w1": np.full((2, 2), 10.0, dtype=np.float32),  # L2 = 20
        "w2": np.full((2, 2), 3.0, dtype=np.float32),  # L2 = 6
    }
    report = analyze(a, b)
    names = [s.name for s in report.common]
    assert names == ["w1", "w2", "w0"]


def test_anomaly_detection_marks_extreme_layer() -> None:
    a = {f"w{i}": np.zeros((2, 2), dtype=np.float32) for i in range(5)}
    b = {k: v.copy() for k, v in a.items()}
    # All layers have a tiny change except one that's 1000x.
    for k in b:
        b[k] = b[k] + 0.001
    b["w2"] = b["w2"] + 5.0
    report = analyze(a, b, anomaly_threshold=10.0)
    flagged = {s.name for s in report.anomalies}
    assert "w2" in flagged


def test_max_abs_above_one_always_anomalous() -> None:
    a = {"w": np.zeros((2, 2), dtype=np.float32)}
    b = {"w": np.full((2, 2), 1.5, dtype=np.float32)}
    report = analyze(a, b, anomaly_threshold=10_000.0)
    assert any(s.is_anomaly for s in report.common)


def test_only_in_a_and_b() -> None:
    a = {"keep": np.zeros((2,)), "gone": np.zeros((2,))}
    b = {"keep": np.zeros((2,)), "added": np.zeros((2,))}
    report = analyze(a, b)
    assert report.only_in_a == ["gone"]
    assert report.only_in_b == ["added"]


def test_shape_mismatch_surfaces_as_anomaly() -> None:
    a = {"w": np.zeros((2, 2), dtype=np.float32)}
    b = {"w": np.zeros((3, 3), dtype=np.float32)}
    report = analyze(a, b)
    s = report.common[0]
    assert s.is_anomaly
    assert s.l2_norm == float("inf")


def test_layerstat_to_dict_round_trip() -> None:
    s = LayerStat(
        name="x", shape=(1,), numel=1, max_abs=0.1, l2_norm=0.1, mean=0.0, std=0.0
    )
    d = s.to_dict()
    assert d["name"] == "x"
    assert d["shape"] == [1]


def test_report_filter_glob() -> None:
    a = {"attn.w": np.zeros((2,)), "mlp.w": np.zeros((2,))}
    b = {"attn.w": np.zeros((2,)), "mlp.w": np.zeros((2,))}
    report = analyze(a, b)
    filtered = analyze(a, b).filter("attn.*")
    # Original report unchanged; filtered only keeps attn.
    assert len(report.common) == 2  # we created a new report, length 2
    assert len(filtered.common) == 1
    assert filtered.common[0].name == "attn.w"


def test_total_params_counted() -> None:
    a = {"w1": np.zeros((3, 3)), "w2": np.zeros((4,))}
    b = {"w1": np.zeros((3, 3)), "w2": np.zeros((4,))}
    report = analyze(a, b)
    assert report.total_params_a == 13
    assert report.total_params_b == 13
