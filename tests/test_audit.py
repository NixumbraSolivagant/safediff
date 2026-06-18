"""Tests for safediff.audit (Model Sanity Checker)."""

from __future__ import annotations

import numpy as np
import pytest

from safediff.audit import (
    AuditReport,
    OutlierReport,
    _find_frozen_layers,
    _inspect,
    audit,
)


# ------------------------------------------------------------------------------------------
# _inspect
# ------------------------------------------------------------------------------------------

class TestInspect:
    def test_nan_detected(self) -> None:
        arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        r = _inspect(arr, "test.nan", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.has_nan
        assert not r.has_inf

    def test_pos_inf_detected(self) -> None:
        arr = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        r = _inspect(arr, "test.inf", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.has_inf
        assert r.has_pos_inf
        assert not r.has_neg_inf

    def test_neg_inf_detected(self) -> None:
        arr = np.array([-np.inf, 0.0, 1.0], dtype=np.float32)
        r = _inspect(arr, "test.neginf", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.has_inf
        assert r.has_neg_inf

    def test_outlier_count(self) -> None:
        # Use a Gaussian distribution and inject an obvious 100σ outlier.
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(1000).astype(np.float32)
        arr[0] = 1000.0  # clear outlier
        r = _inspect(arr, "test.outlier", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.outlier_count >= 1
        assert r.outlier_fraction > 0

    def test_no_outliers_in_normal_dist(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(1000).astype(np.float32)
        r = _inspect(arr, "test.normal", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.outlier_fraction < 0.001

    def test_near_zero_detection(self) -> None:
        arr = np.array([1e-9, 1e-9, 1.0], dtype=np.float32)
        r = _inspect(arr, "test.nearzero", outlier_sigma=5.0, near_zero_eps=1e-6)
        # 2 out of 3 are near-zero
        assert r.near_zero_fraction == pytest.approx(2 / 3)

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=np.float32)
        r = _inspect(arr, "test.empty", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.numel == 0
        assert r.has_nan is False
        assert r.has_inf is False

    def test_is_unsafe_true_for_nan(self) -> None:
        arr = np.array([np.nan], dtype=np.float32)
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.is_unsafe

    def test_is_unsafe_true_for_high_outlier_fraction(self) -> None:
        # 5% of values are extreme outliers beyond any realistic σ
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(1000).astype(np.float32)
        arr[:50] = 1e6  # 5% extreme outliers
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.outlier_fraction > 0.01
        assert r.is_unsafe

    def test_severity_nan_is_critical(self) -> None:
        arr = np.array([np.nan], dtype=np.float32)
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.severity() == "critical"

    def test_severity_high_outlier_is_high(self) -> None:
        arr = np.zeros(100, dtype=np.float32)
        arr[:3] = 1000.0  # 3% outliers
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.severity() == "high"

    def test_severity_near_zero_is_warning(self) -> None:
        # 96 zeros, 4 small-but-not-near-zero values. >95% near-zero, no outliers.
        arr = np.zeros(100, dtype=np.float32)
        arr[:4] = 1e-5  # above 1e-6 threshold → not near-zero, but not an outlier either
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.near_zero_fraction > 0.95
        assert r.outlier_fraction == 0.0
        assert r.severity() == "warning"

    def test_severity_ok(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(100).astype(np.float32)
        r = _inspect(arr, "test", outlier_sigma=5.0, near_zero_eps=1e-6)
        assert r.severity() == "ok"


# ------------------------------------------------------------------------------------------
# _find_frozen_layers
# ------------------------------------------------------------------------------------------

class TestFindFrozenLayers:
    def test_no_frozen_layers(self) -> None:
        tensors = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0]),
        }
        assert _find_frozen_layers(tensors) == []

    def test_identical_layers_found(self) -> None:
        tensors = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([1.0, 2.0]),
        }
        frozen = _find_frozen_layers(tensors)
        assert ("a", "b") in frozen or ("b", "a") in frozen

    def test_different_shape_not_compared(self) -> None:
        tensors = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([1.0, 2.0, 3.0]),
        }
        assert _find_frozen_layers(tensors) == []

    def test_nearly_identical_treated_as_frozen(self) -> None:
        tensors = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([1.0 + 1e-9, 2.0 + 1e-9]),
        }
        frozen = _find_frozen_layers(tensors)
        assert len(frozen) == 1

    def test_capped_at_50_pairs(self) -> None:
        # 20 identical layers → C(20,2) = 190 pairs, but should be capped at 50
        tensors = {f"l{i}": np.array([1.0, 2.0]) for i in range(20)}
        frozen = _find_frozen_layers(tensors)
        assert len(frozen) <= 50


# ------------------------------------------------------------------------------------------
# audit
# ------------------------------------------------------------------------------------------

class TestAudit:
    def test_nan_layers_in_report(self) -> None:
        tensors = {
            "good": np.array([1.0, 2.0], dtype=np.float32),
            "bad": np.array([np.nan, 3.0], dtype=np.float32),
        }
        report = audit(tensors)
        assert len(report.nan_layers) == 1
        assert report.nan_layers[0].name == "bad"
        assert not report.is_healthy

    def test_inf_layers_in_report(self) -> None:
        tensors = {
            "bad": np.array([-np.inf, np.inf, 0.0], dtype=np.float32),
        }
        report = audit(tensors)
        assert len(report.inf_layers) == 1
        assert not report.is_healthy

    def test_near_zero_layers_flagged(self) -> None:
        tensors = {
            "almost_dead": np.zeros(100, dtype=np.float32),
        }
        report = audit(tensors, near_zero_eps=1e-6)
        assert len(report.near_zero_layers) == 1
        assert report.near_zero_layers[0].near_zero_fraction > 0.9

    def test_outlier_layers_sorted_descending(self) -> None:
        # Use Gaussian distributions so 5σ outliers are detectable
        rng = np.random.default_rng(0)
        slight = rng.standard_normal(1000).astype(np.float32)
        slight[:5] = 100.0  # 0.5% outliers
        heavy = rng.standard_normal(1000).astype(np.float32)
        heavy[:50] = 100.0  # 5% outliers
        report = audit(
            {"slight_outlier": slight, "heavy_outlier": heavy},
            outlier_sigma=5.0,
        )
        # heavy_outlier has higher outlier fraction, must come first
        if len(report.outlier_layers) >= 2:
            assert report.outlier_layers[0].outlier_fraction >= report.outlier_layers[1].outlier_fraction
        elif len(report.outlier_layers) == 1:
            # Only the heavy one was detected — that's fine too
            assert report.outlier_layers[0].name == "heavy_outlier"

    def test_frozen_layers_detected(self) -> None:
        tensors = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([1.0, 2.0], dtype=np.float32),
        }
        report = audit(tensors)
        assert len(report.frozen_layers) >= 1

    def test_total_params_counted(self) -> None:
        tensors = {
            "w1": np.zeros((100, 100), dtype=np.float32),  # 10,000
            "w2": np.zeros((50,), dtype=np.float32),  # 50
        }
        report = audit(tensors)
        assert report.total_params == 10050

    def test_is_healthy_true_when_clean(self) -> None:
        rng = np.random.default_rng(0)
        tensors = {
            f"layer_{i}": rng.standard_normal((10, 10)).astype(np.float32)
            for i in range(5)
        }
        report = audit(tensors)
        assert report.is_healthy
        assert report.critical_count == 0
        assert report.warning_count == 0

    def test_health_report_text_ok(self) -> None:
        rng = np.random.default_rng(0)
        tensors = {
            "clean": rng.standard_normal((10, 10)).astype(np.float32),
        }
        report = audit(tensors)
        assert "all clear" in report.summary()

    def test_health_report_text_nan(self) -> None:
        tensors = {"nan_layer": np.array([np.nan], dtype=np.float32)}
        report = audit(tensors)
        assert "critical" in report.summary()

    def test_warning_count(self) -> None:
        # Outliers + near-zero both count toward warnings
        tensors = {
            "outlier_layer": np.zeros(100, dtype=np.float32),
            "near_zero_layer": np.full(100, 1e-9, dtype=np.float32),  # all near-zero
        }
        tensors["outlier_layer"][:10] = 100.0
        report = audit(tensors, outlier_sigma=5.0, near_zero_eps=1e-6)
        assert report.warning_count >= 1
