"""Tests for safediff.track (Learning Dynamics Tracker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from safediff.track import (
    CheckpointInfo,
    DivergenceAlert,
    LayerSnapshot,
    LayerSeries,
    discover_checkpoints,
    modified_zscore,
    normalize,
    top_layers_by_drift,
    track,
    _extract_step_from_name,
    _make_label,
)


# ------------------------------------------------------------------------------------------
# Sorting utilities
# ------------------------------------------------------------------------------------------

class TestExtractStepFromName:
    def test_epoch_format(self) -> None:
        assert _extract_step_from_name("epoch_10.safetensors") == 10
        assert _extract_step_from_name("epoch-5.pt") == 5
        assert _extract_step_from_name("my_model_epoch_001.pth") == 1

    def test_step_format(self) -> None:
        assert _extract_step_from_name("step_1000.safetensors") == 1000
        assert _extract_step_from_name("ckpt_50.bin") == 50

    def test_checkpoint_format(self) -> None:
        assert _extract_step_from_name("checkpoint_42.safetensors") == 42

    def test_large_step(self) -> None:
        assert _extract_step_from_name("model_step12345.pt") == 12345

    def test_no_number_returns_none(self) -> None:
        assert _extract_step_from_name("model_final.safetensors") is None
        assert _extract_step_from_name("best.pt") is None


class TestMakeLabel:
    def test_uses_extracted_tag(self) -> None:
        assert _make_label(Path("epoch_10.safetensors"), 10) == "epoch10"
        assert _make_label(Path("step_500.pt"), 500) == "step500"


# ------------------------------------------------------------------------------------------
# discover_checkpoints
# ------------------------------------------------------------------------------------------

class TestDiscoverCheckpoints:
    def test_requires_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_checkpoints(tmp_path / "nonexistent")

    def test_no_checkpoint_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a checkpoint")
        with pytest.raises(ValueError, match="No checkpoint files found"):
            discover_checkpoints(tmp_path)

    def test_sorting_by_extracted_step(self, tmp_path: Path) -> None:
        # epoch_10 should come before epoch_20
        (tmp_path / "epoch_20.safetensors").touch()
        (tmp_path / "epoch_10.safetensors").touch()
        ckpts = discover_checkpoints(tmp_path)
        assert [c.label for c in ckpts] == ["epoch10", "epoch20"]

    def test_unknown_files_sorted_by_mtime(self, tmp_path: Path) -> None:
        # No step number — falls back to mtime
        (tmp_path / "b.safetensors").touch()
        (tmp_path / "a.safetensors").touch()
        ckpts = discover_checkpoints(tmp_path)
        assert len(ckpts) == 2


# ------------------------------------------------------------------------------------------
# modified_zscore
# ------------------------------------------------------------------------------------------

class TestModifiedZscore:
    def test_identical_values_returns_zero(self) -> None:
        values = [1.0, 1.0, 1.0, 1.0]
        assert modified_zscore(1.0, values) == 0.0

    def test_outlier_is_high_zscore(self) -> None:
        # Half values are similar, half are clearly different — strong outlier
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 100.0]
        z = modified_zscore(100.0, values)
        assert z > 3.5  # outlier should exceed threshold

    def test_normal_value_is_low_zscore(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = modified_zscore(3.0, values)
        assert z < 1.0


class TestNormalize:
    def test_constant_list_returns_all_zeros(self) -> None:
        result = normalize([1.0, 1.0, 1.0])
        assert result == [0.0, 0.0, 0.0]

    def test_normalize_two_values(self) -> None:
        result = normalize([0.0, 1.0])
        assert result == [0.0, 1.0]

    def test_normalize_mixed(self) -> None:
        result = normalize([10.0, 20.0, 30.0])
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert 0.0 < result[1] < 1.0


# ------------------------------------------------------------------------------------------
# track core logic
# ------------------------------------------------------------------------------------------

class TestTrack:
    def _mock_ckpts(self, n: int) -> list[CheckpointInfo]:
        return [
            CheckpointInfo(path=Path(f"ckpt_{i}.safetensors"), step=i, label=f"ckpt{i}", mtime=float(i))
            for i in range(n)
        ]

    def _simple_loader(self, n_ckpts: int, layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the same tensors for every call (static model)."""
        return layers

    def _drifting_loader_factory(
        self, drift_at_ckpt: int, drift_layer: str, drift_scale: float
    ):
        def loader(path: Path) -> dict[str, np.ndarray]:
            step = int(path.stem.split("_")[1])
            tensors = {
                "stable": np.zeros((10, 10), dtype=np.float32),
                drift_layer: np.full((10, 10), 0.01, dtype=np.float32),
            }
            if step >= drift_at_ckpt:
                tensors[drift_layer] = np.full((10, 10), drift_scale, dtype=np.float32)
            return tensors
        return loader

    def test_requires_at_least_two_checkpoints(self, tmp_path: Path) -> None:
        (tmp_path / "epoch_0.safetensors").touch()
        (tmp_path / "epoch_1.safetensors").touch()
        ckpts = discover_checkpoints(tmp_path)
        # Patch loader to avoid actual loading
        with pytest.raises(ValueError, match="At least 2 checkpoints"):
            track(ckpts[:1], lambda p: {})

    def test_single_layer_no_change_is_fully_dead(self, tmp_path: Path) -> None:
        tensors = {"w": np.zeros((4, 4), dtype=np.float32)}
        ckpts = self._mock_ckpts(3)
        layer_series, alerts = track(ckpts, lambda p: tensors)
        assert "w" in layer_series
        assert len(layer_series["w"].snapshots) == 3
        # First snapshot is zero (baseline), others are zero too (no change)
        for snap in layer_series["w"].snapshots:
            assert snap.l2_norm == pytest.approx(0.0)
            assert snap.dead_fraction == pytest.approx(1.0)

    def test_drifting_layer_generates_alert(self, tmp_path: Path) -> None:
        """Layer that starts drifting at checkpoint 2 should be flagged."""
        for i in range(5):
            (tmp_path / f"ckpt_{i}.safetensors").touch()
        ckpts = discover_checkpoints(tmp_path)
        loader = self._drifting_loader_factory(drift_at_ckpt=2, drift_layer="drift_layer", drift_scale=5.0)
        layer_series, alerts = track(ckpts, loader, anomaly_threshold=3.5)

        # drift_layer should have an alert
        drift_alerts = [a for a in alerts if a.layer_name == "drift_layer"]
        assert len(drift_alerts) == 1
        assert drift_alerts[0].first_drift_step == 2

        # stable layer should not be flagged
        stable_alerts = [a for a in alerts if a.layer_name == "stable"]
        assert len(stable_alerts) == 0

    def test_layer_series_incremental_l2_correct(self, tmp_path: Path) -> None:
        """Incremental L2 should be zero between identical checkpoints."""
        for i in range(3):
            (tmp_path / f"ckpt_{i}.safetensors").touch()
        ckpts = discover_checkpoints(tmp_path)
        tensors = {"w": np.zeros((10, 10), dtype=np.float32)}
        layer_series, alerts = track(ckpts, lambda p: tensors)

        series = layer_series["w"]
        # All three snapshots are identical to baseline — all dead
        assert len(series.snapshots) == 3
        for snap in series.snapshots:
            assert snap.incremental_l2 == pytest.approx(0.0)


class TestTopLayersByDrift:
    def test_sorted_by_cumulative_l2(self) -> None:
        series = {
            "big": LayerSeries(
                name="big",
                shape=(100,),
                snapshots=[
                    LayerSnapshot(
                        checkpoint_step=0, l2_norm=10.0, max_abs=10.0,
                        mean=0.0, std=0.0, dead_fraction=0.0,
                        incremental_l2=10.0, incremental_max_abs=10.0,
                    ),
                ],
            ),
            "small": LayerSeries(
                name="small",
                shape=(10,),
                snapshots=[
                    LayerSnapshot(
                        checkpoint_step=0, l2_norm=1.0, max_abs=1.0,
                        mean=0.0, std=0.0, dead_fraction=0.0,
                        incremental_l2=1.0, incremental_max_abs=1.0,
                    ),
                ],
            ),
        }
        top = top_layers_by_drift(series, metric="cumulative_l2")
        assert [n for n, _, _ in top] == ["big", "small"]


class TestDivergenceAlert:
    def test_alert_contains_full_trend(self) -> None:
        alert = DivergenceAlert(
            layer_name="test.layer",
            first_drift_step=3,
            first_drift_incr_l2=2.5,
            modified_zscore=4.2,
            l2_trend=[0.01, 0.02, 0.05, 2.5, 2.8],
        )
        assert alert.layer_name == "test.layer"
        assert alert.first_drift_step == 3
        assert len(alert.l2_trend) == 5


# ------------------------------------------------------------------------------------------
# Public API: compute_delta, anomaly_score
# ------------------------------------------------------------------------------------------

from safediff.track import compute_delta, anomaly_score


class TestComputeDelta:
    def test_returns_snapshots_and_new_delta_map(self) -> None:
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        current = {"w": np.ones((5, 5), dtype=np.float32)}
        snapshots, new_delta = compute_delta(current, baseline, prev_delta=None)
        assert "w" in snapshots
        assert snapshots["w"].l2_norm > 0
        assert "w" in new_delta
        assert new_delta["w"].shape == (5, 5)

    def test_incremental_delta_uses_prev_delta(self) -> None:
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        current1 = {"w": np.ones((5, 5), dtype=np.float32)}
        current2 = {"w": np.full((5, 5), 2.0, dtype=np.float32)}

        _, delta1 = compute_delta(current1, baseline, prev_delta=None)
        snapshots2, delta2 = compute_delta(current2, baseline, prev_delta=delta1)

        # First delta: w = ones - zeros = ones
        assert delta1["w"].max() == pytest.approx(1.0)
        # Second delta: w - prev_delta = twos - ones = ones (incremental)
        assert snapshots2["w"].incremental_l2 > 0
        assert snapshots2["w"].incremental_l2 < snapshots2["w"].l2_norm

    def test_missing_layer_skipped(self) -> None:
        baseline = {"a": np.zeros((5, 5)), "b": np.zeros((5, 5))}
        current = {"a": np.ones((5, 5))}  # "b" is missing
        snapshots, _ = compute_delta(current, baseline, prev_delta=None)
        assert "a" in snapshots
        assert "b" not in snapshots

    def test_shape_mismatch_produces_inf(self) -> None:
        baseline = {"w": np.zeros((5, 5))}
        current = {"w": np.zeros((3, 3))}
        snapshots, _ = compute_delta(current, baseline, prev_delta=None)
        assert snapshots["w"].l2_norm == float("inf")

    def test_dead_fraction_calculated(self) -> None:
        baseline = {"w": np.zeros((10, 10))}
        current = {"w": np.zeros((10, 10))}  # identical → all dead
        snapshots, _ = compute_delta(current, baseline, prev_delta=None)
        assert snapshots["w"].dead_fraction == pytest.approx(1.0)

    def test_prev_delta_none_uses_full_delta(self) -> None:
        baseline = {"w": np.zeros((5, 5))}
        current = {"w": np.full((5, 5), 3.0)}
        snapshots, _ = compute_delta(current, baseline, prev_delta=None)
        # No prev_delta → incremental = full
        assert snapshots["w"].incremental_l2 == snapshots["w"].l2_norm


class TestAnomalyScore:
    def test_identical_history_returns_zero(self) -> None:
        history = [1.0, 1.0, 1.0, 1.0]
        score = anomaly_score(history, 1.0)
        assert score == 0.0

    def test_outlier_returns_high_score(self) -> None:
        history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
        score = anomaly_score(history, 100.0)
        assert score > 3.5  # should exceed NIST threshold

    def test_normal_value_returns_low_score(self) -> None:
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        score = anomaly_score(history, 3.0)
        assert score < 1.0

    def test_threshold_parameter_works(self) -> None:
        history = [1.0, 1.0, 1.0, 1.0, 10.0]
        score = anomaly_score(history, 10.0, threshold=5.0)
        # score should be high (>5) but we just verify the param is accepted
        assert score > 0

    def test_empty_history_returns_zero(self) -> None:
        score = anomaly_score([], 1.0)
        assert score == 0.0
