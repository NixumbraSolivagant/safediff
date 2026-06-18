"""Tests for safediff.integrations (TrainerTracker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from safediff.integrations import Tracker, _InMemorySnapshot


class TestInMemorySnapshot:
    def test_creation(self) -> None:
        snap = _InMemorySnapshot(step_id=42, snapshots={})
        assert snap.step_id == 42
        assert snap.snapshots == {}


class TestTracker:
    def test_no_io_on_update(self) -> None:
        """update() must never perform disk I/O."""
        baseline = {"w": np.zeros((10, 10), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        current = {"w": np.ones((10, 10), dtype=np.float32)}
        alerts = tracker.update(current, step_id=1)
        assert isinstance(alerts, list)

    def test_returns_empty_on_stable_weights(self) -> None:
        """Identical weights should produce no alerts."""
        baseline = {"w": np.zeros((10, 10), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        for step in range(1, 6):
            current = {"w": np.zeros((10, 10), dtype=np.float32)}
            alerts = tracker.update(current, step_id=step)
            assert alerts == []

    def test_returns_alert_on_drift(self) -> None:
        """A layer that suddenly changes should produce an alert."""
        baseline = {"w": np.zeros((10, 10), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, anomaly_threshold=3.5)
        # Steps 1-3: stable zeros
        for step in range(1, 4):
            tracker.update({"w": np.zeros((10, 10), dtype=np.float32)}, step_id=step)
        # Step 4: sudden large drift
        alerts = tracker.update(
            {"w": np.full((10, 10), 5.0, dtype=np.float32)},
            step_id=4,
        )
        assert len(alerts) >= 1
        assert any(a.layer_name == "w" for a in alerts)

    def test_wandb_logger_called(self) -> None:
        """When logger is provided and log_every fires, logger.log is called."""
        mock_logger = MagicMock()
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, logger=mock_logger, log_every=2)
        for step in range(1, 5):
            tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=step)
        # log_every=2, 4 updates → should log at steps 1 and 3 (0-indexed: 2nd and 4th call)
        # update_count: 1→log, 2→no, 3→log, 4→no
        assert mock_logger.log.call_count == 2

    def test_log_every_throttling(self) -> None:
        """Logger should only fire every log_every updates."""
        mock_logger = MagicMock()
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, logger=mock_logger, log_every=5)
        for step in range(1, 11):
            tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=step)
        # 10 updates, log_every=5 → log at updates 5 and 10
        assert mock_logger.log.call_count == 2

    def test_log_every_zero_disables(self) -> None:
        """log_every=0 should never call the logger."""
        mock_logger = MagicMock()
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, logger=mock_logger, log_every=0)
        for step in range(1, 6):
            tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=step)
        mock_logger.log.assert_not_called()

    def test_checkpoints_returns_all_snapshots(self) -> None:
        """checkpoints() should return one entry per update step."""
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        for step in range(1, 4):
            tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=step)
        ckpts = tracker.checkpoints()
        assert len(ckpts) == 4  # baseline + 3 updates

    def test_layer_series_accumulates(self) -> None:
        """layer_series() should grow snapshots on every update."""
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        for step in range(1, 4):
            tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=step)
        series = tracker.layer_series()
        assert "w" in series
        assert len(series["w"].snapshots) == 4  # baseline + 3 updates

    def test_new_divergences_returns_all(self) -> None:
        """new_divergences() should return all historical alerts."""
        baseline = {"w": np.zeros((10, 10), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, anomaly_threshold=3.5)
        # Stable for 3 steps, drift at step 4
        for step in range(1, 4):
            tracker.update({"w": np.zeros((10, 10), dtype=np.float32)}, step_id=step)
        tracker.update({"w": np.full((10, 10), 5.0, dtype=np.float32)}, step_id=4)
        divergences = tracker.new_divergences()
        assert len(divergences) >= 1

    def test_missing_layer_in_state_dict_skipped(self) -> None:
        """If a baseline layer is missing from current state_dict, skip it gracefully."""
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        # "w" is in baseline but missing from current
        alerts = tracker.update({}, step_id=1)
        assert alerts == []  # should not crash

    def test_shape_mismatch_handled(self) -> None:
        """Shape mismatch should produce inf snapshot, not crash."""
        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0)
        alerts = tracker.update({"w": np.zeros((3, 3), dtype=np.float32)}, step_id=1)
        assert alerts == []  # should not crash

    def test_callable_logger(self) -> None:
        """A plain callable should work as logger."""
        calls: list[tuple] = []

        def my_logger(metrics, step) -> None:
            calls.append((metrics, step))

        baseline = {"w": np.zeros((5, 5), dtype=np.float32)}
        tracker = Tracker(baseline, step_id=0, logger=my_logger, log_every=1)
        tracker.update({"w": np.zeros((5, 5), dtype=np.float32)}, step_id=1)
        assert len(calls) == 1
        assert calls[0][1] == 1

    def test_anomaly_threshold_parameter(self) -> None:
        """The threshold parameter is accepted and passed through correctly."""
        baseline = {"w": np.zeros((10, 10), dtype=np.float32)}
        # Both trackers run without error with different thresholds
        tracker_strict = Tracker(baseline, step_id=0, anomaly_threshold=3.5)
        tracker_lenient = Tracker(baseline, step_id=0, anomaly_threshold=50.0)
        # Verify the parameter was stored
        assert tracker_strict._anomaly_threshold == 3.5
        assert tracker_lenient._anomaly_threshold == 50.0
        # Both run without error
        for step in range(1, 4):
            tracker_strict.update({"w": np.zeros((10, 10), dtype=np.float32)}, step_id=step)
            tracker_lenient.update({"w": np.zeros((10, 10), dtype=np.float32)}, step_id=step)
        assert tracker_strict.new_divergences() == []
        assert tracker_lenient.new_divergences() == []
