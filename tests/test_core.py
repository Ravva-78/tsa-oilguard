from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilguard.config import load_config
from oilguard.data import build_event_dataset, sensor_quality
from oilguard.evaluation import forecast_metrics
from oilguard.forecasting import ForecastRun
from oilguard.warning import apply_cooldown, calibrate_threshold, evaluate_warning_run, run_onsets


def test_forecast_metrics_exact_values() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 3.0, 2.0, 5.0])
    metrics = forecast_metrics(actual, predicted)
    assert metrics["n"] == 4
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.75))
    assert metrics["bias"] == pytest.approx(0.25)


def test_run_onsets_respects_persistence() -> None:
    above = np.array([False, True, True, True, False, True, True, False])
    assert run_onsets(above, 3).tolist() == [1]
    assert run_onsets(above, 2).tolist() == [1, 5]
    assert run_onsets(above, 1).tolist() == [1, 5]


def test_cooldown_filters_repeated_alarms() -> None:
    onsets = np.array([0, 10, 30, 45])
    assert apply_cooldown(onsets, 20).tolist() == [0, 30]


def test_calibrate_threshold_uses_quantile_and_floor() -> None:
    residuals = np.array([-1.0, 2.0, -3.0, 4.0])
    assert calibrate_threshold(residuals, 0.5, 0.1) == pytest.approx(3.0)
    assert calibrate_threshold(residuals, 0.99, 0.1) == pytest.approx(4.0)
    assert calibrate_threshold(residuals, 0.99, 5.0) == pytest.approx(5.0)


def test_sensor_quality_flags_invalid_series() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="1s")
    absent = pd.Series([np.nan, np.nan, np.nan, np.nan], index=index)
    all_zero = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
    constant = pd.Series([2.0, 2.0, 2.0, 2.0], index=index)
    assert sensor_quality(absent, "P", 0.2, 600.0).status == "absent"
    assert "all-zero" in sensor_quality(all_zero, "P", 0.2, 600.0).status
    assert "constant" in sensor_quality(constant, "P", 0.2, 600.0).status
    assert not sensor_quality(constant, "P", 0.2, 600.0).usable


def test_sensor_quality_flags_long_gap() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-01 00:00:01"),
            pd.Timestamp("2020-01-01 00:00:02"),
            pd.Timestamp("2020-01-01 00:10:00"),
        ]
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    quality = sensor_quality(series, "P", 0.2, 60.0)
    assert "long-gap" in quality.status
    assert not quality.usable


def test_warning_metrics_separate_pre_event_false_alarms() -> None:
    periods = 100
    index = pd.date_range("2020-01-01", periods=periods, freq="10s")
    class_values = np.zeros(periods)
    class_values[60:80] = 102
    class_values[80:] = 2
    frame = pd.DataFrame({"class": class_values}, index=index)
    residuals = np.zeros(periods)
    residuals[[10, 50, 55, 70, 80]] = 2.0
    run = ForecastRun(
        name="test",
        order=(0, 0, 0),
        start_pos=0,
        pred=np.zeros(periods),
        resid=residuals,
        fit_seconds=0.0,
        aic=0.0,
        converged=True,
        n_train=periods,
    )
    event = SimpleNamespace(
        event_id="class_2/test",
        stem="test",
        well_id="WELL-00099",
        frame=frame,
        resample_seconds=10,
        holdout_start_pos=0,
        transient_pos=60,
        steady_pos=80,
        transient_onset=index[60],
        steady_onset=index[80],
    )
    metrics = evaluate_warning_run(
        run=run,
        event=event,
        threshold=1.0,
        persistence=1,
        cooldown_minutes=0.2,
        detection_grace_minutes=1.0,
        pre_event_grace_minutes=1.0,
    )
    assert metrics["false_alarms"] == 3
    assert metrics["false_alarms_adjusted"] == 2
    assert metrics["pre_event_warnings"] == 1
    assert metrics["normal_hours"] == pytest.approx(0.1667)
    assert metrics["normal_hours_adjusted"] == pytest.approx(0.15)
    assert metrics["false_alarm_rate_adjusted"] == pytest.approx(2 / (54 * 10 / 3600))
    assert metrics["transient_early_detected"] is True
    assert metrics["lead_transient_minutes"] == pytest.approx(500 / 60)
    assert metrics["steady_early_detected"] is True
    assert metrics["lead_steady_minutes"] == pytest.approx(100 / 60)
    assert metrics["warning_precision"] == pytest.approx(2 / 5)
    assert metrics["warning_precision_adjusted"] == pytest.approx(2 / 4)


def test_build_event_dataset_from_synthetic_parquet(tmp_path: Path) -> None:
    config = load_config()
    rng = np.random.default_rng(7)
    periods = 1800
    index = pd.date_range("2020-01-01", periods=periods, freq="1s")
    class_values = np.full(periods, np.nan)
    class_values[100:800] = 0
    class_values[800:1200] = 102
    class_values[1200:] = 2
    target = 10_000_000.0 + np.sin(np.arange(periods) / 30.0) * 50_000.0 + rng.normal(0, 1000, periods)
    exog = 60.0 + np.sin(np.arange(periods) / 60.0) * 2.0 + rng.normal(0, 0.05, periods)
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "P-TPT": target,
            "T-TPT": exog,
            "class": class_values,
        }
    )
    path = tmp_path / "WELL-00099_20200101000000.parquet"
    frame.to_parquet(path, index=False)

    event = build_event_dataset(path, config, event_id="class_2/test", stem="test", well_id="WELL-00099")

    assert event.usable
    assert event.transient_onset == index[800]
    assert event.steady_onset == index[1200]
    assert event.normal_samples >= config.preprocessing.min_normal_samples
    assert event.train_end_pos <= event.calib_start_pos
    assert event.calib_end_pos == event.holdout_start_pos
    assert event.holdout_start_pos < event.transient_pos
    assert event.steady_pos is not None
    assert event.steady_minutes > 0
    assert event.frame["target"].max() < 20.0
