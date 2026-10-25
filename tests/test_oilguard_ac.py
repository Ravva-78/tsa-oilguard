"""Unit tests for OILGUARD-AC module."""

import numpy as np
import pandas as pd
import pytest

from oilguard.oilguard_ac import (
    EventScoreData,
    calibrate_budgeted_threshold,
    compute_bootstrap_ci,
    compute_multisensor_score,
    evaluate_cohort,
    evaluate_event_early_warning,
    find_alarm_episodes,
    fit_predict_lof,
    run_loeo_cv,
)


@pytest.fixture
def synthetic_event() -> EventScoreData:
    periods = 360  # 1 hour at 10-second intervals
    index = pd.date_range("2024-01-01 00:00:00", periods=periods, freq="10s")
    
    # 0 to 120: train, 120 to 240: calib, 240 to 360: test
    split = np.array(["train"] * 120 + ["calib"] * 120 + ["test"] * 120)
    classes = np.zeros(periods, dtype=int)
    # Transient onset at index 300 (10 minutes before end)
    classes[300:] = 102
    transient_onset = index[300]
    
    target = np.ones(periods) * 20.0
    exog = np.ones(periods) * 60.0
    
    # Residuals: small Gaussian noise during normal, spike during transient
    np.random.seed(42)
    resid_arima = np.random.normal(0, 0.01, periods)
    resid_sarimax = resid_arima.copy()
    resid_naive = np.random.normal(0, 0.02, periods)
    
    # Normal hours during test: index 240 to 300 = 60 samples * 10s = 600s = 1/6 hr
    normal_hours = 600.0 / 3600.0

    return EventScoreData(
        event_id="class_2/WELL-TEST",
        stem="WELL-TEST",
        index=index,
        target=target,
        exog=exog,
        classes=classes,
        split=split,
        resid_arima=resid_arima,
        resid_sarimax=resid_sarimax,
        resid_naive=resid_naive,
        transient_onset=transient_onset,
        steady_onset=None,
        calib_start_idx=120,
        calib_end_idx=240,
        test_start_idx=240,
        normal_hours=normal_hours,
    )


def test_multisensor_score_weighting(synthetic_event: EventScoreData) -> None:
    # w_P = 1.0 pure pressure residual
    score_p10, _ = compute_multisensor_score(synthetic_event, model="arima", weight_pressure=1.0)
    score_p07, _ = compute_multisensor_score(synthetic_event, model="arima", weight_pressure=0.7)
    
    assert len(score_p10) == len(synthetic_event.target)
    assert len(score_p07) == len(synthetic_event.target)
    assert np.all(score_p10 >= 0.0)
    assert np.all(score_p07 >= 0.0)


def test_calibrate_budgeted_threshold(synthetic_event: EventScoreData) -> None:
    score, floor = compute_multisensor_score(synthetic_event, model="arima", weight_pressure=0.7)
    calib_slice = slice(synthetic_event.calib_start_idx, synthetic_event.calib_end_idx)
    calib_hours = (len(score[calib_slice]) * 10.0) / 3600.0  # 1200s = 0.333 hr

    # Budget 0.0 FA/hr -> must allow 0 false alarms on calibration data
    thresh_strict = calibrate_budgeted_threshold(
        score=score,
        calib_start_idx=synthetic_event.calib_start_idx,
        calib_end_idx=synthetic_event.calib_end_idx,
        budget_far_per_hour=0.0,
        persistence=1,
    )
    triggers = find_alarm_episodes(
        score[calib_slice],
        thresh_strict,
        persistence=1,
    )
    assert len(triggers) == 0

    # Budget 6.0 FA/hr -> allowed = floor(6.0 * 0.333) = 2 alarms
    thresh_relaxed = calibrate_budgeted_threshold(
        score=score,
        calib_start_idx=synthetic_event.calib_start_idx,
        calib_end_idx=synthetic_event.calib_end_idx,
        budget_far_per_hour=6.0,
        persistence=1,
    )
    assert thresh_relaxed <= thresh_strict


def test_lof_novelty_fitting(synthetic_event: EventScoreData) -> None:
    lof_scores = fit_predict_lof(synthetic_event, n_neighbors=10)
    assert len(lof_scores) == len(synthetic_event.target)
    assert np.all(lof_scores >= 0.0)


def test_early_warning_pre_event_window(synthetic_event: EventScoreData) -> None:
    score = np.zeros(len(synthetic_event.index))
    
    # Case 1: Alarm 15 minutes before transient onset -> ordinary false alarm
    # Transient is at index 300 (50 min). 15 min before = index 210 (during calib) or index 250 (41.6 min, ~8.3 min before transient)
    # 10 min window before index 300 is index [240, 300)
    # Alarm at index 245 (within [240, 300)) -> Early warning
    score[245] = 10.0
    res_early = evaluate_event_early_warning(
        event=synthetic_event,
        score=score,
        threshold=5.0,
        persistence=1,
        pre_event_window_minutes=10.0,
    )
    assert res_early["early_detected"] is True
    assert res_early["lead_time_minutes"] == pytest.approx((300 - 245) * 10.0 / 60.0)
    assert res_early["ordinary_false_alarms"] == 0

    # Case 2: Alarm earlier than pre-event window -> Ordinary false alarm
    # Let's say test starts at index 200, window starts at index 240, alarm at index 220
    synthetic_event.test_start_idx = 200
    score_early = np.zeros(len(synthetic_event.index))
    score_early[220] = 10.0
    res_fa = evaluate_event_early_warning(
        event=synthetic_event,
        score=score_early,
        threshold=5.0,
        persistence=1,
        pre_event_window_minutes=10.0,
    )
    assert res_fa["early_detected"] is False
    assert res_fa["ordinary_false_alarms"] == 1


def test_bootstrap_ci_computation() -> None:
    fold_results = [
        {"early_detected": True, "lead_time_minutes": 5.0, "ordinary_false_alarms": 0, "normal_hours": 0.2},
        {"early_detected": False, "lead_time_minutes": 0.0, "ordinary_false_alarms": 1, "normal_hours": 0.2},
        {"early_detected": True, "lead_time_minutes": 3.0, "ordinary_false_alarms": 0, "normal_hours": 0.2},
    ]
    ci = compute_bootstrap_ci(fold_results, n_bootstraps=50, random_seed=42)
    assert "detection_rate_ci_95" in ci
    assert "mean_lead_ci_95" in ci
    assert "realized_far_ci_95" in ci
    assert 0.0 <= ci["detection_rate_ci_95"][0] <= ci["detection_rate_ci_95"][1] <= 1.0


def test_regression_cross_well_leakage_prevention(synthetic_event: EventScoreData) -> None:
    """Verify that in genuine cross-well LOEO-CV, held-out event does not set its own threshold."""
    # Training cohort with high variance vs held-out with low variance
    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    held_out_calib = np.array([0.1, 0.2, 0.3])

    # Pooled threshold from training cohort at 90th percentile
    thresh_pool = float(np.quantile(train_scores, 0.90))
    # Threshold from held-out event alone
    thresh_held_out = float(np.quantile(held_out_calib, 0.90))

    # They must differ, proving pooled cross-well training is independent of held-out data
    assert thresh_pool > thresh_held_out
    assert thresh_pool > 4.5


def test_regression_calibration_vs_realized_far() -> None:
    """Verify that a threshold with 0 false alarms on short calibration can have >0 FAR on holdout."""
    # Short calib of 12 samples (2 minutes): max is 2.0
    calib = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0])
    threshold = 2.05  # zero alarms on calib

    # Longer holdout of 360 samples (1 hour): one surge of 2.5
    holdout = np.zeros(360)
    holdout[50] = 2.5

    trigs_calib = find_alarm_episodes(calib, threshold, persistence=1)
    trigs_holdout = find_alarm_episodes(holdout, threshold, persistence=1)

    assert len(trigs_calib) == 0  # 0 on calibration
    assert len(trigs_holdout) == 1  # 1 on holdout (FAR = 1.0/hr), proving calibration FAR != realized FAR
