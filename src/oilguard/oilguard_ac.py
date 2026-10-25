"""OILGUARD-AC: Alarm-Constrained, Multisensor Early Warning for Spurious DHSV Closure.

This module implements the proposed novel extension:
1. Multisensor composite anomaly scoring combining pressure forecast residuals (P-TPT)
   with downhole temperature deviation evidence (T-TPT).
2. False-alarm-budgeted threshold calibration on normal operating calibration splits.
3. Benchmark comparator: Local Outlier Factor (LOF) fitted on calibration normal operations.
4. Leak-free Leave-One-Event-Out Cross-Validation (LOEO-CV) harness.
5. Bootstrap confidence intervals for statistical uncertainty quantification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("oilguard.ac")


@dataclass
class EventScoreData:
    event_id: str
    stem: str
    index: pd.DatetimeIndex
    target: np.ndarray          # P-TPT (MPa)
    exog: np.ndarray            # T-TPT (degC)
    classes: np.ndarray         # 0, 102, 2
    split: np.ndarray          # 'train', 'calib', 'test'
    resid_arima: np.ndarray
    resid_sarimax: np.ndarray
    resid_naive: np.ndarray
    transient_onset: pd.Timestamp
    steady_onset: pd.Timestamp | None
    calib_start_idx: int
    calib_end_idx: int
    test_start_idx: int
    normal_hours: float


def load_event_scores(parquet_path: str | Path, event_id: str = "") -> EventScoreData:
    """Load an event's processed parquet and extract timing and residual arrays."""
    path = Path(parquet_path)
    df = pd.read_parquet(path)
    stem = path.stem
    index = df.index

    # Locate class transitions
    transient_mask = df["class"] == 102
    steady_mask = df["class"] == 2

    transient_onset = index[transient_mask][0] if transient_mask.any() else index[-1]
    steady_onset = index[steady_mask][0] if steady_mask.any() else None

    # Split boundaries
    calib_mask = df["split"].isin(["calibration", "calib"])
    eval_mask = df["split"].isin(["holdout", "test", "transient", "steady"])

    calib_indices = np.where(calib_mask)[0]
    eval_indices = np.where(eval_mask)[0]

    calib_start_idx = int(calib_indices[0]) if len(calib_indices) > 0 else 0
    calib_end_idx = int(calib_indices[-1] + 1) if len(calib_indices) > 0 else 0
    test_start_idx = int(eval_indices[0]) if len(eval_indices) > 0 else calib_end_idx

    # Normal operating hours during evaluation split (split in holdout/test and before transient)
    eval_normal_mask = df["split"].isin(["holdout", "test"]) & (df["class"] == 0) & (index < transient_onset)
    normal_seconds = eval_normal_mask.sum() * 10.0
    normal_hours = normal_seconds / 3600.0

    return EventScoreData(
        event_id=event_id or f"class_2/{stem}",
        stem=stem,
        index=index,
        target=df["target"].to_numpy(dtype=float),
        exog=df["exog"].to_numpy(dtype=float),
        classes=df["class"].to_numpy(dtype=int),
        split=df["split"].to_numpy(),
        resid_arima=df["resid_arima"].to_numpy(dtype=float),
        resid_sarimax=df["resid_sarimax"].to_numpy(dtype=float),
        resid_naive=df["resid_naive"].to_numpy(dtype=float),
        transient_onset=transient_onset,
        steady_onset=steady_onset,
        calib_start_idx=calib_start_idx,
        calib_end_idx=calib_end_idx,
        test_start_idx=test_start_idx,
        normal_hours=normal_hours,
    )


def compute_multisensor_score(
    event: EventScoreData,
    model: str = "arima",
    weight_pressure: float = 0.7,
) -> tuple[np.ndarray, float]:
    """Compute normalized composite score combining pressure residual and temperature deviation.

    Returns:
        (composite_score_array, calib_threshold_floor)
    """
    if model == "arima":
        raw_resid = np.abs(event.resid_arima)
    elif model == "sarimax":
        raw_resid = np.abs(event.resid_sarimax)
    elif model == "naive":
        raw_resid = np.abs(event.resid_naive)
    else:
        raise ValueError(f"Unknown model: {model}")

    calib_slice = slice(event.calib_start_idx, event.calib_end_idx)
    calib_resid = raw_resid[calib_slice]
    calib_temp = event.exog[calib_slice]

    # Pressure residual standardization
    sigma_resid = float(np.nanstd(calib_resid))
    sigma_resid = max(sigma_resid, 1e-4)
    z_pressure = raw_resid / sigma_resid

    # Temperature deviation standardization
    mu_temp = float(np.nanmean(calib_temp))
    sigma_temp = float(np.nanstd(calib_temp))
    sigma_temp = max(sigma_temp, 1e-4)
    z_temp = np.abs(event.exog - mu_temp) / sigma_temp

    # Composite score
    w = np.clip(weight_pressure, 0.0, 1.0)
    composite = np.sqrt(w * (z_pressure ** 2) + (1.0 - w) * (z_temp ** 2))
    return composite, 1e-4


def fit_predict_lof(
    event: EventScoreData,
    n_neighbors: int = 20,
) -> np.ndarray:
    """Fit Local Outlier Factor on calibration data and return positive outlier scores."""
    calib_slice = slice(event.calib_start_idx, event.calib_end_idx)
    X_raw = np.column_stack([event.target, event.exog])
    X_calib = X_raw[calib_slice]

    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib)
    X_scaled = scaler.transform(X_raw)

    n_samples = len(X_calib_scaled)
    k = min(n_neighbors, max(2, n_samples - 1))
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
    lof.fit(X_calib_scaled)

    # decision_function: large positive = inlier, large negative = outlier
    scores = -lof.decision_function(X_scaled)
    # shift to positive domain
    scores = np.maximum(scores - np.nanmin(scores[:event.calib_end_idx]), 0.0)
    return scores


def find_alarm_episodes(
    score: np.ndarray,
    threshold: float,
    persistence: int = 1,
    cooldown_samples: int = 12,
) -> list[int]:
    """Identify alarm trigger sample indices under persistence and cooldown constraints."""
    above = score >= threshold
    n = len(score)
    triggers: list[int] = []
    run_length = 0
    last_trigger = -999999

    for i in range(n):
        if above[i]:
            run_length += 1
            if run_length == persistence:
                if (i - last_trigger) >= cooldown_samples:
                    triggers.append(i)
                    last_trigger = i
        else:
            run_length = 0
    return triggers


def calibrate_budgeted_threshold(
    score: np.ndarray,
    calib_start_idx: int,
    calib_end_idx: int,
    budget_far_per_hour: float,
    persistence: int = 1,
    cooldown_minutes: float = 2.0,
    floor: float = 1e-4,
) -> float:
    """Calibrate the lowest score threshold such that calibration false alarms <= budget."""
    cooldown_samples = max(1, int(round((cooldown_minutes * 60.0) / 10.0)))
    calib_score = score[calib_start_idx:calib_end_idx]
    calib_hours = (len(calib_score) * 10.0) / 3600.0
    max_allowed_alarms = int(np.floor(budget_far_per_hour * calib_hours))

    if len(calib_score) == 0:
        return floor

    # Candidate quantiles from 0.80 to 1.0 plus an upper bound exceeding max
    quantiles = np.linspace(0.80, 1.0, 101)
    candidate_thresholds = np.unique(np.quantile(calib_score, quantiles))
    max_val = float(np.nanmax(calib_score)) if len(calib_score) > 0 else floor
    candidate_thresholds = np.unique(np.append(candidate_thresholds, max_val + 1e-4))
    candidate_thresholds = np.sort(candidate_thresholds)

    selected_threshold = float(candidate_thresholds[-1])
    for thresh in candidate_thresholds:
        triggers = find_alarm_episodes(
            calib_score,
            thresh,
            persistence=persistence,
            cooldown_samples=cooldown_samples,
        )
        if len(triggers) <= max_allowed_alarms:
            selected_threshold = float(thresh)
            break

    return max(selected_threshold, floor)


def evaluate_event_early_warning(
    event: EventScoreData,
    score: np.ndarray,
    threshold: float,
    persistence: int = 1,
    cooldown_minutes: float = 2.0,
    pre_event_window_minutes: float = 10.0,
) -> dict[str, Any]:
    """Evaluate early warning metrics on an event under the audited pre-event window policy."""
    cooldown_samples = max(1, int(round((cooldown_minutes * 60.0) / 10.0)))
    
    # We evaluate only prospective observations (test split onwards)
    eval_slice_start = event.test_start_idx
    eval_score = score[eval_slice_start:]
    eval_index = event.index[eval_slice_start:]
    
    alarm_rel_indices = find_alarm_episodes(
        eval_score,
        threshold,
        persistence=persistence,
        cooldown_samples=cooldown_samples,
    )
    alarm_timestamps = [eval_index[i] for i in alarm_rel_indices]

    pre_event_window_start = event.transient_onset - pd.Timedelta(minutes=pre_event_window_minutes)

    ordinary_false_alarms = 0
    pre_event_alarms: list[pd.Timestamp] = []
    post_onset_alarms: list[pd.Timestamp] = []

    for ts in alarm_timestamps:
        if ts < pre_event_window_start:
            ordinary_false_alarms += 1
        elif pre_event_window_start <= ts < event.transient_onset:
            pre_event_alarms.append(ts)
        else:
            post_onset_alarms.append(ts)

    early_detected = len(pre_event_alarms) > 0
    if early_detected:
        first_pre_event = pre_event_alarms[0]
        lead_time_minutes = (event.transient_onset - first_pre_event).total_seconds() / 60.0
    else:
        lead_time_minutes = 0.0

    return {
        "event_id": event.event_id,
        "stem": event.stem,
        "threshold": threshold,
        "persistence": persistence,
        "early_detected": early_detected,
        "lead_time_minutes": lead_time_minutes,
        "ordinary_false_alarms": ordinary_false_alarms,
        "normal_hours": event.normal_hours,
        "pre_event_alarms_count": len(pre_event_alarms),
        "post_onset_alarms_count": len(post_onset_alarms),
    }


def evaluate_cohort(
    events: list[EventScoreData],
    model: str,
    weight_pressure: float,
    budget_far_per_hour: float,
    persistence: int,
    cooldown_minutes: float = 2.0,
    pre_event_window_minutes: float = 10.0,
) -> dict[str, Any]:
    """Evaluate a specific configuration across all events in a cohort."""
    total_events = len(events)
    early_hits = 0
    lead_times: list[float] = []
    total_fa = 0
    total_normal_hours = 0.0

    for ev in events:
        if model == "lof":
            score = fit_predict_lof(ev)
            floor = 1e-4
        else:
            score, floor = compute_multisensor_score(ev, model=model, weight_pressure=weight_pressure)

        thresh = calibrate_budgeted_threshold(
            score=score,
            calib_start_idx=ev.calib_start_idx,
            calib_end_idx=ev.calib_end_idx,
            budget_far_per_hour=budget_far_per_hour,
            persistence=persistence,
            cooldown_minutes=cooldown_minutes,
            floor=floor,
        )

        res = evaluate_event_early_warning(
            event=ev,
            score=score,
            threshold=thresh,
            persistence=persistence,
            cooldown_minutes=cooldown_minutes,
            pre_event_window_minutes=pre_event_window_minutes,
        )

        if res["early_detected"]:
            early_hits += 1
            lead_times.append(res["lead_time_minutes"])
        total_fa += res["ordinary_false_alarms"]
        total_normal_hours += res["normal_hours"]

    detection_rate = early_hits / total_events if total_events > 0 else 0.0
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    realized_far = total_fa / total_normal_hours if total_normal_hours > 0 else 0.0

    return {
        "model": model,
        "weight_pressure": weight_pressure,
        "budget_far": budget_far_per_hour,
        "persistence": persistence,
        "total_events": total_events,
        "early_hits": early_hits,
        "detection_rate": detection_rate,
        "mean_lead_minutes": mean_lead,
        "median_lead_minutes": median_lead,
        "total_false_alarms": total_fa,
        "total_normal_hours": total_normal_hours,
        "realized_far_per_hour": realized_far,
    }


def run_loeo_cv(
    events: list[EventScoreData],
    model: str,
    weight_pressure: float,
    budget_far_per_hour: float,
    candidate_persistence: list[int] = [1, 2, 3],
    cooldown_minutes: float = 2.0,
    pre_event_window_minutes: float = 10.0,
) -> dict[str, Any]:
    """Execute strict Leave-One-Event-Out Cross-Validation (LOEO-CV) across all events."""
    n_events = len(events)
    fold_results: list[dict[str, Any]] = []

    for held_out_idx in range(n_events):
        held_out_event = events[held_out_idx]
        training_events = [events[i] for i in range(n_events) if i != held_out_idx]

        # 1. On training_events, select optimal persistence under the budget
        best_p = candidate_persistence[0]
        best_score_metric = -1.0

        for p in candidate_persistence:
            train_eval = evaluate_cohort(
                events=training_events,
                model=model,
                weight_pressure=weight_pressure,
                budget_far_per_hour=budget_far_per_hour,
                persistence=p,
                cooldown_minutes=cooldown_minutes,
                pre_event_window_minutes=pre_event_window_minutes,
            )
            # Objective: prioritize detection rate under budget
            score_metric = train_eval["detection_rate"] * 100.0 - train_eval["realized_far_per_hour"]
            if score_metric > best_score_metric:
                best_score_metric = score_metric
                best_p = p

        # 2. Evaluate held-out event using best_p with threshold calibrated on held_out's own calib data
        if model == "lof":
            score = fit_predict_lof(held_out_event)
            floor = 1e-4
        else:
            score, floor = compute_multisensor_score(
                held_out_event,
                model=model,
                weight_pressure=weight_pressure,
            )

        thresh = calibrate_budgeted_threshold(
            score=score,
            calib_start_idx=held_out_event.calib_start_idx,
            calib_end_idx=held_out_event.calib_end_idx,
            budget_far_per_hour=budget_far_per_hour,
            persistence=best_p,
            cooldown_minutes=cooldown_minutes,
            floor=floor,
        )

        res = evaluate_event_early_warning(
            event=held_out_event,
            score=score,
            threshold=thresh,
            persistence=best_p,
            cooldown_minutes=cooldown_minutes,
            pre_event_window_minutes=pre_event_window_minutes,
        )
        res["selected_persistence"] = best_p
        fold_results.append(res)

    early_hits = sum(1 for r in fold_results if r["early_detected"])
    lead_times = [r["lead_time_minutes"] for r in fold_results if r["early_detected"]]
    total_fa = sum(r["ordinary_false_alarms"] for r in fold_results)
    total_normal_hours = sum(r["normal_hours"] for r in fold_results)

    detection_rate = early_hits / n_events if n_events > 0 else 0.0
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    realized_far = total_fa / total_normal_hours if total_normal_hours > 0 else 0.0

    return {
        "model": model,
        "weight_pressure": weight_pressure,
        "budget_far": budget_far_per_hour,
        "total_events": n_events,
        "early_hits": early_hits,
        "unbiased_detection_rate": detection_rate,
        "unbiased_mean_lead_minutes": mean_lead,
        "unbiased_median_lead_minutes": median_lead,
        "total_false_alarms": total_fa,
        "total_normal_hours": total_normal_hours,
        "unbiased_realized_far": realized_far,
        "fold_results": fold_results,
    }


def compute_bootstrap_ci(
    fold_results: list[dict[str, Any]],
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals across event evaluation folds."""
    rng = np.random.default_rng(random_seed)
    n = len(fold_results)
    det_rates = []
    leads = []
    fars = []

    for _ in range(n_bootstraps):
        sample_indices = rng.integers(0, n, size=n)
        sample = [fold_results[i] for i in sample_indices]
        hits = sum(1 for r in sample if r["early_detected"])
        sample_leads = [r["lead_time_minutes"] for r in sample if r["early_detected"]]
        sample_fa = sum(r["ordinary_false_alarms"] for r in sample)
        sample_nh = sum(r["normal_hours"] for r in sample)

        det_rates.append(hits / n)
        leads.append(float(np.mean(sample_leads)) if sample_leads else 0.0)
        fars.append(sample_fa / sample_nh if sample_nh > 0 else 0.0)

    return {
        "detection_rate_ci_95": (float(np.percentile(det_rates, 2.5)), float(np.percentile(det_rates, 97.5))),
        "mean_lead_ci_95": (float(np.percentile(leads, 2.5)), float(np.percentile(leads, 97.5))),
        "realized_far_ci_95": (float(np.percentile(fars, 2.5)), float(np.percentile(fars, 97.5))),
    }
