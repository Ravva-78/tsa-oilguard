from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import EventDataset
from .forecasting import ForecastRun


def calibrate_threshold(
    calibration_residuals: np.ndarray,
    quantile: float,
    floor: float,
) -> float:
    values = np.abs(np.asarray(calibration_residuals, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(floor)
    threshold = float(np.quantile(values, quantile, method="higher"))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.nanmax(values))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(floor)
    return max(float(floor), threshold)


def run_onsets(above: np.ndarray, persistence: int) -> np.ndarray:
    above = np.asarray(above, dtype=bool)
    onsets: list[int] = []
    start: int | None = None
    for i, value in enumerate(above):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= persistence:
                onsets.append(start)
            start = None
    if start is not None and len(above) - start >= persistence:
        onsets.append(start)
    return np.asarray(onsets, dtype=int)


def apply_cooldown(onsets: np.ndarray, cooldown_samples: int) -> np.ndarray:
    if len(onsets) == 0:
        return onsets
    kept = [int(onsets[0])]
    for onset in onsets[1:]:
        if int(onset) - kept[-1] >= cooldown_samples:
            kept.append(int(onset))
    return np.asarray(kept, dtype=int)


def _phase(position: int, transient_pos: int, steady_pos: int | None) -> str:
    if position < transient_pos:
        return "normal"
    if steady_pos is not None and position >= steady_pos:
        return "steady"
    return "transient"


def evaluate_warning_run(
    run: ForecastRun,
    event: EventDataset,
    threshold: float,
    persistence: int,
    cooldown_minutes: float,
    detection_grace_minutes: float,
    pre_event_grace_minutes: float,
) -> dict[str, Any]:
    resample_seconds = event.resample_seconds
    cooldown_samples = max(1, round(cooldown_minutes * 60.0 / resample_seconds))
    grace_samples = max(0, round(detection_grace_minutes * 60.0 / resample_seconds))
    pre_event_grace_samples = max(0, round(pre_event_grace_minutes * 60.0 / resample_seconds))

    eval_offset = event.holdout_start_pos - run.start_pos
    eval_residuals = run.resid[eval_offset:]
    eval_positions = np.arange(event.holdout_start_pos, event.holdout_start_pos + len(eval_residuals))
    if len(eval_positions) != len(eval_residuals):
        raise ValueError("Warning evaluation positions do not match residuals")

    eval_class = event.frame["class"].to_numpy()
    eval_labels = eval_class[eval_positions]
    normal_mask = eval_labels == 0
    pre_event_start = event.transient_pos - pre_event_grace_samples

    above = np.abs(eval_residuals) > threshold
    onsets_relative = run_onsets(above, persistence)
    onsets_absolute = eval_positions[onsets_relative] if len(onsets_relative) else np.array([], dtype=int)

    normal_onsets = onsets_absolute[normal_mask[onsets_absolute - event.holdout_start_pos]]
    pre_event_onsets = normal_onsets[normal_onsets >= pre_event_start]
    false_alarm_onsets = normal_onsets[normal_onsets < pre_event_start]
    normal_episodes = apply_cooldown(normal_onsets, cooldown_samples)
    false_alarm_episodes = apply_cooldown(false_alarm_onsets, cooldown_samples)
    all_episodes = apply_cooldown(onsets_absolute, cooldown_samples)

    normal_hours = float(normal_mask.sum() * resample_seconds / 3600.0)
    normal_positions = eval_positions[normal_mask]
    adjusted_normal_samples = int((normal_positions < pre_event_start).sum())
    normal_hours_adjusted = float(adjusted_normal_samples * resample_seconds / 3600.0)
    false_alarms = len(normal_episodes)
    false_alarms_adjusted = len(false_alarm_episodes)
    false_alarm_rate = float(false_alarms / normal_hours) if normal_hours > 0 else np.nan
    false_alarm_rate_adjusted = (
        float(false_alarms_adjusted / normal_hours_adjusted)
        if normal_hours_adjusted > 0
        else np.nan
    )

    # Audited rule: An early warning must occur inside the accepted pre-event window [pre_event_start, transient_pos)
    transient_early = bool(len(pre_event_onsets) > 0)
    
    # Detections occurring after transient onset within detection grace window [transient_pos, transient_pos + grace_samples]
    post_transient_onsets = onsets_absolute[
        (onsets_absolute >= event.transient_pos)
        & (onsets_absolute <= event.transient_pos + grace_samples)
    ]
    transient_post_detected = bool(len(post_transient_onsets) > 0)
    transient_with_grace = transient_early or transient_post_detected

    steady_early = False
    steady_with_grace = False
    lead_transient_minutes = np.nan
    lead_steady_minutes = np.nan
    first_warning_time: pd.Timestamp | None = None
    first_warning_phase = "none"
    
    if len(onsets_absolute):
        first_position = int(onsets_absolute[0])
        first_warning_time = event.frame.index[first_position]
        first_warning_phase = _phase(first_position, event.transient_pos, event.steady_pos)
        
    if transient_early:
        lead_transient_minutes = float(
            (event.frame.index[event.transient_pos] - event.frame.index[int(pre_event_onsets[0])]).total_seconds()
            / 60.0
        )
        
    if event.steady_pos is not None:
        steady_onsets = onsets_absolute[
            (onsets_absolute >= event.transient_pos) & (onsets_absolute < event.steady_pos)
        ]
        steady_early = bool(len(steady_onsets))
        steady_with_grace = bool(
            (
                (onsets_absolute >= event.transient_pos)
                & (onsets_absolute <= event.steady_pos + grace_samples)
            ).any()
        )
        if len(steady_onsets):
            lead_steady_minutes = float(
                (
                    event.frame.index[event.steady_pos]
                    - event.frame.index[int(steady_onsets[0])]
                ).total_seconds()
                / 60.0
            )

    early_detected = steady_early if event.steady_pos is not None else transient_early
    early_target = "steady" if event.steady_pos is not None else "transient"
    early_lead_minutes = (
        lead_steady_minutes if event.steady_pos is not None else lead_transient_minutes
    )

    true_episodes = int((all_episodes >= event.transient_pos).sum())
    precision = float(true_episodes / (true_episodes + false_alarms)) if (true_episodes + false_alarms) else np.nan
    precision_adjusted = (
        float(true_episodes / (true_episodes + false_alarms_adjusted))
        if (true_episodes + false_alarms_adjusted)
        else np.nan
    )

    return {
        "model": run.name,
        "event_id": event.event_id,
        "stem": event.stem,
        "well_id": event.well_id,
        "threshold": float(threshold),
        "persistence": int(persistence),
        "cooldown_minutes": float(cooldown_minutes),
        "detection_grace_minutes": float(detection_grace_minutes),
        "pre_event_grace_minutes": float(pre_event_grace_minutes),
        "normal_hours": round(normal_hours, 4),
        "normal_hours_adjusted": round(normal_hours_adjusted, 4),
        "false_alarms": false_alarms,
        "false_alarms_adjusted": false_alarms_adjusted,
        "false_alarm_rate_per_hour": false_alarm_rate,
        "false_alarm_rate_adjusted": false_alarm_rate_adjusted,
        "pre_event_warnings": len(pre_event_onsets),
        "warning_episodes": len(all_episodes),
        "normal_warning_episodes": false_alarms,
        "normal_warning_episodes_adjusted": false_alarms_adjusted,
        "abnormal_warning_episodes": true_episodes,
        "warning_precision": precision,
        "warning_precision_adjusted": precision_adjusted,
        "transient_early_detected": transient_early,
        "transient_post_detected": transient_post_detected,
        "transient_detected_with_grace": transient_with_grace,
        "steady_early_detected": steady_early if event.steady_pos is not None else None,
        "steady_detected_with_grace": steady_with_grace if event.steady_pos is not None else None,
        "early_detected": bool(early_detected),
        "early_target": early_target,
        "early_lead_minutes": early_lead_minutes,
        "lead_transient_minutes": lead_transient_minutes,
        "lead_steady_minutes": lead_steady_minutes,
        "first_warning_time": first_warning_time,
        "first_warning_phase": first_warning_phase,
        "has_steady": event.steady_pos is not None,
        "transient_onset": event.transient_onset,
        "steady_onset": event.steady_onset,
    }


def threshold_sensitivity(
    run: ForecastRun,
    event: EventDataset,
    quantiles: tuple[float, ...],
    persistence_values: tuple[int, ...],
    cooldown_minutes: float,
    detection_grace_minutes: float,
    pre_event_grace_minutes: float,
    threshold_floor: float,
) -> list[dict[str, Any]]:
    calibration_length = event.calib_end_pos - event.calib_start_pos
    calibration_residuals = run.resid[:calibration_length]
    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        threshold = calibrate_threshold(calibration_residuals, quantile, threshold_floor)
        for persistence in persistence_values:
            metrics = evaluate_warning_run(
                run=run,
                event=event,
                threshold=threshold,
                persistence=persistence,
                cooldown_minutes=cooldown_minutes,
                detection_grace_minutes=detection_grace_minutes,
                pre_event_grace_minutes=pre_event_grace_minutes,
            )
            rows.append(
                {
                    "model": run.name,
                    "event_id": event.event_id,
                    "stem": event.stem,
                    "well_id": event.well_id,
                    "quantile": quantile,
                    "persistence": persistence,
                    "threshold": threshold,
                    "early_detected": metrics["early_detected"],
                    "early_target": metrics["early_target"],
                    "early_lead_minutes": metrics["early_lead_minutes"],
                    "lead_transient_minutes": metrics["lead_transient_minutes"],
                    "lead_steady_minutes": metrics["lead_steady_minutes"],
                    "false_alarm_rate_per_hour": metrics["false_alarm_rate_per_hour"],
                    "false_alarm_rate_adjusted": metrics["false_alarm_rate_adjusted"],
                    "warning_precision": metrics["warning_precision"],
                    "warning_precision_adjusted": metrics["warning_precision_adjusted"],
                    "transient_early_detected": metrics["transient_early_detected"],
                    "steady_early_detected": metrics["steady_early_detected"],
                }
            )
    return rows
