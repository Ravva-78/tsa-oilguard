from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import EventDataset
from .forecasting import ForecastRun


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    n = len(actual)
    if n == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "r2": np.nan,
            "bias": np.nan,
        }
    error = predicted - actual
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error**2)))
    nonzero = np.abs(actual) > 1e-9
    mape = (
        float(np.mean(abs_error[nonzero] / np.abs(actual[nonzero])) * 100.0)
        if nonzero.any()
        else np.nan
    )
    denominator = np.abs(actual) + np.abs(predicted)
    smape_mask = denominator > 1e-9
    smape = (
        float(np.mean(2.0 * abs_error[smape_mask] / denominator[smape_mask]) * 100.0)
        if smape_mask.any()
        else np.nan
    )
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / variance) if variance > 0 else np.nan
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "bias": float(np.mean(error)),
    }


def forecast_metric_rows(run: ForecastRun, event: EventDataset) -> list[dict[str, Any]]:
    frame = event.frame
    actual = frame["target"].to_numpy(dtype=float)
    predicted = np.full(len(actual), np.nan, dtype=float)
    predicted[run.start_pos :] = run.pred
    periods = {
        "validation": (event.calib_start_pos, event.calib_end_pos),
        "test": (event.holdout_start_pos, len(actual)),
        "all_eval": (event.calib_start_pos, len(actual)),
    }
    rows: list[dict[str, Any]] = []
    for period, (start, end) in periods.items():
        metrics = forecast_metrics(actual[start:end], predicted[start:end])
        rows.append(
            {
                "model": run.name,
                "event_id": event.event_id,
                "stem": event.stem,
                "well_id": event.well_id,
                "period": period,
                "order": str(run.order),
                "aic": run.aic,
                "fit_seconds": run.fit_seconds,
                "converged": run.converged,
                "n_train": run.n_train,
                **metrics,
            }
        )
    return rows


def summarize_forecast_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        "n": ["mean", "sum"],
        "mae": ["mean", "median", "std"],
        "rmse": ["mean", "median", "std"],
        "mape": ["mean", "median"],
        "smape": ["mean", "median"],
        "r2": ["mean", "median"],
        "bias": ["mean", "median"],
    }
    summary = metrics.groupby(["model", "period"]).agg(aggregation)
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def summarize_warning_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=False):
        steady = group[group["has_steady"].fillna(False).astype(bool)]
        transient_early = group["transient_early_detected"].fillna(False).astype(bool)
        steady_early = steady["steady_early_detected"].fillna(False).astype(bool)
        lead_transient = group.loc[transient_early, "lead_transient_minutes"].dropna()
        lead_steady = steady.loc[steady_early, "lead_steady_minutes"].dropna()
        total_normal_hours = float(group["normal_hours"].sum())
        total_normal_hours_adjusted = float(group["normal_hours_adjusted"].sum())
        total_false_alarms = int(group["false_alarms"].sum())
        total_false_alarms_adjusted = int(group["false_alarms_adjusted"].sum())
        rows.append(
            {
                "model": model,
                "n_events": len(group),
                "n_steady": len(steady),
                "transient_early_rate": float(transient_early.mean()),
                "transient_grace_rate": float(group["transient_detected_with_grace"].mean()),
                "steady_early_rate": float(steady_early.mean()) if len(steady) else np.nan,
                "steady_grace_rate": float(steady["steady_detected_with_grace"].mean())
                if len(steady)
                else np.nan,
                "mean_lead_transient_minutes": float(lead_transient.mean())
                if len(lead_transient)
                else np.nan,
                "median_lead_transient_minutes": float(lead_transient.median())
                if len(lead_transient)
                else np.nan,
                "q25_lead_transient_minutes": float(lead_transient.quantile(0.25))
                if len(lead_transient)
                else np.nan,
                "q75_lead_transient_minutes": float(lead_transient.quantile(0.75))
                if len(lead_transient)
                else np.nan,
                "mean_lead_steady_minutes": float(lead_steady.mean())
                if len(lead_steady)
                else np.nan,
                "median_lead_steady_minutes": float(lead_steady.median())
                if len(lead_steady)
                else np.nan,
                "total_false_alarms": total_false_alarms,
                "total_normal_hours": total_normal_hours,
                "false_alarm_rate_per_hour": float(total_false_alarms / total_normal_hours)
                if total_normal_hours > 0
                else np.nan,
                "mean_event_false_alarm_rate": float(group["false_alarm_rate_per_hour"].mean()),
                "median_event_false_alarm_rate": float(group["false_alarm_rate_per_hour"].median()),
                "total_false_alarms_adjusted": total_false_alarms_adjusted,
                "total_normal_hours_adjusted": total_normal_hours_adjusted,
                "false_alarm_rate_adjusted": float(
                    total_false_alarms_adjusted / total_normal_hours_adjusted
                )
                if total_normal_hours_adjusted > 0
                else np.nan,
                "mean_event_false_alarm_rate_adjusted": float(
                    group["false_alarm_rate_adjusted"].mean()
                ),
                "median_event_false_alarm_rate_adjusted": float(
                    group["false_alarm_rate_adjusted"].median()
                ),
                "total_pre_event_warnings": int(group["pre_event_warnings"].sum()),
                "mean_warning_precision": float(group["warning_precision"].mean()),
                "mean_warning_precision_adjusted": float(
                    group["warning_precision_adjusted"].mean()
                ),
                "total_warning_episodes": int(group["warning_episodes"].sum()),
                "total_abnormal_warning_episodes": int(group["abnormal_warning_episodes"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in sensitivity.groupby(["model", "quantile", "persistence"], sort=True):
        model, quantile, persistence = keys
        transient_early = group["transient_early_detected"].fillna(False).astype(bool)
        steady_rows = group[group["steady_early_detected"].notna()]
        steady_early = steady_rows["steady_early_detected"].astype(bool)
        lead_transient = group.loc[transient_early, "lead_transient_minutes"].dropna()
        lead_steady = steady_rows.loc[steady_early, "lead_steady_minutes"].dropna()
        rows.append(
            {
                "model": model,
                "quantile": float(quantile),
                "persistence": int(persistence),
                "n_events": len(group),
                "transient_early_rate": float(transient_early.mean()),
                "steady_early_rate": float(steady_early.mean())
                if group["steady_early_detected"].notna().any()
                else np.nan,
                "mean_lead_transient_minutes": float(lead_transient.mean())
                if len(lead_transient)
                else np.nan,
                "median_lead_transient_minutes": float(lead_transient.median())
                if len(lead_transient)
                else np.nan,
                "mean_lead_steady_minutes": float(lead_steady.mean())
                if len(lead_steady)
                else np.nan,
                "median_lead_steady_minutes": float(lead_steady.median())
                if len(lead_steady)
                else np.nan,
                "mean_false_alarm_rate_per_hour": float(
                    group["false_alarm_rate_per_hour"].mean()
                ),
                "mean_false_alarm_rate_adjusted": float(
                    group["false_alarm_rate_adjusted"].mean()
                ),
                "mean_warning_precision": float(group["warning_precision"].mean()),
                "mean_warning_precision_adjusted": float(
                    group["warning_precision_adjusted"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, ensure_ascii=False)


def save_dataframe(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
