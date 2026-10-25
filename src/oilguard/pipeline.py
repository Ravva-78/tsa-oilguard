from __future__ import annotations

import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from . import eda
from .config import ExperimentConfig, load_config, package_versions
from .data import EventDataset, build_candidate_table, build_event_dataset
from .evaluation import (
    forecast_metric_rows,
    save_dataframe,
    save_json,
    summarize_forecast_metrics,
    summarize_sensitivity,
    summarize_warning_metrics,
)
from .forecasting import (
    ForecastRun,
    rolling_forecast_arima,
    rolling_forecast_sarimax,
    rolling_naive,
    select_arima_order,
)
from .manifest import load_or_build_manifest
from .warning import calibrate_threshold, evaluate_warning_run, threshold_sensitivity


def setup_logging(log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("oilguard")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _safe_adf(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 20 or np.nanstd(values) == 0:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(values, autolag="AIC")[1])
    except Exception:
        return np.nan


def _safe_kpss(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 20 or np.nanstd(values) == 0:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(values, regression="c", nlags="auto")[1])
    except Exception:
        return np.nan


def stationarity_table(datasets: dict[str, EventDataset]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in datasets.values():
        train = event.frame["target"].iloc[: event.train_end_pos]
        diff = train.diff().dropna()
        rows.append(
            {
                "event_id": event.event_id,
                "stem": event.stem,
                "well_id": event.well_id,
                "n_train": len(train),
                "adf_p_level": _safe_adf(train),
                "adf_p_diff": _safe_adf(diff),
                "kpss_p_level": _safe_kpss(train),
            }
        )
    return pd.DataFrame(rows)


def choose_examples(
    datasets: dict[str, EventDataset],
    config: ExperimentConfig,
    max_examples: int,
) -> list[EventDataset]:
    if config.examples:
        selected = [
            datasets[f"{config.event_id}/{stem}"]
            for stem in config.examples
            if f"{config.event_id}/{stem}" in datasets
        ]
        if selected:
            return selected[:max_examples]

    events = list(datasets.values())
    steady = [event for event in events if event.steady_onset is not None]
    by_well: dict[str | None, list[EventDataset]] = {}
    for event in sorted(steady, key=lambda item: (str(item.well_id), item.transient_onset)):
        by_well.setdefault(event.well_id, []).append(event)

    selected: list[EventDataset] = []
    wells = sorted(by_well, key=lambda value: str(value))
    while len(selected) < max_examples:
        added = False
        for well in wells:
            if by_well[well]:
                selected.append(by_well[well].pop(0))
                added = True
                if len(selected) >= max_examples:
                    break
        if not added:
            break

    if len(selected) < max_examples:
        for event in sorted(events, key=lambda item: (str(item.well_id), item.transient_onset)):
            if event not in selected:
                selected.append(event)
            if len(selected) >= max_examples:
                break
    return selected


def _split_column(event: EventDataset) -> np.ndarray:
    split = np.full(len(event.frame), "train", dtype=object)
    split[event.calib_start_pos : event.calib_end_pos] = "calibration"
    split[event.holdout_start_pos : event.transient_pos] = "holdout"
    if event.steady_pos is not None:
        split[event.transient_pos : event.steady_pos] = "transient"
        split[event.steady_pos :] = "steady"
    else:
        split[event.transient_pos :] = "transient"
    return split


def save_event_predictions(
    event: EventDataset,
    runs: dict[str, ForecastRun],
    path: str | Path,
) -> None:
    frame = event.frame.copy()
    frame["split"] = _split_column(event)
    for name, run in runs.items():
        pred = np.full(len(frame), np.nan, dtype=float)
        resid = np.full(len(frame), np.nan, dtype=float)
        pred[run.start_pos :] = run.pred
        resid[run.start_pos :] = run.resid
        frame[f"pred_{name}"] = pred
        frame[f"resid_{name}"] = resid
    frame.index.name = "timestamp"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def run_pipeline(
    config_path: str | Path | None = None,
    refresh_manifest: bool = False,
    skip_eda: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    logger = setup_logging(config.logs_dir / "pipeline.log")
    started = time.perf_counter()
    np.random.seed(config.seed)
    logger.info("OILGUARD pipeline started")
    logger.info("Dataset root: %s", config.dataset_root)
    logger.info("Event: class %s %s", config.event.class_id, config.event.name)
    logger.info("Target: %s (%s), exogenous: %s (%s)", config.event.target, config.event.target_unit, config.event.exog, config.event.exog_unit)

    save_json(config.to_dict(), config.metrics_dir / "experiment_config.json")

    manifest_path = config.processed_dir / "manifest.parquet"
    manifest = load_or_build_manifest(
        dataset_root=config.dataset_root,
        output_path=manifest_path,
        summary_path=config.tables_dir / "manifest_summary_by_class.csv",
        target=config.event.target,
        exog=config.event.exog,
        refresh=refresh_manifest,
    )
    logger.info("Manifest: %s files, %.2f GB", len(manifest), manifest["size_mb"].sum() / 1000.0)

    candidates = build_candidate_table(config, manifest)
    save_dataframe(candidates, config.tables_dir / "class2_candidates.csv")
    gold_candidates = candidates[candidates["gold"].fillna(False).astype(bool)].copy()
    logger.info(
        "Class-2 real candidates: %s; gold cohort: %s",
        len(candidates),
        len(gold_candidates),
    )

    datasets: dict[str, EventDataset] = {}
    load_failures: list[dict[str, Any]] = []
    for row in gold_candidates.itertuples():
        try:
            event = build_event_dataset(
                path=Path(row.path),
                config=config,
                event_id=row.event_id,
                stem=row.stem,
                well_id=row.well_id,
            )
            if event.usable:
                datasets[event.event_id] = event
            else:
                load_failures.append(
                    {
                        "event_id": event.event_id,
                        "stem": event.stem,
                        "error": event.excluded_reason,
                    }
                )
        except Exception as exc:
            logger.exception("Failed to load %s", row.stem)
            load_failures.append(
                {"event_id": row.event_id, "stem": row.stem, "error": str(exc)}
            )
    logger.info("Loaded %s modeling events", len(datasets))
    if not datasets:
        raise RuntimeError("No usable events were loaded")

    examples = choose_examples(datasets, config, config.eda.max_examples)
    longest = max(datasets.values(), key=lambda event: event.normal_samples)
    normal_series = longest.frame.loc[
        longest.frame.index < longest.transient_onset, "target"
    ]
    stationarity = stationarity_table(datasets)
    save_dataframe(stationarity, config.tables_dir / "stationarity_tests.csv")

    if not skip_eda:
        logger.info("Generating EDA figures")
        eda.plot_manifest_summary(manifest, config.figures_dir / "dataset_composition.png")
        eda.plot_candidate_table(candidates, config.figures_dir / "class2_candidate_audit.png")
        eda.plot_timeline_grid(
            examples,
            config.figures_dir / "class2_event_windows.png",
            max_examples=config.eda.max_examples,
        )
        for event in examples:
            eda.plot_event_timeline(
                event,
                config.figures_dir / "timelines" / f"{event.stem}.png",
            )
        eda.plot_acf_pacf(
            normal_series,
            config.figures_dir / "acf_pacf_normal.png",
            lags=config.eda.acf_lags,
        )
        eda.plot_decomposition(
            normal_series,
            config.eda.decomposition_period,
            config.figures_dir / "stl_decomposition_normal.png",
        )

    forecast_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    order_candidates: list[pd.DataFrame] = []
    model_failures: list[dict[str, Any]] = []
    event_runs: dict[str, dict[str, ForecastRun]] = {}
    event_thresholds: dict[str, dict[str, float]] = {}

    for index, event in enumerate(datasets.values(), start=1):
        event_started = time.perf_counter()
        logger.info(
            "[%s/%s] Modeling %s (normal %.2f h, holdout %.2f h, transient %.0f min, steady %.0f min)",
            index,
            len(datasets),
            event.stem,
            event.normal_hours,
            event.holdout_hours,
            event.transient_minutes,
            event.steady_minutes,
        )
        try:
            y_series = event.frame["target"]
            y = y_series.to_numpy(dtype=float)
            X_raw = event.frame[["exog_lag1"]].to_numpy(dtype=float)
            start = event.calib_start_pos
            train_y = y_series.iloc[:start]
            exog_mean = float(np.nanmean(X_raw[:start]))
            exog_std = float(np.nanstd(X_raw[:start]))
            if not np.isfinite(exog_std) or exog_std <= 0:
                exog_std = 1.0
            X = (X_raw - exog_mean) / exog_std
            train_X = X[:start]

            selection = select_arima_order(
                train_y=train_y,
                max_p=config.models.max_p,
                max_q=config.models.max_q,
                max_order=config.models.max_order,
                d_max=config.models.d_max,
                trend=config.models.trend,
            )
            order_rows.append(
                {
                    "event_id": event.event_id,
                    "stem": event.stem,
                    "well_id": event.well_id,
                    "order": str(selection.order),
                    "adf_pvalue": selection.adf_pvalue,
                    "aic": selection.aic,
                    "fit_seconds": selection.fit_seconds,
                    "converged": selection.converged,
                    "n_train": len(train_y),
                    "n_candidates": len(selection.candidates),
                    "exog_mean": exog_mean,
                    "exog_std": exog_std,
                }
            )
            candidates_frame = selection.candidates.copy()
            candidates_frame.insert(0, "event_id", event.event_id)
            candidates_frame.insert(1, "stem", event.stem)
            order_candidates.append(candidates_frame)

            arima_run = rolling_forecast_arima(
                fitted_result=selection.result,
                y=y,
                start_pos=start,
                fit_seconds=selection.fit_seconds,
                aic=selection.aic,
                converged=selection.converged,
                n_train=len(train_y),
            )
            runs: dict[str, ForecastRun] = {"arima": arima_run}
            try:
                sarimax_run = rolling_forecast_sarimax(
                    train_y=train_y,
                    train_X=train_X,
                    y=y,
                    X=X,
                    start_pos=start,
                    order=selection.order,
                    trend=config.models.trend,
                )
                runs["sarimax"] = sarimax_run
            except Exception as exc:
                logger.exception("SARIMAX failed for %s", event.stem)
                model_failures.append(
                    {
                        "event_id": event.event_id,
                        "stem": event.stem,
                        "model": "sarimax",
                        "error": str(exc),
                    }
                )
            runs["naive"] = rolling_naive(y=y, start_pos=start)

            thresholds: dict[str, float] = {}
            calibration_length = event.calib_end_pos - event.calib_start_pos
            for name, run in runs.items():
                threshold = calibrate_threshold(
                    run.resid[:calibration_length],
                    config.warning.quantile,
                    config.warning.threshold_floor,
                )
                thresholds[name] = threshold
                warning_metrics = evaluate_warning_run(
                    run=run,
                    event=event,
                    threshold=threshold,
                    persistence=config.warning.persistence,
                    cooldown_minutes=config.warning.cooldown_minutes,
                    detection_grace_minutes=config.warning.detection_grace_minutes,
                    pre_event_grace_minutes=config.warning.pre_event_grace_minutes,
                )
                warning_rows.append(warning_metrics)
                forecast_rows.extend(forecast_metric_rows(run, event))
                sensitivity_rows.extend(
                    threshold_sensitivity(
                        run=run,
                        event=event,
                        quantiles=config.warning.sensitivity_quantiles,
                        persistence_values=config.warning.sensitivity_persistence,
                        cooldown_minutes=config.warning.cooldown_minutes,
                        detection_grace_minutes=config.warning.detection_grace_minutes,
                        pre_event_grace_minutes=config.warning.pre_event_grace_minutes,
                        threshold_floor=config.warning.threshold_floor,
                    )
                )

            save_event_predictions(
                event,
                runs,
                config.processed_dir / "events" / f"{event.stem}.parquet",
            )
            event_runs[event.event_id] = runs
            event_thresholds[event.event_id] = thresholds
            logger.info(
                "Completed %s in %.1f s | order=%s | thresholds=%s",
                event.stem,
                time.perf_counter() - event_started,
                selection.order,
                {name: round(value, 5) for name, value in thresholds.items()},
            )
        except Exception as exc:
            logger.exception("Modeling failed for %s", event.stem)
            model_failures.append(
                {
                    "event_id": event.event_id,
                    "stem": event.stem,
                    "model": "all",
                    "error": str(exc),
                }
            )
            continue

    if not forecast_rows:
        raise RuntimeError("No models completed successfully")

    forecast_metrics = pd.DataFrame(forecast_rows)
    warning_metrics = pd.DataFrame(warning_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    orders = pd.DataFrame(order_rows)
    failures = pd.DataFrame(model_failures + load_failures)

    forecast_summary = summarize_forecast_metrics(forecast_metrics)
    warning_summary = summarize_warning_metrics(warning_metrics)
    sensitivity_summary = summarize_sensitivity(sensitivity)

    save_dataframe(forecast_metrics, config.metrics_dir / "forecast_metrics_by_event.csv")
    save_dataframe(warning_metrics, config.metrics_dir / "warning_metrics_by_event.csv")
    save_dataframe(sensitivity, config.metrics_dir / "threshold_sensitivity_by_event.csv")
    save_dataframe(forecast_summary, config.metrics_dir / "forecast_summary.csv")
    save_dataframe(warning_summary, config.metrics_dir / "warning_summary.csv")
    save_dataframe(sensitivity_summary, config.metrics_dir / "threshold_sensitivity_summary.csv")
    save_dataframe(orders, config.tables_dir / "arima_orders.csv")
    if order_candidates:
        save_dataframe(
            pd.concat(order_candidates, ignore_index=True),
            config.tables_dir / "arima_order_candidates.csv",
        )
    if not failures.empty:
        save_dataframe(failures, config.tables_dir / "failures.csv")

    logger.info("Generating result figures")
    eda.plot_forecast_metrics(forecast_summary, config.figures_dir / "forecast_accuracy.png")
    eda.plot_warning_summary(warning_summary, config.figures_dir / "warning_performance.png")
    eda.plot_lead_times(warning_metrics, config.figures_dir / "lead_times_by_event.png")
    eda.plot_sensitivity(sensitivity_summary, config.figures_dir / "threshold_sensitivity.png")
    for event in examples:
        runs = event_runs.get(event.event_id)
        thresholds = event_thresholds.get(event.event_id)
        if runs and thresholds:
            eda.plot_forecast_example(
                event=event,
                runs=runs,
                thresholds=thresholds,
                persistence=config.warning.persistence,
                path=config.figures_dir / "forecast_examples" / f"{event.stem}.png",
            )

    runtime = time.perf_counter() - started
    run_summary = {
        "project": "OILGUARD",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_root": config.dataset_root,
        "dataset_version": "2.0.0",
        "event": {
            "class_id": config.event.class_id,
            "name": config.event.name,
            "steady_label": config.event.steady_label,
            "transient_label": config.event.transient_label,
            "target": config.event.target,
            "target_unit": config.event.target_unit,
            "exog": config.event.exog,
            "exog_unit": config.event.exog_unit,
        },
        "n_manifest_files": len(manifest),
        "n_class2_real_candidates": len(candidates),
        "n_gold_events": len(gold_candidates),
        "n_loaded_events": len(datasets),
        "n_modeled_events": len(event_runs),
        "models": ["arima", "sarimax", "naive"],
        "example_events": [event.stem for event in examples],
        "forecast_summary": forecast_summary.to_dict(orient="records"),
        "warning_summary": warning_summary.to_dict(orient="records"),
        "sensitivity_summary": sensitivity_summary.to_dict(orient="records"),
        "failures": failures.to_dict(orient="records") if not failures.empty else [],
        "runtime_seconds": runtime,
        "packages": package_versions(),
        "config": config.to_dict(),
    }
    save_json(run_summary, config.metrics_dir / "run_summary.json")
    save_json(
        {
            "forecast_summary": forecast_summary.to_dict(orient="records"),
            "warning_summary": warning_summary.to_dict(orient="records"),
            "sensitivity_summary": sensitivity_summary.to_dict(orient="records"),
        },
        config.metrics_dir / "headline_metrics.json",
    )
    logger.info("Pipeline completed in %.1f s", runtime)
    logger.info("Results: %s", config.results_dir)
    return run_summary
