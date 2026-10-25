from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from .data import EventDataset
from .forecasting import ForecastRun
from .warning import run_onsets

COLORS = {
    "real": "#1f77b4",
    "simulated": "#ff7f0e",
    "drawn": "#2ca02c",
    "arima": "#1f77b4",
    "sarimax": "#d62728",
    "naive": "#7f7f7f",
    "normal": "#dbe9f6",
    "transient": "#f6d9b0",
    "steady": "#f4b6b6",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_manifest_summary(manifest: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    classes = sorted(manifest["class_dir"].unique())
    pivot = (
        manifest.groupby(["class_dir", "source"]).size().unstack(fill_value=0).reindex(classes).fillna(0)
    )
    for source in ("real", "simulated", "drawn"):
        if source not in pivot.columns:
            pivot[source] = 0
    pivot = pivot[["real", "simulated", "drawn"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    bottom = np.zeros(len(pivot))
    for source in pivot.columns:
        values = pivot[source].to_numpy(dtype=float)
        axes[0].bar(
            pivot.index.astype(str),
            values,
            bottom=bottom,
            color=COLORS[source],
            label=source,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values
    axes[0].set_title("Instances by 3W class and source")
    axes[0].set_xlabel("Class label")
    axes[0].set_ylabel("Number of instances")
    axes[0].legend(title="Source")
    for i, total in enumerate(bottom):
        axes[0].text(i, total + max(bottom) * 0.01, f"{int(total)}", ha="center", va="bottom", fontsize=8)

    rows = manifest.groupby("class_dir")["rows"].sum().reindex(classes).fillna(0)
    axes[1].bar(rows.index.astype(str), rows.to_numpy(dtype=float) / 1e6, color="#4c72b0")
    axes[1].set_title("Labeled samples by class")
    axes[1].set_xlabel("Class label")
    axes[1].set_ylabel("Million timestamped rows")
    for i, value in enumerate(rows.to_numpy(dtype=float) / 1e6):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("3W dataset 2.0.0 composition", fontsize=13)
    save_figure(fig, path)


def plot_candidate_table(candidates: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    frame = candidates.sort_values(["well_id", "stem"]).reset_index(drop=True)
    labels = frame["stem"].tolist()
    y = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.35 * len(frame) + 2)), gridspec_kw={"width_ratios": [1.6, 1]})
    normal = frame["normal_hours"].fillna(0).to_numpy(dtype=float)
    transient = frame["transient_minutes"].fillna(0).to_numpy(dtype=float) / 60.0
    steady = frame["steady_minutes"].fillna(0).to_numpy(dtype=float) / 60.0
    axes[0].barh(y, normal, color="#4c72b0", label="normal before event")
    axes[0].barh(y, transient, left=normal, color=COLORS["transient"], label="transient 102")
    axes[0].barh(y, steady, left=normal + transient, color=COLORS["steady"], label="steady 2")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Duration (hours)")
    axes[0].set_title("Class-2 event windows")
    axes[0].legend(loc="lower right")
    for i, gold in enumerate(frame["gold"].fillna(False).astype(bool)):
        if not gold:
            axes[0].text(
                normal[i] + transient[i] + steady[i] + 0.03,
                i,
                "excluded",
                va="center",
                fontsize=7,
                color="#b22222",
            )

    quality_columns = ["target_usable", "exog_usable", "gold", "has_steady"]
    quality = frame[quality_columns].fillna(False).astype(bool).to_numpy(dtype=int)
    axes[1].imshow(quality, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_xticks(np.arange(len(quality_columns)))
    axes[1].set_xticklabels(["P-TPT usable", "T-TPT usable", "gold", "steady"], rotation=25, ha="right")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_title("Usability and cohort flags")
    for i in range(quality.shape[0]):
        for j in range(quality.shape[1]):
            axes[1].text(j, i, "Y" if quality[i, j] else "N", ha="center", va="center", fontsize=7)
    fig.suptitle("Class-2 real instances: candidate cohort audit", fontsize=13)
    save_figure(fig, path)


def _phase_shading(ax: plt.Axes, event: EventDataset) -> None:
    transient_time = event.frame.index[event.transient_pos]
    holdout_time = event.frame.index[event.holdout_start_pos]
    end_time = event.frame.index[-1]

    def to_hours(value: pd.Timestamp) -> float:
        return (value - transient_time).total_seconds() / 3600.0

    ax.axvspan(to_hours(holdout_time), 0.0, color=COLORS["normal"], alpha=0.8, label="holdout normal")
    if event.steady_pos is not None:
        steady_time = event.frame.index[event.steady_pos]
        ax.axvspan(0.0, to_hours(steady_time), color=COLORS["transient"], alpha=0.8, label="transient 102")
        ax.axvspan(to_hours(steady_time), to_hours(end_time), color=COLORS["steady"], alpha=0.8, label="steady 2")
        ax.axvline(to_hours(steady_time), color="#8b0000", linewidth=1.0, linestyle="--")
    else:
        ax.axvspan(0.0, to_hours(end_time), color=COLORS["transient"], alpha=0.8, label="transient 102")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")


def plot_event_timeline(event: EventDataset, path: str | Path) -> None:
    configure_style()
    transient_time = event.frame.index[event.transient_pos]
    hours = (event.frame.index - transient_time).total_seconds() / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(hours, event.frame["target"], color="#1f4e79", linewidth=0.9)
    axes[0].set_ylabel("P-TPT (MPa)")
    axes[0].set_title("Target pressure")
    axes[1].plot(hours, event.frame["exog"], color="#8c564b", linewidth=0.9)
    axes[1].set_ylabel("T-TPT (degC)")
    axes[1].set_xlabel("Hours relative to transient onset (102)")
    for ax in axes:
        _phase_shading(ax, event)
    handles = [
        Patch(color=COLORS["normal"], alpha=0.8, label="holdout normal"),
        Patch(color=COLORS["transient"], alpha=0.8, label="transient 102"),
        Patch(color=COLORS["steady"], alpha=0.8, label="steady 2"),
    ]
    axes[0].legend(handles=handles, loc="best")
    fig.suptitle(
        f"{event.stem} | normal {event.normal_hours:.2f} h, transient {event.transient_minutes:.0f} min, steady {event.steady_minutes:.0f} min",
        fontsize=12,
    )
    save_figure(fig, path)


def plot_timeline_grid(events: Iterable[EventDataset], path: str | Path, max_examples: int = 6) -> None:
    configure_style()
    selected = list(events)[:max_examples]
    if not selected:
        return
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), sharex=False, squeeze=False)
    for ax, event in zip(axes.ravel(), selected):
        transient_time = event.frame.index[event.transient_pos]
        hours = (event.frame.index - transient_time).total_seconds() / 3600.0
        ax.plot(hours, event.frame["target"], color="#1f4e79", linewidth=0.8)
        _phase_shading(ax, event)
        ax.set_title(
            f"{event.stem}\nnormal {event.normal_hours:.2f} h | steady {'yes' if event.steady_onset is not None else 'no'}",
            fontsize=8,
        )
        ax.set_ylabel("P-TPT (MPa)", fontsize=8)
        ax.set_xlabel("h from transient", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Representative class-2 event windows", fontsize=13)
    save_figure(fig, path)


def plot_acf_pacf(series: pd.Series, path: str | Path, lags: int = 60) -> None:
    configure_style()
    values = series.dropna()
    max_lags = max(1, min(lags, len(values) // 2 - 1))
    diff = values.diff().dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_acf(values, lags=max_lags, ax=axes[0], zero=False, title="ACF: pressure level")
        plot_pacf(values, lags=max_lags, ax=axes[1], zero=False, method="ywm", title="PACF: pressure level")
        plot_acf(diff, lags=max_lags, ax=axes[2], zero=False, title="ACF: first difference")
    axes[0].set_xlabel("Lag (10 s samples)")
    axes[1].set_xlabel("Lag (10 s samples)")
    axes[2].set_xlabel("Lag (10 s samples)")
    fig.suptitle("Autocorrelation diagnostics on a normal segment", fontsize=12)
    save_figure(fig, path)


def plot_decomposition(series: pd.Series, period: int, path: str | Path) -> None:
    configure_style()
    values = series.dropna()
    chosen_period = int(min(period, max(10, len(values) // 4)))
    if len(values) < 3 * chosen_period:
        chosen_period = max(5, len(values) // 3)
    if len(values) < 2 * chosen_period or chosen_period < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(values.index, values.to_numpy(), color="#1f4e79")
        ax.set_title("Normal pressure segment (too short for stable decomposition)")
        save_figure(fig, path)
        return
    stl = STL(values, period=chosen_period, robust=True)
    result = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(values.index, values.to_numpy(), color="#1f4e79", linewidth=0.9)
    axes[0].set_ylabel("Observed")
    axes[1].plot(values.index, result.trend, color="#d62728", linewidth=1.0)
    axes[1].set_ylabel("Trend")
    axes[2].plot(values.index, result.seasonal, color="#2ca02c", linewidth=0.8)
    axes[2].set_ylabel("Component")
    axes[3].plot(values.index, result.resid, color="#7f7f7f", linewidth=0.7)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time")
    axes[2].set_title(
        f"STL with period={chosen_period} samples ({chosen_period * 10 / 60:.0f} min); short-term component, not daily seasonality",
        fontsize=10,
    )
    fig.suptitle("Illustrative decomposition of a normal pressure segment", fontsize=12)
    save_figure(fig, path)


def _first_warning_time(run: ForecastRun, event: EventDataset, threshold: float, persistence: int) -> pd.Timestamp | None:
    eval_offset = event.holdout_start_pos - run.start_pos
    eval_residuals = run.resid[eval_offset:]
    onsets = run_onsets(np.abs(eval_residuals) > threshold, persistence)
    if len(onsets) == 0:
        return None
    return event.frame.index[event.holdout_start_pos + int(onsets[0])]


def plot_forecast_example(
    event: EventDataset,
    runs: dict[str, ForecastRun],
    thresholds: dict[str, float],
    persistence: int,
    path: str | Path,
) -> None:
    configure_style()
    frame = event.frame
    start = event.holdout_start_pos
    transient_time = frame.index[event.transient_pos]
    hours = (frame.index[start:] - transient_time).total_seconds() / 3600.0
    actual = frame["target"].to_numpy(dtype=float)[start:]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    axes[0].plot(hours, actual, color="black", linewidth=1.4, label="observed P-TPT")
    for name, run in runs.items():
        pred = run.pred[event.holdout_start_pos - run.start_pos :]
        axes[0].plot(hours, pred, color=COLORS.get(name, None), linewidth=1.0, alpha=0.9, label=f"{name.upper()} forecast")
        warning_time = _first_warning_time(run, event, thresholds[name], persistence)
        if warning_time is not None:
            x = (warning_time - transient_time).total_seconds() / 3600.0
            axes[0].axvline(x, color=COLORS.get(name, "black"), linestyle=":", linewidth=1.2)
            axes[0].annotate(
                f"{name.upper()} warning",
                xy=(x, axes[0].get_ylim()[1]),
                xytext=(2, -12),
                textcoords="offset points",
                fontsize=7,
                color=COLORS.get(name, "black"),
                rotation=90,
                va="top",
            )
    _phase_shading(axes[0], event)
    axes[0].set_ylabel("P-TPT (MPa)")
    axes[0].set_title(f"{event.stem}: one-step forecasts from holdout normal onward")
    axes[0].legend(loc="best", fontsize=8)

    for name, run in runs.items():
        residual = run.resid[event.holdout_start_pos - run.start_pos :]
        axes[1].plot(hours, residual, color=COLORS.get(name, None), linewidth=0.8, label=name.upper())
        axes[2].plot(hours, np.abs(residual), color=COLORS.get(name, None), linewidth=0.8, label=name.upper())
        axes[2].axhline(
            thresholds[name],
            color=COLORS.get(name, None),
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
        )
    for ax in axes[1:]:
        _phase_shading(ax, event)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual (MPa)")
    axes[2].set_ylabel("|Residual| (MPa)")
    axes[2].set_xlabel("Hours relative to transient onset (102)")
    axes[2].set_title("Absolute residuals and calibration thresholds (dashed)")
    fig.suptitle("Forecast-based warning example", fontsize=13)
    save_figure(fig, path)


def plot_forecast_metrics(summary: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    metrics = [
        ("mae_mean", "MAE (MPa)"),
        ("rmse_mean", "RMSE (MPa)"),
        ("mape_mean", "MAPE (%)"),
        ("smape_mean", "sMAPE (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label) in zip(axes.ravel(), metrics):
        if column not in summary.columns:
            ax.axis("off")
            continue
        subset = summary[summary["period"].isin(["validation", "test"])]
        pivot = subset.pivot(index="model", columns="period", values=column)
        pivot = pivot.reindex(columns=["validation", "test"])
        pivot.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3g", fontsize=7, padding=2)
    fig.suptitle("Forecast accuracy by model and period", fontsize=13)
    save_figure(fig, path)


def plot_warning_summary(summary: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    models = summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    axes[0, 0].bar(x - width / 2, summary["transient_early_rate"], width=width, label="before onset", color="#55a868")
    axes[0, 0].bar(x + width / 2, summary["transient_grace_rate"], width=width, label="within grace", color="#8c8c8c")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Detection rate")
    axes[0, 0].set_title("Transient phase")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x - width / 2, summary["steady_early_rate"], width=width, label="before onset", color="#c44e52")
    axes[0, 1].bar(x + width / 2, summary["steady_grace_rate"], width=width, label="within grace", color="#8c8c8c")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([m.upper() for m in models])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Detection rate")
    axes[0, 1].set_title("Steady phase")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x - width / 2, summary["median_lead_transient_minutes"], width=width, label="transient", color="#55a868")
    axes[1, 0].bar(x + width / 2, summary["median_lead_steady_minutes"], width=width, label="steady", color="#c44e52")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([m.upper() for m in models])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Median lead (minutes, log scale)")
    axes[1, 0].set_title("Lead time when a pre-onset warning occurs")
    axes[1, 0].legend(fontsize=8)
    for i, model in enumerate(models):
        for offset, column in ((-width / 2, "median_lead_transient_minutes"), (width / 2, "median_lead_steady_minutes")):
            value = summary.loc[summary["model"] == model, column].iloc[0]
            if pd.notna(value):
                axes[1, 0].text(i + offset, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    axes[1, 1].bar(x - width / 2, summary["false_alarm_rate_per_hour"], width=width, label="conservative", color="#937860")
    axes[1, 1].bar(x + width / 2, summary["false_alarm_rate_adjusted"], width=width, label="excl. pre-event", color="#4c72b0")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([m.upper() for m in models])
    axes[1, 1].set_ylabel("False alarms per normal hour")
    axes[1, 1].set_title("False alarm burden")
    axes[1, 1].legend(fontsize=8)
    for i, model in enumerate(models):
        for offset, column in ((-width / 2, "false_alarm_rate_per_hour"), (width / 2, "false_alarm_rate_adjusted")):
            value = summary.loc[summary["model"] == model, column].iloc[0]
            if pd.notna(value):
                axes[1, 1].text(i + offset, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Warning performance: onset detection and false-alarm burden", fontsize=13)
    fig.tight_layout()
    save_figure(fig, path)


def plot_lead_times(metrics: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    steady = metrics[metrics["has_steady"].fillna(False).astype(bool)]
    panels = (
        (metrics, "lead_transient_minutes", "Transient phase"),
        (steady, "lead_steady_minutes", "Steady phase"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    for ax, (data, column, label) in zip(axes[0], panels):
        subset = data.dropna(subset=[column])
        if subset.empty:
            ax.text(0.5, 0.5, "No pre-onset warning", ha="center", va="center")
            ax.axis("off")
            ax.set_title(label)
            continue
        models = subset["model"].unique().tolist()
        events = (
            subset.groupby("stem")[column]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        y = np.arange(len(events))
        height = 0.8 / max(1, len(models))
        for i, model in enumerate(models):
            values = subset[subset["model"] == model].set_index("stem")[column].reindex(events)
            ax.barh(
                y + i * height,
                values,
                height=height,
                color=COLORS.get(model),
                label=model.upper(),
            )
        ax.set_yticks(y + height * (len(models) - 1) / 2)
        ax.set_yticklabels(events, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Lead time (minutes)")
        ax.set_title(f"{label}: {len(events)} events with a pre-onset warning")
        ax.legend(fontsize=8)
    fig.suptitle("Per-event warning lead time by target phase", fontsize=13)
    fig.tight_layout()
    save_figure(fig, path)


def plot_sensitivity(summary: pd.DataFrame, path: str | Path) -> None:
    configure_style()
    models = summary["model"].unique().tolist()
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.8 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        subset = summary[summary["model"] == model]
        detection = subset.pivot(index="quantile", columns="persistence", values="transient_early_rate")
        false_alarm = subset.pivot(
            index="quantile", columns="persistence", values="mean_false_alarm_rate_adjusted"
        )
        for ax, data, title, cmap, vmax in (
            (axes[row, 0], detection, "Pre-onset transient detection rate", "YlGn", 1.0),
            (axes[row, 1], false_alarm, "Adjusted false alarms / normal hour", "YlOrRd", None),
        ):
            values = data.to_numpy(dtype=float)
            image = ax.imshow(
                values,
                aspect="auto",
                cmap=cmap,
                vmin=0,
                vmax=vmax if vmax is not None else np.nanmax(values) if np.isfinite(values).any() else 1,
            )
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_xticklabels([str(c) for c in data.columns])
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_yticklabels([f"{r:.3f}" for r in data.index])
            ax.set_xlabel("Persistence (samples)")
            ax.set_ylabel("Residual quantile")
            ax.set_title(f"{model.upper()}: {title}")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    value = values[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Threshold sensitivity: detection versus false alarms", fontsize=13)
    save_figure(fig, path)
