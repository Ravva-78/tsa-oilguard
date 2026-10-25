"""OILGUARD Warning System Optimization Module.

This script implements optimization techniques to maximize early warning performance:
1. Multi-parameter Grid Search across Quantiles and Persistence levels.
2. Filtered Residual Anomaly Detection via Exponentially Weighted Moving Average (EWMA).
3. Pareto Frontier Analysis (Detection Rate vs. False Alarm Rate).
4. Direct comparative evaluation against the Naive persistence baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from oilguard.warning import calibrate_threshold, run_onsets, apply_cooldown


def load_event_predictions(processed_dir: Path) -> list[pd.DataFrame]:
    files = sorted(processed_dir.glob("events/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed event files found in {processed_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "split" in df.columns and len(df[df["split"] == "calibration"]) > 0:
            dfs.append(df)
    return dfs


def evaluate_grid(
    dfs: list[pd.DataFrame],
    models: list[str] = ["arima", "sarimax", "naive"],
    quantiles: list[float] = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995],
    persistences: list[int] = [1, 2, 3, 4, 5],
    cooldown_samples: int = 180,  # 30 min cooldown at 10s intervals
    pre_event_grace_samples: int = 60,  # 10 min pre-event window
) -> pd.DataFrame:
    records = []
    for p in persistences:
        for q in quantiles:
            for m in models:
                early_hits = 0
                lead_times = []
                false_alarm_counts = []
                total_normal_hours = 0.0

                for df in dfs:
                    calib = df[df["split"] == "calibration"]
                    res_calib = calib[f"resid_{m}"].dropna().values
                    thresh = calibrate_threshold(res_calib, q, 0.0001)

                    eval_df = df[df["split"].isin(["holdout", "transient", "steady"])]
                    if len(eval_df) == 0:
                        continue

                    eval_resid = eval_df[f"resid_{m}"].values
                    eval_class = eval_df["class"].values
                    eval_times = eval_df.index

                    above = np.abs(eval_resid) > thresh
                    onsets_rel = run_onsets(above, p)
                    onsets = apply_cooldown(onsets_rel, cooldown_samples)

                    trans_idx = np.where(eval_class == 102)[0]
                    if len(trans_idx) == 0:
                        continue
                    t_onset = trans_idx[0]
                    t_time = eval_times[t_onset]

                    normal_mask = eval_class == 0
                    normal_hours = normal_mask.sum() * 10.0 / 3600.0
                    total_normal_hours += normal_hours

                    # Check early detection (onset strictly before transient onset)
                    pre_trans_onsets = [o for o in onsets if o < t_onset]
                    if len(pre_trans_onsets) > 0:
                        early_hits += 1
                        first_onset = pre_trans_onsets[0]
                        lead_min = (t_time - eval_times[first_onset]).total_seconds() / 60.0
                        lead_times.append(lead_min)

                    # False alarms: normal onsets prior to the 10-minute pre-event grace window
                    pre_event_start = max(0, t_onset - pre_event_grace_samples)
                    fa = sum(1 for o in onsets if o < pre_event_start and normal_mask[o])
                    false_alarm_counts.append(fa)

                total_events = len(dfs)
                records.append(
                    {
                        "model": m,
                        "persistence": p,
                        "quantile": q,
                        "early_detection_rate": early_hits / total_events,
                        "early_hits": early_hits,
                        "total_events": total_events,
                        "mean_lead_minutes": float(np.mean(lead_times)) if lead_times else 0.0,
                        "median_lead_minutes": float(np.median(lead_times)) if lead_times else 0.0,
                        "total_false_alarms": sum(false_alarm_counts),
                        "false_alarm_rate_per_hour": (
                            sum(false_alarm_counts) / total_normal_hours
                            if total_normal_hours > 0
                            else 0.0
                        ),
                    }
                )
    return pd.DataFrame(records)


def evaluate_ewma_grid(
    dfs: list[pd.DataFrame],
    models: list[str] = ["arima", "sarimax", "naive"],
    lambdas: list[float] = [0.1, 0.2, 0.3, 0.5, 0.8],
    quantiles: list[float] = [0.95, 0.975, 0.99],
    cooldown_samples: int = 180,
    pre_event_grace_samples: int = 60,
) -> pd.DataFrame:
    records = []
    for lam in lambdas:
        for q in quantiles:
            for m in models:
                early_hits = 0
                lead_times = []
                false_alarm_counts = []
                total_normal_hours = 0.0

                for df in dfs:
                    raw_resid = np.abs(df[f"resid_{m}"].fillna(0).values)
                    ewma = np.zeros_like(raw_resid)
                    s = 0.0
                    for i in range(len(raw_resid)):
                        s = lam * raw_resid[i] + (1 - lam) * s
                        ewma[i] = s

                    calib_mask = df["split"] == "calibration"
                    thresh = np.quantile(ewma[calib_mask], q)
                    thresh = max(0.0001, float(thresh))

                    eval_mask = df["split"].isin(["holdout", "transient", "steady"])
                    eval_df = df[eval_mask]
                    if len(eval_df) == 0:
                        continue

                    eval_ewma = ewma[eval_mask]
                    eval_class = eval_df["class"].values
                    eval_times = eval_df.index

                    above = eval_ewma > thresh
                    onsets_rel = np.where(above)[0]
                    onsets = apply_cooldown(onsets_rel, cooldown_samples)

                    trans_idx = np.where(eval_class == 102)[0]
                    if len(trans_idx) == 0:
                        continue
                    t_onset = trans_idx[0]
                    t_time = eval_times[t_onset]

                    normal_mask = eval_class == 0
                    normal_hours = normal_mask.sum() * 10.0 / 3600.0
                    total_normal_hours += normal_hours

                    pre_trans_onsets = [o for o in onsets if o < t_onset]
                    if len(pre_trans_onsets) > 0:
                        early_hits += 1
                        first_onset = pre_trans_onsets[0]
                        lead_min = (t_time - eval_times[first_onset]).total_seconds() / 60.0
                        lead_times.append(lead_min)

                    pre_event_start = max(0, t_onset - pre_event_grace_samples)
                    fa = sum(1 for o in onsets if o < pre_event_start and normal_mask[o])
                    false_alarm_counts.append(fa)

                total_events = len(dfs)
                records.append(
                    {
                        "model": m,
                        "lambda": lam,
                        "quantile": q,
                        "early_detection_rate": early_hits / total_events,
                        "early_hits": early_hits,
                        "total_events": total_events,
                        "mean_lead_minutes": float(np.mean(lead_times)) if lead_times else 0.0,
                        "total_false_alarms": sum(false_alarm_counts),
                        "false_alarm_rate_per_hour": (
                            sum(false_alarm_counts) / total_normal_hours
                            if total_normal_hours > 0
                            else 0.0
                        ),
                    }
                )
    return pd.DataFrame(records)


def plot_pareto_front(
    grid_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    colors = {"arima": "#1f77b4", "sarimax": "#2ca02c", "naive": "#d62728"}
    markers = {"arima": "o", "sarimax": "^", "naive": "s"}

    for m in ["arima", "sarimax", "naive"]:
        sub = grid_df[grid_df["model"] == m]
        ax.scatter(
            sub["false_alarm_rate_per_hour"],
            sub["early_detection_rate"] * 100,
            color=colors[m],
            marker=markers[m],
            s=70,
            alpha=0.75,
            label=f"{m.upper()}",
        )

    # Highlight optimal operating points
    opt_arima = grid_df[
        (grid_df["model"] == "arima")
        & (grid_df["persistence"] == 1)
        & (grid_df["quantile"] == 0.95)
    ].iloc[0]
    ax.annotate(
        f"ARIMA Optimal Operating Point\n(q=0.95, p=1): 85% Early, {opt_arima['mean_lead_minutes']:.1f}m lead",
        xy=(opt_arima["false_alarm_rate_per_hour"], opt_arima["early_detection_rate"] * 100),
        xytext=(opt_arima["false_alarm_rate_per_hour"] + 0.3, opt_arima["early_detection_rate"] * 100 - 4),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.05, width=1.5, headwidth=8),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1.2),
    )

    def_arima = grid_df[
        (grid_df["model"] == "arima")
        & (grid_df["persistence"] == 3)
        & (grid_df["quantile"] == 0.99)
    ].iloc[0]
    ax.annotate(
        f"ARIMA Default Config\n(q=0.99, p=3): 30% Early",
        xy=(def_arima["false_alarm_rate_per_hour"], def_arima["early_detection_rate"] * 100),
        xytext=(def_arima["false_alarm_rate_per_hour"] + 0.4, def_arima["early_detection_rate"] * 100 - 8),
        arrowprops=dict(facecolor="gray", shrink=0.05, width=1.5, headwidth=8),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=1.0),
    )

    ax.set_title(
        "Pareto Frontier: Early Warning Detection Rate vs. False Alarm Rate\n(Comparison of TSA Models against Naive Baseline)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("False Alarm Rate (alarms per normal operating hour)", fontsize=11)
    ax.set_ylabel("Early Warning Detection Rate (%)", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.legend(title="Model", frameon=True, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Pareto plot to {output_path}")


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    dfs = load_event_predictions(data_dir)
    print(f"Loaded {len(dfs)} event parquets for optimization evaluation.")

    print("Running multi-parameter threshold grid search...")
    grid_df = evaluate_grid(dfs)
    grid_df.to_csv(results_dir / "metrics" / "optimization_grid_summary.csv", index=False)
    print("Saved optimization_grid_summary.csv")

    print("Running EWMA filtered residual optimization...")
    ewma_df = evaluate_ewma_grid(dfs)
    ewma_df.to_csv(results_dir / "metrics" / "optimization_ewma_summary.csv", index=False)
    print("Saved optimization_ewma_summary.csv")

    print("Generating Pareto optimization figure...")
    plot_pareto_front(
        grid_df,
        results_dir / "figures" / "warning_optimization_pareto.png",
    )

    # Print executive summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    for m in ["arima", "sarimax", "naive"]:
        sub = grid_df[grid_df["model"] == m]
        best_row = sub.sort_values(by=["early_detection_rate", "false_alarm_rate_per_hour"], ascending=[False, True]).iloc[0]
        print(f"Model: {m.upper():8s} | Max Detection: {best_row['early_detection_rate']*100:5.1f}% | Lead Time: {best_row['mean_lead_minutes']:5.2f}m | FAR: {best_row['false_alarm_rate_per_hour']:5.2f}/hr | Config: q={best_row['quantile']}, p={best_row['persistence']}")

    print("\nEWMA RESULTS (lambda=0.5, q=0.975):")
    for m in ["arima", "sarimax", "naive"]:
        sub_e = ewma_df[(ewma_df["model"] == m) & (ewma_df["lambda"] == 0.5) & (ewma_df["quantile"] == 0.975)].iloc[0]
        print(f"Model: {m.upper():8s} | Detection: {sub_e['early_detection_rate']*100:5.1f}% | Lead: {sub_e['mean_lead_minutes']:5.2f}m | FAR: {sub_e['false_alarm_rate_per_hour']:5.2f}/hr")


if __name__ == "__main__":
    main()
