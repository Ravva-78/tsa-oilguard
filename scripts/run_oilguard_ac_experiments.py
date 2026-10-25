"""Experiment execution script for OILGUARD-AC research paper.

Runs:
1. Multisensor ablation (weights w_P in [0.5, 0.7, 0.85, 1.0])
2. Budgeted threshold calibration verification (B in [0.5, 1.0, 2.0, 3.0])
3. Benchmarks comparison (OILGUARD-AC vs ARIMA vs SARIMAX vs Naive vs LOF)
4. Unbiased Leave-One-Event-Out Cross-Validation (LOEO-CV) with 95% Bootstrap CIs
5. Generates publication-ready figures.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilguard.oilguard_ac import (
    EventScoreData,
    compute_bootstrap_ci,
    evaluate_cohort,
    load_event_scores,
    run_loeo_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("oilguard_ac_runner")


def load_gold_cohort(events_dir: Path, candidates_csv: Path) -> list[EventScoreData]:
    cand = pd.read_csv(candidates_csv)
    gold = cand[cand["gold"] == True]
    events: list[EventScoreData] = []
    for row in gold.itertuples():
        parquet_file = events_dir / f"{row.stem}.parquet"
        if parquet_file.exists():
            ev = load_event_scores(parquet_file, event_id=row.event_id)
            events.append(ev)
        else:
            logger.warning("Missing parquet for gold event: %s", row.stem)
    return events


def main() -> None:
    events_dir = PROJECT_ROOT / "data" / "processed" / "events"
    candidates_csv = PROJECT_ROOT / "results" / "tables" / "class2_candidates.csv"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    events = load_gold_cohort(events_dir, candidates_csv)
    n_events = len(events)
    logger.info("Loaded %d gold events (Cohort denominator = %d)", n_events, n_events)
    assert n_events == 19, f"Expected 19 gold events, got {n_events}"

    # -------------------------------------------------------------
    # Experiment 1: Multisensor Ablation Study (w_P weights)
    # -------------------------------------------------------------
    logger.info("--- Running Experiment 1: Multisensor Ablation ---")
    weights = [0.5, 0.7, 0.85, 1.0]
    budgets = [0.5, 1.0, 2.0, 3.0]
    persistences = [1, 2, 3]

    ablation_rows = []
    for w in weights:
        for b in budgets:
            for p in persistences:
                res = evaluate_cohort(
                    events=events,
                    model="arima",
                    weight_pressure=w,
                    budget_far_per_hour=b,
                    persistence=p,
                )
                ablation_rows.append(res)
    df_ablation = pd.DataFrame(ablation_rows)
    ablation_path = metrics_dir / "oilguard_ac_multisensor_ablation.csv"
    df_ablation.to_csv(ablation_path, index=False)
    logger.info("Saved %s (%d rows)", ablation_path, len(df_ablation))

    # -------------------------------------------------------------
    # Experiment 2: False-Alarm Budget Calibration Verification
    # -------------------------------------------------------------
    logger.info("--- Running Experiment 2: Budget Calibration Verification ---")
    budget_rows = []
    for b in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        res_ac = evaluate_cohort(
            events=events,
            model="arima",
            weight_pressure=0.7,
            budget_far_per_hour=b,
            persistence=1,
        )
        res_p_only = evaluate_cohort(
            events=events,
            model="arima",
            weight_pressure=1.0,
            budget_far_per_hour=b,
            persistence=1,
        )
        budget_rows.append({
            "target_budget_far": b,
            "ac_detection_rate": res_ac["detection_rate"],
            "ac_realized_far": res_ac["realized_far_per_hour"],
            "ac_mean_lead": res_ac["mean_lead_minutes"],
            "p_only_detection_rate": res_p_only["detection_rate"],
            "p_only_realized_far": res_p_only["realized_far_per_hour"],
            "p_only_mean_lead": res_p_only["mean_lead_minutes"],
        })
    df_budget = pd.DataFrame(budget_rows)
    budget_path = metrics_dir / "oilguard_ac_budget_calibration.csv"
    df_budget.to_csv(budget_path, index=False)
    logger.info("Saved %s", budget_path)

    # -------------------------------------------------------------
    # Experiment 3 & 4: Benchmarks & Unbiased LOEO-CV Evaluation
    # -------------------------------------------------------------
    logger.info("--- Running Experiment 3 & 4: Unbiased LOEO-CV Benchmarks ---")
    models_config = [
        ("OILGUARD-AC (Multisensor, w=0.7)", "arima", 0.7, 1.0),
        ("Univariate ARIMA (p-only, w=1.0)", "arima", 1.0, 1.0),
        ("Univariate SARIMAX (p-only, w=1.0)", "sarimax", 1.0, 1.0),
        ("Naive Persistence Baseline", "naive", 1.0, 1.0),
        ("Local Outlier Factor (LOF)", "lof", 0.0, 1.0),
    ]

    loeo_rows = []
    for label, m, w, b in models_config:
        logger.info("Evaluating LOEO-CV for %s...", label)
        cv_res = run_loeo_cv(
            events=events,
            model=m,
            weight_pressure=w,
            budget_far_per_hour=b,
            candidate_persistence=[1, 2, 3],
        )
        ci = compute_bootstrap_ci(cv_res["fold_results"], n_bootstraps=1000, random_seed=42)

        loeo_rows.append({
            "method": label,
            "model_type": m,
            "weight_pressure": w,
            "budget_far": b,
            "total_events": cv_res["total_events"],
            "early_hits": cv_res["early_hits"],
            "unbiased_detection_rate": cv_res["unbiased_detection_rate"],
            "detection_rate_ci_lower": ci["detection_rate_ci_95"][0],
            "detection_rate_ci_upper": ci["detection_rate_ci_95"][1],
            "unbiased_mean_lead_minutes": cv_res["unbiased_mean_lead_minutes"],
            "mean_lead_ci_lower": ci["mean_lead_ci_95"][0],
            "mean_lead_ci_upper": ci["mean_lead_ci_95"][1],
            "unbiased_median_lead_minutes": cv_res["unbiased_median_lead_minutes"],
            "total_false_alarms": cv_res["total_false_alarms"],
            "unbiased_realized_far": cv_res["unbiased_realized_far"],
            "realized_far_ci_lower": ci["realized_far_ci_95"][0],
            "realized_far_ci_upper": ci["realized_far_ci_95"][1],
        })

    df_loeo = pd.DataFrame(loeo_rows)
    loeo_path = metrics_dir / "oilguard_ac_loeo_cv_summary.csv"
    df_loeo.to_csv(loeo_path, index=False)
    logger.info("Saved %s", loeo_path)

    # -------------------------------------------------------------
    # Generate Publication Figures
    # -------------------------------------------------------------
    logger.info("--- Generating Visual Figures ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Figure 1: Trade-off between Budget, Realized FAR, and Detection Rate
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax1.plot(df_budget["target_budget_far"], df_budget["ac_detection_rate"] * 100, "o-", color="#1f77b4", linewidth=2.5, label="OILGUARD-AC Detection Rate (%)")
    ax1.plot(df_budget["target_budget_far"], df_budget["p_only_detection_rate"] * 100, "s--", color="#ff7f0e", linewidth=2, label="Univariate ARIMA Detection Rate (%)")
    ax1.set_xlabel("Operational False-Alarm Budget $B_{\\mathrm{FAR}}$ (Alarms / Normal Hour)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Early Warning Detection Rate (%) [N=19]", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(df_budget["target_budget_far"], df_budget["ac_realized_far"], "^-.", color="#2ca02c", linewidth=2, label="OILGUARD-AC Realized FAR (/hr)")
    ax2.plot(df_budget["target_budget_far"], df_budget["target_budget_far"], ":", color="gray", alpha=0.7, label="Ideal Calibration Line (Realized = Budget)")
    ax2.set_ylabel("Realized False Alarm Rate (Alarms / Normal Hour)", fontsize=11, fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)
    ax1.set_title("OILGUARD-AC: Detection Rate and False Alarm Realization vs. Target Budget", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig1_path = figures_dir / "oilguard_ac_tradeoff.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info("Saved %s", fig1_path)

    # Figure 2: LOEO-CV Benchmark Comparison Bar Chart with Error Bars
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    methods = df_loeo["method"].tolist()
    det_rates = (df_loeo["unbiased_detection_rate"] * 100).to_numpy()
    err_low = det_rates - (df_loeo["detection_rate_ci_lower"] * 100).to_numpy()
    err_high = (df_loeo["detection_rate_ci_upper"] * 100).to_numpy() - det_rates
    y_err = [np.maximum(err_low, 0.0), np.maximum(err_high, 0.0)]

    colors = ["#2b5c8f", "#418ab3", "#6baed6", "#fc9272", "#9ecae1"]
    bars = ax.bar(methods, det_rates, yerr=y_err, capsize=5, color=colors, edgecolor="black", alpha=0.9, width=0.55)
    ax.set_ylabel("Unbiased Detection Rate (%) with 95% Bootstrap CI", fontsize=11, fontweight="bold")
    ax.set_title("Leave-One-Event-Out (LOEO-CV) Comparison: OILGUARD-AC vs. Baselines (N=19)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right", fontsize=9.5)

    for bar, r in zip(bars, df_loeo.itertuples()):
        lead = r.unbiased_mean_lead_minutes
        far = r.unbiased_realized_far
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 3.0,
            f"{bar.get_height():.1f}%\n(Lead: {lead:.1f}m | FAR: {far:.2f}/h)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig2_path = figures_dir / "oilguard_ac_benchmarks.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info("Saved %s", fig2_path)

    logger.info("All experiments and figures completed successfully!")


if __name__ == "__main__":
    main()
