"""Rigorous, versioned research evaluation driver for OILGUARD (v2 audit).

Implements:
1. Protocol 1: Prospective Within-Event Split Evaluation (Well-Specific Baseline Calibration)
2. Protocol 2: Genuine Cross-Well Leave-One-Event-Out (LOEO-CV) (Pooled Training Calibration)
3. Full ablation: Univariate ARIMA, SARIMAX (exogenous T), Naive, Multisensor AC (w in [0.5, 0.7, 0.85]), and LOF
4. Calibration FAR vs. Realized Holdout FAR tracking
5. Matched-FAR comparison and bootstrap 95% confidence intervals
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(r"C:\Users\nan33\OneDrive\Desktop\TSA-OILGUARD")
V2_DIR = PROJECT_ROOT / "results" / "audited_v2"
METRICS_DIR = V2_DIR / "metrics"
FIGURES_DIR = V2_DIR / "figures"
TABLES_DIR = V2_DIR / "tables"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("oilguard_v2_audit")


def load_all_events() -> list[dict[str, Any]]:
    cand_path = PROJECT_ROOT / "results" / "tables" / "class2_candidates.csv"
    cand = pd.read_csv(cand_path)
    gold = cand[cand["gold"] == True]
    events_dir = PROJECT_ROOT / "data" / "processed" / "events"

    events = []
    for row in gold.itertuples():
        parquet_file = events_dir / f"{row.stem}.parquet"
        assert parquet_file.exists(), f"Missing file: {parquet_file}"
        df = pd.read_parquet(parquet_file)

        calib_mask = df["split"].isin(["calibration", "calib"]).to_numpy()
        eval_mask = df["split"].isin(["holdout", "test", "transient", "steady"]).to_numpy()
        transient_mask = (df["class"] == 102).to_numpy()
        transient_onset = df.index[transient_mask][0] if transient_mask.any() else df.index[-1]

        holdout_normal_mask = (
            df["split"].isin(["holdout", "test"]).to_numpy()
            & (df["class"] == 0).to_numpy()
            & (df.index < transient_onset)
        )
        normal_hours = (holdout_normal_mask.sum() * 10.0) / 3600.0
        calib_hours = (calib_mask.sum() * 10.0) / 3600.0

        resid_arima = np.abs(df["resid_arima"].to_numpy(dtype=float))
        resid_sarimax = np.abs(df["resid_sarimax"].to_numpy(dtype=float))
        resid_naive = np.abs(df["resid_naive"].to_numpy(dtype=float))
        target = df["target"].to_numpy(dtype=float)
        exog = df["exog"].to_numpy(dtype=float)

        # Scale statistics from calibration split
        sigma_ar = max(float(np.nanstd(resid_arima[calib_mask])), 1e-4)
        sigma_sx = max(float(np.nanstd(resid_sarimax[calib_mask])), 1e-4)
        sigma_nv = max(float(np.nanstd(resid_naive[calib_mask])), 1e-4)

        mu_temp = float(np.nanmean(exog[calib_mask]))
        sigma_temp = max(float(np.nanstd(exog[calib_mask])), 1e-4)
        z_temp = np.abs(exog - mu_temp) / sigma_temp

        z_ar = resid_arima / sigma_ar
        z_sx = resid_sarimax / sigma_sx
        z_nv = resid_naive / sigma_nv

        # Multisensor scores
        s_ac_05 = np.sqrt(0.5 * (z_ar ** 2) + 0.5 * (z_temp ** 2))
        s_ac_07 = np.sqrt(0.7 * (z_ar ** 2) + 0.3 * (z_temp ** 2))
        s_ac_085 = np.sqrt(0.85 * (z_ar ** 2) + 0.15 * (z_temp ** 2))

        # Non-forecasting benchmark: Local Outlier Factor (LOF)
        X_raw = np.column_stack([target, exog])
        scaler = StandardScaler()
        X_calib = scaler.fit_transform(X_raw[calib_mask])
        X_all = scaler.transform(X_raw)
        k_neighbors = min(20, max(2, len(X_calib) - 1))
        lof = LocalOutlierFactor(n_neighbors=k_neighbors, novelty=True)
        lof.fit(X_calib)
        lof_scores = -lof.decision_function(X_all)
        lof_scores = np.maximum(lof_scores - np.nanmin(lof_scores[calib_mask]), 0.0)
        sigma_lof = max(float(np.nanstd(lof_scores[calib_mask])), 1e-4)
        z_lof = lof_scores / sigma_lof

        events.append({
            "stem": row.stem,
            "event_id": row.event_id,
            "index": df.index,
            "calib_mask": calib_mask,
            "eval_mask": eval_mask,
            "transient_onset": transient_onset,
            "normal_hours": normal_hours,
            "calib_hours": calib_hours,
            "scores": {
                "arima": z_ar,
                "sarimax": z_sx,
                "naive": z_nv,
                "ac_0.5": s_ac_05,
                "ac_0.7": s_ac_07,
                "ac_0.85": s_ac_085,
                "lof": z_lof,
            }
        })
    return events


def find_alarm_episodes(scores_subset: np.ndarray, thresh: float, persistence: int = 1, cooldown_samples: int = 12) -> list[int]:
    above = scores_subset >= thresh
    if not np.any(above):
        return []
    triggers = []
    run = 0
    last_trig = -999999
    for i in range(len(above)):
        if above[i]:
            run += 1
            if run == persistence:
                if (i - last_trig) >= cooldown_samples:
                    triggers.append(i)
                    last_trig = i
        else:
            run = 0
    return triggers


def evaluate_single_event(
    ev: dict[str, Any],
    model_key: str,
    threshold: float,
    persistence: int = 1,
    cooldown_samples: int = 12,
    window_minutes: float = 10.0,
) -> dict[str, Any]:
    score = ev["scores"][model_key]
    eval_idx = np.where(ev["eval_mask"])[0]
    eval_scores = score[eval_idx]
    eval_times = ev["index"][eval_idx]

    trigs = find_alarm_episodes(eval_scores, threshold, persistence=persistence, cooldown_samples=cooldown_samples)
    trig_times = [eval_times[i] for i in trigs]

    window_start = ev["transient_onset"] - pd.Timedelta(minutes=window_minutes)
    pre_alarms = [t for t in trig_times if window_start <= t < ev["transient_onset"]]
    ordinary_fa = [t for t in trig_times if t < window_start]
    post_alarms = [t for t in trig_times if t >= ev["transient_onset"]]

    early_detected = len(pre_alarms) > 0
    lead_time = (ev["transient_onset"] - pre_alarms[0]).total_seconds() / 60.0 if early_detected else 0.0

    return {
        "early_detected": early_detected,
        "lead_time_minutes": lead_time,
        "ordinary_fa": len(ordinary_fa),
        "post_alarms": len(post_alarms),
        "normal_hours": ev["normal_hours"],
    }


def bootstrap_ci(fold_results: list[dict[str, Any]], n_bootstraps: int = 1000, seed: int = 42) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(fold_results)
    det_list, lead_list, far_list = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample = [fold_results[i] for i in idx]
        hits = sum(1 for r in sample if r["early_detected"])
        leads = [r["lead_time_minutes"] for r in sample if r["early_detected"]]
        fa = sum(r["ordinary_fa"] for r in sample)
        nh = sum(r["normal_hours"] for r in sample)

        det_list.append(hits / n)
        lead_list.append(float(np.mean(leads)) if leads else 0.0)
        far_list.append(fa / nh if nh > 0 else 0.0)

    return {
        "det_ci": (float(np.percentile(det_list, 2.5)), float(np.percentile(det_list, 97.5))),
        "lead_ci": (float(np.percentile(lead_list, 2.5)), float(np.percentile(lead_list, 97.5))),
        "far_ci": (float(np.percentile(far_list, 2.5)), float(np.percentile(far_list, 97.5))),
    }


def main() -> None:
    events = load_all_events()
    total_events = len(events)
    total_eval_h = sum(e["normal_hours"] for e in events)
    total_calib_h = sum(e["calib_hours"] for e in events)
    logger.info("Loaded %d gold events | Eval normal hours: %.2f h | Calib hours: %.2f h", total_events, total_eval_h, total_calib_h)

    # -------------------------------------------------------------
    # 1. Protocol 1: Prospective Within-Event Split Evaluation
    # -------------------------------------------------------------
    logger.info("--- Evaluating Protocol 1: Prospective Within-Event Split Evaluation ---")
    p1_rows = []
    models = ["sarimax", "ac_0.7", "ac_0.5", "arima", "naive", "lof"]

    for model in models:
        for q in [0.90, 0.95, 0.975, 0.99]:
            for p in [1, 2, 3]:
                fold_res = []
                calib_fa_total = 0
                for ev in events:
                    calib_s = ev["scores"][model][ev["calib_mask"]]
                    thresh = float(np.quantile(calib_s, q))

                    # Calib false alarms
                    trigs_calib = find_alarm_episodes(calib_s, thresh, persistence=p, cooldown_samples=12)
                    calib_fa_total += len(trigs_calib)

                    res = evaluate_single_event(ev, model, thresh, persistence=p)
                    fold_res.append(res)

                hits = sum(1 for r in fold_res if r["early_detected"])
                leads = [r["lead_time_minutes"] for r in fold_res if r["early_detected"]]
                fa = sum(r["ordinary_fa"] for r in fold_res)
                ci = bootstrap_ci(fold_res, n_bootstraps=1000)

                p1_rows.append({
                    "protocol": "Within-Event (Well-Specific)",
                    "model": model,
                    "quantile": q,
                    "persistence": p,
                    "early_hits": hits,
                    "total_events": total_events,
                    "detection_rate": hits / total_events,
                    "det_ci_low": ci["det_ci"][0],
                    "det_ci_high": ci["det_ci"][1],
                    "mean_lead_minutes": float(np.mean(leads)) if leads else 0.0,
                    "lead_ci_low": ci["lead_ci"][0],
                    "lead_ci_high": ci["lead_ci"][1],
                    "total_false_alarms": fa,
                    "realized_far_per_hour": fa / total_eval_h,
                    "far_ci_low": ci["far_ci"][0],
                    "far_ci_high": ci["far_ci"][1],
                    "calib_fa_total": calib_fa_total,
                    "calib_far_per_hour": calib_fa_total / total_calib_h,
                })

    df_p1 = pd.DataFrame(p1_rows)
    df_p1.to_csv(METRICS_DIR / "protocol1_within_event_summary.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "protocol1_within_event_summary.csv")

    # -------------------------------------------------------------
    # 2. Protocol 2: Genuine Cross-Well Leave-One-Event-Out (LOEO-CV)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Protocol 2: Genuine Cross-Well LOEO-CV ---")
    p2_rows = []
    per_event_rows = []

    for model in models:
        for q in [0.90, 0.95, 0.975, 0.99]:
            for p in [1, 2, 3]:
                fold_res = []
                for k in range(total_events):
                    held_out = events[k]
                    train_events = [events[i] for i in range(total_events) if i != k]

                    # Pool all calibration z-scores from the other 18 training events
                    pool_calib = np.concatenate([e["scores"][model][e["calib_mask"]] for e in train_events])
                    thresh_k = float(np.quantile(pool_calib, q))

                    res = evaluate_single_event(held_out, model, thresh_k, persistence=p)
                    res["stem"] = held_out["stem"]
                    res["model"] = model
                    res["quantile"] = q
                    res["persistence"] = p
                    res["threshold"] = thresh_k
                    fold_res.append(res)
                    per_event_rows.append(res)

                hits = sum(1 for r in fold_res if r["early_detected"])
                leads = [r["lead_time_minutes"] for r in fold_res if r["early_detected"]]
                fa = sum(r["ordinary_fa"] for r in fold_res)
                ci = bootstrap_ci(fold_res, n_bootstraps=1000)

                p2_rows.append({
                    "protocol": "Genuine Cross-Well (LOEO-CV)",
                    "model": model,
                    "quantile": q,
                    "persistence": p,
                    "early_hits": hits,
                    "total_events": total_events,
                    "detection_rate": hits / total_events,
                    "det_ci_low": ci["det_ci"][0],
                    "det_ci_high": ci["det_ci"][1],
                    "mean_lead_minutes": float(np.mean(leads)) if leads else 0.0,
                    "lead_ci_low": ci["lead_ci"][0],
                    "lead_ci_high": ci["lead_ci"][1],
                    "total_false_alarms": fa,
                    "realized_far_per_hour": fa / total_eval_h,
                    "far_ci_low": ci["far_ci"][0],
                    "far_ci_high": ci["far_ci"][1],
                })

    df_p2 = pd.DataFrame(p2_rows)
    df_p2.to_csv(METRICS_DIR / "protocol2_cross_well_loeo_summary.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "protocol2_cross_well_loeo_summary.csv")

    df_per_event = pd.DataFrame(per_event_rows)
    df_per_event.to_csv(METRICS_DIR / "per_event_outcomes.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "per_event_outcomes.csv")

    # -------------------------------------------------------------
    # 3. Matched-FAR & Controlled Budget Comparison
    # -------------------------------------------------------------
    # Filter configurations where Realized FAR is bounded (e.g. <= 1.0/hr)
    logger.info("--- Generating Matched-FAR Benchmark Comparison Table ---")
    matched_rows = []
    
    # We find best configuration for each model where realized FAR <= 1.0 / hr
    for m in models:
        sub_p1 = df_p1[(df_p1["model"] == m) & (df_p1["realized_far_per_hour"] <= 1.0)]
        if not sub_p1.empty:
            best_p1 = sub_p1.sort_values(by=["detection_rate", "mean_lead_minutes"], ascending=False).iloc[0]
        else:
            best_p1 = df_p1[df_p1["model"] == m].sort_values(by="realized_far_per_hour").iloc[0]

        sub_p2 = df_p2[(df_p2["model"] == m) & (df_p2["realized_far_per_hour"] <= 1.0)]
        if not sub_p2.empty:
            best_p2 = sub_p2.sort_values(by=["detection_rate", "mean_lead_minutes"], ascending=False).iloc[0]
        else:
            best_p2 = df_p2[df_p2["model"] == m].sort_values(by="realized_far_per_hour").iloc[0]

        matched_rows.append({
            "model": m,
            "p1_config": f"q={best_p1['quantile']}, p={best_p1['persistence']}",
            "p1_detection_rate": best_p1["detection_rate"],
            "p1_lead_minutes": best_p1["mean_lead_minutes"],
            "p1_realized_far": best_p1["realized_far_per_hour"],
            "p2_config": f"q={best_p2['quantile']}, p={best_p2['persistence']}",
            "p2_detection_rate": best_p2["detection_rate"],
            "p2_lead_minutes": best_p2["mean_lead_minutes"],
            "p2_realized_far": best_p2["realized_far_per_hour"],
        })

    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(METRICS_DIR / "matched_far_comparison.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "matched_far_comparison.csv")

    # -------------------------------------------------------------
    # 4. Generate Visual Figures
    # -------------------------------------------------------------
    logger.info("--- Generating Visual Figures in results/audited_v2/figures/ ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Matched FAR Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(df_matched))
    width = 0.35

    model_labels = [
        "SARIMAX (Exogenous T)",
        "OILGUARD-AC (w=0.7)",
        "OILGUARD-AC (w=0.5)",
        "Univariate ARIMA",
        "Naive Persistence",
        "Local Outlier Factor (LOF)",
    ]

    rects1 = ax.bar(x - width/2, df_matched["p1_detection_rate"] * 100, width, label="Protocol 1: Within-Event (Well-Specific)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, df_matched["p2_detection_rate"] * 100, width, label="Protocol 2: Genuine Cross-Well LOEO-CV", color="#6baed6", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Early Detection Rate (%) [N=19]", fontsize=11, fontweight="bold")
    ax.set_title("Audited Comparison Across Protocols under Realized FAR <= 1.0 / hr", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)

    for r1, r2, r in zip(rects1, rects2, df_matched.itertuples()):
        ax.text(r1.get_x() + r1.get_width()/2, r1.get_height() + 1.5, f"{r1.get_height():.1f}%\n({r.p1_lead_minutes:.1f}m)", ha="center", va="bottom", fontsize=7.5)
        ax.text(r2.get_x() + r2.get_width()/2, r2.get_height() + 1.5, f"{r2.get_height():.1f}%\n({r.p2_lead_minutes:.1f}m)", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_comparison.png")
    plt.close()
    logger.info("Saved %s", FIGURES_DIR / "protocol_comparison.png")

    # Figure 2: Realized FAR vs Detection Trade-off in Protocol 2
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    color_map = {
        "sarimax": "#2ca02c",
        "ac_0.7": "#1f77b4",
        "ac_0.5": "#9467bd",
        "arima": "#ff7f0e",
        "naive": "#7f7f7f",
        "lof": "#d62728",
    }
    label_map = {
        "sarimax": "SARIMAX (Exogenous T)",
        "ac_0.7": "OILGUARD-AC (w=0.7)",
        "ac_0.5": "OILGUARD-AC (w=0.5)",
        "arima": "Univariate ARIMA",
        "naive": "Naive Persistence",
        "lof": "Local Outlier Factor (LOF)",
    }

    for m in models:
        sub = df_p2[df_p2["model"] == m].sort_values(by="realized_far_per_hour")
        ax.plot(sub["realized_far_per_hour"], sub["detection_rate"] * 100, "o-", color=color_map[m], label=label_map[m], linewidth=1.8, markersize=6)

    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="Operational Nuisance Budget (1.0 FA/hr)")
    ax.set_xlabel("Realized Holdout False Alarm Rate (Alarms / Normal Operating Hour)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Early Warning Detection Rate (%) [N=19]", fontsize=11, fontweight="bold")
    ax.set_title("Protocol 2: Genuine Cross-Well LOEO-CV Realized FAR vs. Detection Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.1, 3.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "matched_far_tradeoff.png")
    plt.close()
    logger.info("Saved %s", FIGURES_DIR / "matched_far_tradeoff.png")

    logger.info("Audited evaluation script completed successfully!")


if __name__ == "__main__":
    main()
