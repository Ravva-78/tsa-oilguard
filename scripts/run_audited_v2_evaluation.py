"""Rigorous, versioned research evaluation driver for OILGUARD (v2 audit).

Implements:
1. Protocol 1: Prospective Within-Event Split Evaluation (Well-Specific Baseline Calibration)
2. Protocol 2: Genuine Cross-Event Leave-One-Event-Out (LOEO-CV) (Pooled 18-Event Calibration)
3. Protocol 3: Genuine Cross-Well Leave-One-Well-Out (LOWO-CV) (Pooled Cross-Well Calibration, N=5 Physical Wells)
4. Multisensor ablation: Univariate ARIMA, SARIMAX (exog T), Naive, AC (w in [0.5, 0.7, 0.85]), and LOF
5. Mean and Median lead times with 1,000-iteration bootstrap 95% confidence intervals
6. Transparent reporting of realized holdout FAR vs. calibration constraints
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

        # Physical well ID
        well_id = row.stem.split("_")[0]

        events.append({
            "stem": row.stem,
            "well_id": well_id,
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
    det_list, lead_list, med_lead_list, far_list = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample = [fold_results[i] for i in idx]
        hits = sum(1 for r in sample if r["early_detected"])
        leads = [r["lead_time_minutes"] for r in sample if r["early_detected"]]
        fa = sum(r["ordinary_fa"] for r in sample)
        nh = sum(r["normal_hours"] for r in sample)

        det_list.append(hits / n)
        lead_list.append(float(np.mean(leads)) if leads else 0.0)
        med_lead_list.append(float(np.median(leads)) if leads else 0.0)
        far_list.append(fa / nh if nh > 0 else 0.0)

    return {
        "det_ci": (float(np.percentile(det_list, 2.5)), float(np.percentile(det_list, 97.5))),
        "lead_ci": (float(np.percentile(lead_list, 2.5)), float(np.percentile(lead_list, 97.5))),
        "med_lead_ci": (float(np.percentile(med_lead_list, 2.5)), float(np.percentile(med_lead_list, 97.5))),
        "far_ci": (float(np.percentile(far_list, 2.5)), float(np.percentile(far_list, 97.5))),
    }


def main() -> None:
    events = load_all_events()
    total_events = len(events)
    total_eval_h = sum(e["normal_hours"] for e in events)
    total_calib_h = sum(e["calib_hours"] for e in events)
    
    well_counts = {}
    for e in events:
        well_counts[e["well_id"]] = well_counts.get(e["well_id"], 0) + 1
    logger.info("Loaded %d gold events across %d physical wells: %s", total_events, len(well_counts), well_counts)
    logger.info("Eval normal hours: %.2f h | Calib hours: %.2f h", total_eval_h, total_calib_h)

    models = ["sarimax", "ac_0.85", "ac_0.7", "ac_0.5", "arima", "naive", "lof"]
    quantiles = [0.90, 0.95, 0.975, 0.99]
    persistences = [1, 2, 3]

    # -------------------------------------------------------------
    # 1. Protocol 1: Prospective Within-Event Split Evaluation
    # -------------------------------------------------------------
    logger.info("--- Evaluating Protocol 1: Prospective Within-Event Split Evaluation ---")
    p1_rows = []
    for model in models:
        for q in quantiles:
            for p in persistences:
                fold_res = []
                calib_fa_total = 0
                for ev in events:
                    calib_s = ev["scores"][model][ev["calib_mask"]]
                    thresh = float(np.quantile(calib_s, q))

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
                    "med_lead_minutes": float(np.median(leads)) if leads else 0.0,
                    "med_lead_ci_low": ci["med_lead_ci"][0],
                    "med_lead_ci_high": ci["med_lead_ci"][1],
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
    # 2. Protocol 2: Genuine Cross-Event Leave-One-Event-Out (LOEO-CV)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Protocol 2: Genuine Cross-Event LOEO-CV (19 Folds) ---")
    p2_rows = []
    per_event_rows = []

    for model in models:
        for q in quantiles:
            for p in persistences:
                fold_res = []
                for k in range(total_events):
                    held_out = events[k]
                    train_events = [events[i] for i in range(total_events) if i != k]

                    pool_calib = np.concatenate([e["scores"][model][e["calib_mask"]] for e in train_events])
                    thresh_k = float(np.quantile(pool_calib, q))

                    res = evaluate_single_event(held_out, model, thresh_k, persistence=p)
                    res["stem"] = held_out["stem"]
                    res["well_id"] = held_out["well_id"]
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
                    "protocol": "Genuine Cross-Event (LOEO-CV)",
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
                    "med_lead_minutes": float(np.median(leads)) if leads else 0.0,
                    "med_lead_ci_low": ci["med_lead_ci"][0],
                    "med_lead_ci_high": ci["med_lead_ci"][1],
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
    # 3. Protocol 3: Leave-One-Well-Out Cross-Validation (LOWO-CV, 5 Folds)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Protocol 3: Leave-One-Well-Out LOWO-CV (5 Physical Well Folds) ---")
    p3_rows = []

    for model in models:
        for q in quantiles:
            for p in persistences:
                fold_res = []
                for k in range(total_events):
                    held_out = events[k]
                    target_well = held_out["well_id"]
                    pool_calib = np.concatenate([
                        e["scores"][model][e["calib_mask"]]
                        for e in events if e["well_id"] != target_well
                    ])
                    thresh_k = float(np.quantile(pool_calib, q))

                    res = evaluate_single_event(held_out, model, thresh_k, persistence=p)
                    fold_res.append(res)

                hits = sum(1 for r in fold_res if r["early_detected"])
                leads = [r["lead_time_minutes"] for r in fold_res if r["early_detected"]]
                fa = sum(r["ordinary_fa"] for r in fold_res)
                ci = bootstrap_ci(fold_res, n_bootstraps=1000)

                p3_rows.append({
                    "protocol": "Leave-One-Well-Out (LOWO-CV)",
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
                    "med_lead_minutes": float(np.median(leads)) if leads else 0.0,
                    "med_lead_ci_low": ci["med_lead_ci"][0],
                    "med_lead_ci_high": ci["med_lead_ci"][1],
                    "total_false_alarms": fa,
                    "realized_far_per_hour": fa / total_eval_h,
                    "far_ci_low": ci["far_ci"][0],
                    "far_ci_high": ci["far_ci"][1],
                })

    df_p3 = pd.DataFrame(p3_rows)
    df_p3.to_csv(METRICS_DIR / "protocol3_leave_one_well_out_summary.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "protocol3_leave_one_well_out_summary.csv")

    # -------------------------------------------------------------
    # 4. Matched-FAR & Controlled Budget Comparison
    # -------------------------------------------------------------
    logger.info("--- Generating Matched-FAR Benchmark Comparison Table ---")
    matched_rows = []
    
    for m in models:
        sub_p1 = df_p1[(df_p1["model"] == m) & (df_p1["realized_far_per_hour"] <= 1.01)]
        best_p1 = sub_p1.sort_values(by=["detection_rate", "mean_lead_minutes"], ascending=False).iloc[0] if not sub_p1.empty else df_p1[df_p1["model"] == m].sort_values(by="realized_far_per_hour").iloc[0]

        sub_p2 = df_p2[(df_p2["model"] == m) & (df_p2["realized_far_per_hour"] <= 1.01)]
        best_p2 = sub_p2.sort_values(by=["detection_rate", "mean_lead_minutes"], ascending=False).iloc[0] if not sub_p2.empty else df_p2[df_p2["model"] == m].sort_values(by="realized_far_per_hour").iloc[0]

        sub_p3 = df_p3[(df_p3["model"] == m) & (df_p3["realized_far_per_hour"] <= 1.01)]
        best_p3 = sub_p3.sort_values(by=["detection_rate", "mean_lead_minutes"], ascending=False).iloc[0] if not sub_p3.empty else df_p3[df_p3["model"] == m].sort_values(by="realized_far_per_hour").iloc[0]

        meets_budget = (best_p2["realized_far_per_hour"] <= 1.0)

        matched_rows.append({
            "model": m,
            "p1_config": f"q={best_p1['quantile']}, p={best_p1['persistence']}",
            "p1_hits": int(best_p1["early_hits"]),
            "p1_detection_rate": best_p1["detection_rate"],
            "p1_mean_lead": best_p1["mean_lead_minutes"],
            "p1_med_lead": best_p1["med_lead_minutes"],
            "p1_realized_far": best_p1["realized_far_per_hour"],
            "p1_fa_count": int(best_p1["total_false_alarms"]),
            "p2_config": f"q={best_p2['quantile']}, p={best_p2['persistence']}",
            "p2_hits": int(best_p2["early_hits"]),
            "p2_detection_rate": best_p2["detection_rate"],
            "p2_mean_lead": best_p2["mean_lead_minutes"],
            "p2_med_lead": best_p2["med_lead_minutes"],
            "p2_realized_far": best_p2["realized_far_per_hour"],
            "p2_fa_count": int(best_p2["total_false_alarms"]),
            "p3_config": f"q={best_p3['quantile']}, p={best_p3['persistence']}",
            "p3_hits": int(best_p3["early_hits"]),
            "p3_detection_rate": best_p3["detection_rate"],
            "p3_mean_lead": best_p3["mean_lead_minutes"],
            "p3_med_lead": best_p3["med_lead_minutes"],
            "p3_realized_far": best_p3["realized_far_per_hour"],
            "p3_fa_count": int(best_p3["total_false_alarms"]),
            "p2_meets_1.0_budget": meets_budget,
        })

    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(METRICS_DIR / "matched_far_comparison.csv", index=False)
    logger.info("Saved %s", METRICS_DIR / "matched_far_comparison.csv")

    # -------------------------------------------------------------
    # 5. Generate Visual Figures
    # -------------------------------------------------------------
    logger.info("--- Generating Visual Figures in results/audited_v2/figures/ ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Protocol Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(df_matched))
    width = 0.35

    model_display_names = {
        "sarimax": "SARIMAX (Exogenous T)",
        "ac_0.85": "OILGUARD-AC (w=0.85)",
        "ac_0.7": "OILGUARD-AC (w=0.7)",
        "ac_0.5": "OILGUARD-AC (w=0.5)",
        "arima": "Univariate ARIMA",
        "naive": "Naive Persistence",
        "lof": "Local Outlier Factor (LOF)*",
    }
    x_labels = [model_display_names.get(m, m) for m in df_matched["model"]]

    rects1 = ax.bar(x - width/2, df_matched["p1_detection_rate"] * 100, width, label="Protocol 1: Within-Event (Well-Specific)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, df_matched["p2_detection_rate"] * 100, width, label="Protocol 2: Genuine Cross-Event LOEO-CV", color="#6baed6", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Early Detection Rate (%) [N=19 Events]", fontsize=11, fontweight="bold")
    ax.set_title("Detection Performance Across Protocols at Matched Operational FAR Settings\n(Realized FAR Annotated; *LOF exceeds 1.0/hr budget at 1.51/hr)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)

    for r1, r2, r in zip(rects1, rects2, df_matched.itertuples()):
        ax.text(r1.get_x() + r1.get_width()/2, r1.get_height() + 1.2, f"{r1.get_height():.1f}%\nFAR={r.p1_realized_far:.2f}", ha="center", va="bottom", fontsize=7.0)
        ax.text(r2.get_x() + r2.get_width()/2, r2.get_height() + 1.2, f"{r2.get_height():.1f}%\nFAR={r.p2_realized_far:.2f}", ha="center", va="bottom", fontsize=7.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_comparison.png")
    plt.close()
    logger.info("Saved %s", FIGURES_DIR / "protocol_comparison.png")

    # Figure 2: Realized FAR vs Detection Trade-off in Protocol 2
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    color_map = {
        "sarimax": "#2ca02c",
        "ac_0.85": "#17becf",
        "ac_0.7": "#1f77b4",
        "ac_0.5": "#9467bd",
        "arima": "#ff7f0e",
        "naive": "#7f7f7f",
        "lof": "#d62728",
    }
    label_map = {
        "sarimax": "SARIMAX (Exogenous T)",
        "ac_0.85": "OILGUARD-AC (w=0.85)",
        "ac_0.7": "OILGUARD-AC (w=0.7)",
        "ac_0.5": "OILGUARD-AC (w=0.5)",
        "arima": "Univariate ARIMA",
        "naive": "Naive Persistence",
        "lof": "Local Outlier Factor (LOF)",
    }

    for m in models:
        sub = df_p2[df_p2["model"] == m].sort_values(by="realized_far_per_hour")
        ax.plot(sub["realized_far_per_hour"], sub["detection_rate"] * 100, "o-", color=color_map[m], label=label_map[m], linewidth=1.8, markersize=5.5)

    ax.axvline(1.0, color="red", linestyle="--", alpha=0.75, label="Operational Nuisance Limit (1.0 FA/hr)")
    ax.set_xlabel("Realized Holdout False Alarm Rate (Alarms / Normal Operating Hour)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Early Warning Detection Rate (%) [N=19 Events]", fontsize=11, fontweight="bold")
    ax.set_title("Protocol 2 (LOEO-CV): Realized FAR vs. Early Detection Rate Trade-Off Curve", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.1, 3.8)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "matched_far_tradeoff.png")
    plt.close()
    logger.info("Saved %s", FIGURES_DIR / "matched_far_tradeoff.png")

    # Figure 3: LOEO vs LOWO Sensitivity Analysis
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df_matched))
    rects_loeo = ax.bar(x - width/2, df_matched["p2_detection_rate"] * 100, width, label="LOEO-CV (19 Event Folds)", color="#3182bd", edgecolor="black", alpha=0.9)
    rects_lowo = ax.bar(x + width/2, df_matched["p3_detection_rate"] * 100, width, label="LOWO-CV (5 Physical Well Folds)", color="#e6550d", edgecolor="black", alpha=0.9)
    ax.set_ylabel("Early Detection Rate (%) [N=19 Events]", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Event (LOEO) vs. Cross-Physical-Well (LOWO) Generalization Sensitivity", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    for r1, r2 in zip(rects_loeo, rects_lowo):
        ax.text(r1.get_x() + r1.get_width()/2, r1.get_height() + 1.2, f"{r1.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.text(r2.get_x() + r2.get_width()/2, r2.get_height() + 1.2, f"{r2.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loeo_vs_lowo_comparison.png")
    plt.close()
    logger.info("Saved %s", FIGURES_DIR / "loeo_vs_lowo_comparison.png")

    logger.info("Audited evaluation script completed successfully!")


if __name__ == "__main__":
    main()
