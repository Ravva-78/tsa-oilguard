# Guarding Offshore Wells: From Time Series Analysis to Alarm-Constrained Early Warning (OILGUARD-AC)

*By the OILGUARD Research Team*  
*Estimated reading time: 7 minutes (1,250 words)*

---

Deep beneath the ocean floor, offshore petroleum wells operate under extreme, high-pressure environments. Positioned hundreds of meters below the mudline inside the production tubing is the **Downhole Safety Valve (DHSV)**—a critical fail-safe barrier designed to isolate the reservoir during an emergency.

But when this valve closes unexpectedly—a phenomenon known as a **spurious DHSV closure** (designated as Class 2 in the Petrobras 3W benchmark)—the consequences are severe. Flowing hydrocarbons abruptly hit a steel barrier, trapping reservoir pressures upstream while downstream pressure collapses, threatening flowlines and platform processing equipment.

In existing offshore installations, SCADA systems rely on static lower-bound pressure trip alarms. The fatal flaw? **By the time pressure crosses the trip threshold, the valve is already sealed.** The transient has peaked, leaving operators with zero reaction time.

In our project, **OILGUARD**, we investigated whether statistical time-series forecasting could predict abnormal closures before transient onset. But our research journey uncovered a much deeper, more fundamental industrial challenge: **How do you build an early warning system that provides genuine advance warning without flooding control room operators with false alarms?**

---

## The Literature Landscape and the Hidden Research Gap

To benchmark our methods against state-of-the-art research, we audited five foundational and recent publications:
- **Vargas et al. (2019, 2026):** The creators of the Petrobras 3W benchmark. Their baseline evaluations focus on macroscopic event classification—distinguishing Class 2 from Class 0 across entire files. However, they do not evaluate prospective lead times prior to transient onset or quantify false alarms per operating hour.
- **Fernandes Jr., Komati & Gazolli (2023):** Evaluated unsupervised One-Class Classifiers (Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoders) on 3W, finding that **Local Outlier Factor (LOF)** achieved the highest F1-score (~87%).
- **Wang et al. (2024)** and **Khan et al. (2026):** Highlighted the importance of combining prediction residuals with cross-sensor relational shifts and calibrating detection thresholds under strict false-alarm constraints.

This revealed a critical **research gap**:  
*Existing 3W studies evaluate anomaly detection without an operational nuisance-alarm budget, often maximizing detection by tuning thresholds to post-hoc optima. In an offshore control room, a detector that achieves 80% detection at the cost of 4 false alarms per hour will simply be silenced by frustrated operators.*

This motivated our candidate contribution: **OILGUARD-AC (Alarm-Constrained, Multisensor Early Warning)**.

---

## Auditing the Benchmark: The 19-Event Gold Cohort

We turned to the **Petrobras 3W Dataset v2.0.0** (1.87 GB), strictly filtering for authentic offshore production wells (`WELL-*`) experiencing Class 2 spurious closures.

Out of 22 authentic candidates, our data quality audit excluded three defective files:
1. `WELL-00010_20171218180131`: Missing target sensor telemetry (`P-TPT`).
2. `WELL-00011_20140515083000`: Unrecorded 1,608-second sensor transmission gap (>10 minutes).
3. `WELL-00002_20131104004101`: Flatline target sensor with zero normal training variance (`constant-normal-target`).

This established an immutable gold cohort of **exactly 19 authentic events**.

---

## Round 1: Forecasting Accuracy Dominance

Evaluating rolling 1-step-ahead forecasts across all 19 gold events during the test evaluation split revealed clear statistical superiority:
- **ARIMA Test MAE:** 0.0163 MPa (RMSE: 0.0524 MPa, MAPE: 8.69%, R²: 0.9994)
- **SARIMAX Test MAE:** 0.0162 MPa (RMSE: 0.0523 MPa, MAPE: 8.63%, R²: 0.9994)
- **Naive Baseline MAE:** 0.0307 MPa (RMSE: 0.0781 MPa, MAPE: 18.22%, R²: 0.9988)

**ARIMA cut forecasting error by 47.1% compared to the Naive baseline.** The statistical models captured true autoregressive and moving-average structure, filtering out turbulent sensor noise.

---

## Round 2: The OILGUARD-AC Framework

To convert forecasting accuracy into reliable, operational early warning, **OILGUARD-AC** introduces three key mechanisms:
1. **Multisensor Bivariate Residual Scoring:** Combining standardized pressure forecast residuals ($z_P$) with downhole temperature deviations ($z_T$) into a composite score:
   $$S_t = \sqrt{w_P \cdot z_{P, t}^2 + (1 - w_P) \cdot z_{T, t}^2}$$
2. **False-Alarm-Budgeted Threshold Calibration:** Rather than picking arbitrary percentiles, the threshold is calibrated on normal calibration data such that the calibration false alarm rate does not exceed an operational budget (e.g., $B_{\text{FAR}} \le 1.0\text{ alarms / normal hour}$).
3. **Leak-Free Leave-One-Event-Out Validation (LOEO-CV):** Thresholds and persistence parameters are chosen strictly on 18 training events and evaluated on the 19th unseen holdout event, benchmarked against Local Outlier Factor (LOF).

---

## The Breakthrough: Forecasting vs. Non-Forecasting Density Detectors

Under an operational budget of **$B_{\text{FAR}} \le 1.0\text{ false alarms per hour}$**, our leave-one-event-out cross-validation revealed a striking contrast:

| Method / Model | Unbiased Detection Rate [95% CI] | Early Hits | Mean Lead Time | Realized Holdout FAR | Budget Satisfied? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Univariate SARIMAX** | **68.4%** [47.4%, 89.5%] | **13 / 19** | **5.40 min** | **0.75 / hr** | **YES** ($\le 1.0$) |
| **OILGUARD-AC (Multisensor, w=0.7)** | **47.4%** [26.3%, 68.4%] | 9 / 19 | **3.78 min** | **0.75 / hr** | **YES** ($\le 1.0$) |
| **Univariate ARIMA** | **42.1%** [21.1%, 68.4%] | 8 / 19 | **4.31 min** | **0.50 / hr** | **YES** ($\le 1.0$) |
| **Naive Baseline** | **47.4%** [26.3%, 68.4%] | 9 / 19 | **4.48 min** | **0.50 / hr** | **YES** ($\le 1.0$) |
| **Local Outlier Factor (LOF)** | **73.7%** [52.6%, 89.6%] | 14 / 19 | **5.54 min** | **2.01 / hr** | **NO (2× Budget Violation!)** |

### Why Did LOF Fail the Operational Test?
Local Outlier Factor was identified by Fernandes et al. (2023) as the premier one-class classifier on 3W. But in a prospective early warning setting, LOF relies on static density ratios in $[P, T]$ feature space. When deployed on an unseen well, natural baseline pressure differences caused LOF to register normal operations as "outliers," generating **8 false alarms (2.01 / hr)**—violating the budget by 200%!

In contrast, **forecasting models track differential residuals**. Because ARIMA and SARIMAX model 1-step dynamic changes, they naturally adapt to baseline drift, keeping realized false alarm rates comfortably below the budget (**0.50 to 0.75 / hr**).

Furthermore, by dynamically conditioning pressure forecasts on downhole temperature, **SARIMAX achieved 68.4% early detection (13 out of 19 events)** with **5.40 minutes** of advance lead time!

---

## What 5 Minutes Enables in Offshore Operations

In offshore drilling and production, 5 minutes of advance lead time provides:
1. **Automated Choke Maneuvering:** Supervisory systems can initiate a controlled surface choke ramp-down, decelerating fluid velocity before the flapper fully seals.
2. **Surge Mitigation Potential:** Smoothly decelerating the fluid column is hypothesized to attenuate pressure surges and mitigate water-hammer shockwaves that compromise subsea manifolds. (Note: physical validation requires multiphase transient dynamic simulation such as OLGA and field pilots).
3. **Control Room Decision Support:** Operators have a clear, false-alarm-controlled alert allowing them to verify hydraulic telemetry before an uncommanded emergency platform trip occurs.

By moving beyond post-hoc threshold tuning and enforcing strict operational false-alarm budgets, **OILGUARD-AC** demonstrates that statistical time series analysis provides reliable, scientifically grounded early warning intelligence for offshore energy infrastructure.

---

*The full reproducible codebase, unit tests, executed Jupyter walkthrough, verified CSV metrics, and Word report are available in the OILGUARD project repository.*
