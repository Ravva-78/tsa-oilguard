# Guarding Offshore Wells with Time Series Analysis: Audited Findings on Petrobras 3W and 5-Minute Early Warnings

*By the OILGUARD Research Team*  
*Estimated reading time: 6 minutes (1,150 words)*

---

Deep beneath the ocean floor, offshore petroleum wells operate under hostile, high-pressure environments. Among the most critical safety barriers installed in every modern subsea completion is the **Downhole Safety Valve (DHSV)**. Positioned hundreds of meters below the mudline inside the production tubing, this fail-safe valve exists to isolate the reservoir in the event of an emergency.

But what happens when the safety valve closes by mistake?

In offshore petroleum operations, this undesirable event is classified as a **spurious DHSV closure** (designated as Class 2 in the benchmark Petrobras 3W taxonomy). An inadvertent loss of hydraulic control fluid or mechanical actuator failure causes the valve flapper to travel toward its seat. The flowing hydrocarbon column abruptly hits a restriction, trapping extreme reservoir pressures upstream while inducing rapid pressure collapse downstream. 

Traditional offshore supervisory control systems rely on static threshold alarms—essentially alerting operators when downhole pressure drops below a fixed lower boundary. The fatal limitation? **By the time pressure crosses that trip threshold, the valve is already sealed.** The transient has already peaked, leaving operators with zero reaction time.

In our project, **OILGUARD**, we investigated an essential question: **Can statistical time-series forecasting predict abnormal oil well events before transient onset?**

---

## The Petrobras 3W Benchmark: Auditing the Cohort

To test our hypothesis on authentic industrial physics, we turned to the premier open benchmark for machine learning in oil well monitoring: the **Petrobras 3W dataset**.

Spanning over 2,200 multivariate time-series records and 1.87 GB of raw data from offshore Brazilian production basins, the benchmark contains sensor streams collected at 1 Hz. Crucially, the dataset contains both synthetic simulations and real-world offshore well recordings. 

Because numerical simulations often smooth away the messy turbulent noise inherent in authentic multiphase flow, we enforced strict filtering: **we evaluated exclusively on authentic offshore well recordings (`WELL-*`) experiencing Class 2 (Spurious Closure of DHSV).**

Our audit identified 22 real-world Class 2 candidates. We then applied rigorous data quality criteria:
- **Missing Sensor Stream:** `WELL-00010_20171218180131` was excluded due to an unrecorded pressure transducer stream.
- **Excessive Transmission Gap:** `WELL-00011_20140515083000` was excluded due to an unrecorded 1,608-second gap (>10 minutes).
- **Sensor Flatline:** `WELL-00002_20131104004101` was excluded due to zero variance in the normal training target (`constant-normal-target`).

This established an immutable, verified cohort of **19 authentic offshore shut-in events**.

---

## The Architecture: ARIMA, SARIMAX, and the Naive Baseline

To establish early warning capability, we implemented rolling one-step-ahead forecasts across three distinct models:

1. **ARIMA(p, d, q):** Dynamically selecting autoregressive and moving-average orders per well via Akaike Information Criterion (AIC) minimization, with differencing $d$ guided by Augmented Dickey-Fuller (ADF) stationarity testing.
2. **SARIMAX(p, d, q):** Augmenting ARIMA with standardized, lag-1 downhole temperature (`T-TPT`) as an exogenous regressor, capturing thermodynamic fluid inertia.
3. **The Naive Persistence Baseline:** Predicting that the next observation equals the current one ($\hat{y}_t = y_{t-1}$). In industrial telemetry, this represents the standard benchmark.

As sensor data arrives at 10-second intervals, the model forecasts expected pressure. The forecast residual—the difference between actual observation and model prediction—serves as our anomaly detector:

$$e_t = y_t - \hat{y}_t$$

During normal operating conditions, forecast errors remain small and stochastic. When an incipient valve closure begins, flow restriction induces subtle pressure deviations, causing residuals to depart from the noise floor well before the macroscopic transient unfolds.

---

## Forecasting Accuracy: Statistical TSA Dominance

When we evaluated pure forecasting accuracy on the untouched test split across all 19 events, the statistical models achieved total dominance:

- **ARIMA Test MAE:** 0.0163 MPa (RMSE: 0.0524 MPa, MAPE: 8.69%, R²: 0.9994)
- **SARIMAX Test MAE:** 0.0162 MPa (RMSE: 0.0523 MPa, MAPE: 8.63%, R²: 0.9994)
- **Naive Baseline MAE:** 0.0307 MPa (RMSE: 0.0781 MPa, MAPE: 18.22%, R²: 0.9988)

**ARIMA cut forecasting error by 47.1% compared to the Naive baseline.** The statistical models captured true autocorrelation and thermodynamic relationships, rejecting high-frequency turbulent fluctuations that trip naive estimators.

---

## Auditing the Early Warning Rule: Eliminating Evaluation Leakage

Early alarm evaluation algorithms often suffer from a hidden flaw: if a model fires an alarm during ordinary operation 45 minutes before an event, a simplistic rule might label it an "early warning" with a 45-minute lead time!

To ensure rigorous industrial validity, we established an audited evaluation policy:
1. **Explicit Pre-Event Window:** An alarm counts as a valid early warning if and only if it occurs inside $[T_{\text{transient}} - 10\text{ min}, T_{\text{transient}})$.
2. **True False Alarms:** Any alert triggering prior to this window is strictly cataloged as an ordinary operation false alarm, directly penalizing the False Alarm Rate (FAR / hr) and warning precision.
3. **Lead Time Measurement:** Lead time is measured strictly from the first alert within the pre-event window.

---

## Honest Validation: Leave-One-Event-Out Cross-Validation (LOEO-CV)

A common trap in anomaly detection is reporting the best result of a post-hoc grid search as the model's performance. That constitutes data leakage.

To establish genuine generalization performance, we implemented **Leave-One-Event-Out Cross-Validation (LOEO-CV)** across all 19 events:
- In each fold, optimal detection parameters were chosen strictly on 18 events and tested on the 19th unseen holdout event.
- **ARIMA:** Achieved an unbiased **42.1% early detection rate (8/19 events)** with a **5.10-minute mean lead time** (FAR = 4.31/hr).
- **SARIMAX:** Achieved an unbiased **36.8% early detection rate (7/19 events)** with a **4.26-minute mean lead time** (FAR = 5.17/hr).
- **Naive Baseline:** Achieved **36.8% early detection (7/19 events)** with only a **1.69-minute mean lead time** (FAR = 3.44/hr).

**The Crucial Insight:** In an honest, leak-free test, ARIMA provides **over 3× the advance warning lead time** of the baseline (5.10 min vs. 1.69 min). Those extra 3.4 minutes represent the difference between an automated choke adjustment and an emergency shutdown.

---

## Filtering Residuals: EWMA Smoothing

Sensor telemetry is noisy. In Statistical Process Control (SPC), engineers smooth residuals using an Exponentially Weighted Moving Average (EWMA):

$$S_t = \lambda |e_t| + (1 - \lambda) S_{t-1}$$

With $\lambda = 0.3$ and quantile threshold $q = 0.975$:
- **ARIMA & SARIMAX:** **57.9% Early Warning Rate (11/19 events)** with **3.53 minutes** lead time and only 1.72 false alarms per normal hour.
- **Naive Baseline:** 42.1% Early Warning Rate (8/19 events) with 3.65 minutes lead time.

EWMA filtering delivers a **+15.8 percentage point detection advantage** over the persistence baseline while keeping false alarm rates low.

---

## The Operational Horizon: What 5 Minutes Enables

In offshore production engineering, 5 minutes of advance warning is highly significant:
1. **Automated Choke Maneuvering:** Supervisory control systems can initiate a controlled surface choke ramp-down, decelerating fluid velocity before the flapper fully seals.
2. **Surge Mitigation Hypothesis:** Smoothly slowing the fluid column is hypothesized to attenuate pressure surges and mitigate water-hammer shockwaves that compromise subsea manifolds. While physical verification requires transient multiphase flow modeling (e.g., OLGA simulators), the time series detector reliably creates the operational window needed to execute it.
3. **Control Room Decision Support:** Clear differentiation between ordinary sensor drift and incipient valve travel allows operators to verify hydraulic pressures before an uncommanded emergency platform trip occurs.

By combining rigorous statistical forecasting with robust residual filtering, **OILGUARD** demonstrates that classical time series analysis provides actionable, scientifically grounded early warning intelligence for offshore energy infrastructure.

---

*The full reproducible codebase, unit tests, executed Jupyter walkthrough, verified CSV metrics, and Word report are available in the OILGUARD project repository.*
