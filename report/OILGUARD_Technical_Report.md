# OILGUARD: Forecasting-Based Early Warning of Abnormal Events in Oil Wells
## Technical Report: Time Series Analysis Applied to the Petrobras 3W Offshore Benchmark Dataset

**Author:** Antigravity Research Team  
**Domain:** Offshore Petroleum Engineering & Time Series Anomaly Detection  
**Primary Event:** Class 2 — Spurious Closure of Downhole Safety Valve (DHSV)  
**Sensors:** Downhole Pressure Transducer (`P-TPT`, MPa) & Temperature Transducer (`T-TPT`, °C)  
**Dataset:** Petrobras 3W Dataset v2.0.0 (Authentic Well Instances)  
**Verified Cohort:** 19 Gold-Standard Real Well Events  

---

## 1. Executive Summary

In offshore oil and gas production, the Downhole Safety Valve (DHSV) serves as a critical safety barrier installed inside the production tubing hundreds of meters below the mudline. An inadvertent closure of this valve (designated as Class 2 in the Petrobras 3W taxonomy) violently interrupts hydrocarbon flow, trapping reservoir pressures upstream while inducing rapid pressure collapse downstream. Conventional SCADA alarms rely on fixed static thresholds that trigger only after the shut-in transient has fully manifested, offering zero actionable mitigation time.

This project investigates whether statistical **Time Series Analysis (TSA)** models—specifically Autoregressive Integrated Moving Average (**ARIMA**) and Seasonal/Exogenous ARIMAX (**SARIMAX**)—can detect anomalous forecast residuals ($e_t = y_t - \hat{y}_t$) before transient onset, providing actionable early warnings.

### Key Audited Findings:
1. **Forecasting Accuracy Dominance:** Across the 19 gold-standard offshore well events, ARIMA and SARIMAX achieved test-set Mean Absolute Errors (MAE) of **0.0163 MPa** and **0.0162 MPa** with a MAPE of **8.69%**, outperforming the persistent Naive baseline (**0.0307 MPa MAE**, **18.22% MAPE**) by **47.1%**.
2. **Event Cohort Audit Resolution:** The original candidate audit identified 22 authentic Class 2 events. Quality auditing excluded 3 instances (1 load failure with missing sensor stream, 1 unrecorded transmission gap >10 minutes, and 1 sensor flatline with zero normal variance), establishing an immutable gold cohort of **19 events**. An earlier discrepancy reporting 20 events was traced to a stale uncleaned parquet artifact from an initial scan; removing this stale file and enforcing strict candidate table joining resolves the denominator strictly to $N=19$.
3. **Audited Early Warning Rule:** To prevent false alarms during normal operations from masquerading as early detections, we enforced an explicit pre-event warning window of $[T_{\text{transient}} - 10\text{ min}, T_{\text{transient}})$. Any alarm occurring prior to this window is strictly cataloged as an ordinary operation false alarm and contributes to the False Alarm Rate (FAR / hr), while lead times are measured exclusively from the first alert within the valid pre-event window.
4. **Unbiased Generalization (LOEO-CV):** To eliminate post-hoc grid search leakage, we implemented **Leave-One-Event-Out Cross-Validation (LOEO-CV)** across all 19 events. On unseen holdout events, ARIMA achieved an unbiased **42.1% early detection rate (8/19 events)** with an average lead time of **5.10 minutes**, compared to the Naive baseline's **36.8% detection rate (7/19 events)** and **1.69 minutes** lead time. ARIMA provides **over 3× greater advance warning lead time** than the baseline without data snooping.
5. **Retrospective Optimization Ceilings:** Retrospective grid tuning ($p=1, q=0.95$) reached a sensitivity ceiling of **63.2% (12/19)** for ARIMA vs. **52.6% (10/19)** for Naive. Applying an Exponentially Weighted Moving Average (EWMA, $\lambda=0.3, q=0.975$) residual filter enabled **57.9% early detection (11/19)** with low false alarm rates (1.72/hr) vs. **42.1% (8/19)** for Naive.

---

## 2. Problem Formulation & Physical Context

A spurious DHSV closure is characterized by three distinct physical stages:
1. **Normal Operation (Class 0):** Stable multiphase fluid flow governed by steady reservoir drawdown and surface choke settings. Sensor readings exhibit minor stochastic turbulence around a stationary mean.
2. **Transient Phase (Class 102):** The hydraulic valve flapper begins inadvertent travel toward the seat. Cross-sectional flow area restricts, creating micro-acoustic turbulence, throttling pressure drop, and flow deceleration.
3. **Steady Fault Phase (Class 2):** Complete flapper seal. Downstream pressure drops to near-separator levels while upstream pressure rises to shut-in tubing head pressure.

### The Operational Challenge:
Traditional supervisory alarms are parameterized with static low-pressure trip points (e.g., $P < P_{\text{trip}}$). Because reservoir pressure varies between wells and over time, trip points must be set far below normal operating levels to prevent nuisance trips. Consequently, alarms trigger only after the valve is fully seated. An advance warning of 4 to 8 minutes provides an operational window to adjust surface chokes smoothly. Note that while attenuating pressure surges and preventing water hammer are sound engineering hypotheses, field verification requires multiphase transient dynamic simulation (e.g., OLGA) and dedicated pilot testing.

---

## 3. Dataset Auditing & Cohort Resolution

The Petrobras 3W benchmark contains 2,228 instances (1.87 GB) across synthetic and authentic well files. To ensure authentic fluid dynamics, OILGUARD enforces strict selection criteria:
- **Filtering:** Exclusively authentic offshore production well instances (`WELL-*`). Synthetic/simulated files were strictly purged.
- **Sensor Selection:** Target: Downhole Pressure Transducer (`P-TPT`, scaled to MPa); Exogenous: Downhole Temperature Transducer (`T-TPT`, scaled to °C).
- **Resampling:** Uniform 10-second aggregation with a forward-fill threshold limit of 6 raw seconds ($1\text{ min}$).
- **Data Quality Audit:** Screening for flatline (constant target/exogenous variance), missing data fraction ($<20\%$), and long gaps ($<10\text{ min}$).

### Cohort Reconciliation:
Out of 22 authentic Class 2 events:
- `WELL-00010_20171218180131` was excluded due to an absent target sensor stream.
- `WELL-00011_20140515083000` was excluded due to an unrecorded 1,608-second sensor transmission gap (>10 minutes).
- `WELL-00002_20131104004101` was excluded due to zero variance in the normal training target (`constant-normal-target`).
This yields an exact gold cohort of **19 events**. An earlier discrepancy where 20 events appeared in optimization scripts was traced to an orphaned parquet file from a preliminary scan; deleting this file and enforcing strict candidate table joins resolved the denominator to 19 across all modules.

---

## 4. Time Series Modeling Architectures

Three rolling 1-step-ahead forecasting models were deployed:

### 4.1 ARIMA(p, d, q)
$$\phi(B)(1 - B)^d (y_t - \mu) = \theta(B)\epsilon_t$$
Stationarity order $d \in \{0, 1\}$ was determined per event via the Augmented Dickey-Fuller (ADF) test ($p < 0.05 \implies d=0$). Autoregressive order $p \le 2$ and moving average order $q \le 2$ with $p+q \le 3$ were selected by minimizing the Akaike Information Criterion (AIC).

### 4.2 SARIMAX(p, d, q) with Exogenous Temperature
$$\phi(B)(1 - B)^d (y_t - \beta X_{t-1}) = \theta(B)\epsilon_t$$
Where $X_{t-1}$ represents normalized, lag-1 downhole temperature (`T-TPT`). Because temperature varies with fluid thermal inertia and mass flow rate, it provides thermodynamic feedback to the pressure forecaster.

### 4.3 Naive Persistence Baseline
$$\hat{y}_t = y_{t-1}$$
Represents standard industrial telemetry assumption of state invariance ($d=1, p=0, q=0$).

---

## 5. Experimental Results: Forecast Accuracy

One-step-ahead rolling forecasts were evaluated across all 19 events during the test evaluation split:

| Model | MAE (MPa) | RMSE (MPa) | MAPE (%) | R² Score | Bias (MPa) | Performance vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA** | **0.0163** | **0.0524** | **8.69%** | **0.9994** | +0.0075 | **47.1% Lower MAE** |
| **SARIMAX** | **0.0162** | **0.0523** | **8.63%** | **0.9994** | +0.0074 | **47.4% Lower MAE** |
| **Naive Baseline** | 0.0307 | 0.0781 | 18.22% | 0.9988 | +0.0194 | Reference Baseline |

Statistical TSA models deliver nearly half the forecasting error of the persistence baseline across all 19 events, accurately capturing underlying pressure trends.

---

## 6. Audited Early-Warning Evaluation Policy

In early iterations, alarms triggering hours prior to transient onset during ordinary normal operation were erroneously classified as "early detections", distorting lead times. Under our audited evaluation framework:
1. **Valid Early Warning Window:** Defined strictly as $[T_{\text{transient}} - 10\text{ min}, T_{\text{transient}})$.
2. **Lead Time Calculation:** Computed as $T_{\text{transient}} - T_{\text{alarm}}$ exclusively for the first alarm occurring inside the pre-event window.
3. **Ordinary Operation False Alarms:** Any alarm firing when $t < T_{\text{transient}} - 10\text{ min}$ is categorized as a false alarm during ordinary normal operation, directly contributing to False Alarm Rate (FAR / hr) and reducing warning precision.
4. **Post-Onset Detections:** Alarms firing at or after transient onset ($t \ge T_{\text{transient}}$) are classified as late/post-onset detections.

---

## 7. Optimization & Leakage Prevention

### 7.1 Unbiased Leave-One-Event-Out Cross-Validation (LOEO-CV)
To prevent optimization leakage, thresholds cannot be selected on the same data used to report performance. We conducted Leave-One-Event-Out Cross-Validation across all 19 events. In each fold $k$, the optimal quantile and persistence were chosen on the 18 training events and evaluated on the untouched $k$-th holdout event:

| Model | CV Strategy | Denominator | Unbiased Detection Rate | Early Hits | Mean Lead Time | Median Lead Time | False Alarms | FAR (/hr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA** | LOEO-CV | 19 | **42.1%** | 8 | **5.10 min** | **5.17 min** | 5 | 4.31 |
| **SARIMAX** | LOEO-CV | 19 | **36.8%** | 7 | 4.26 min | 4.50 min | 6 | 5.17 |
| **Naive Baseline** | LOEO-CV | 19 | 36.8% | 7 | 1.69 min | 1.17 min | 4 | 3.44 |

**Unbiased Generalization Takeaway:** In strict cross-validation, ARIMA achieves both higher detection rate and **over 3× the advance warning lead time** of the baseline (5.10 minutes vs. 1.69 minutes), validating that TSA residuals carry genuine incipient fault signals.

### 7.2 Retrospective Grid Search Sensitivity Ceiling
Evaluating persistence $p \in [1..5]$ and calibration quantiles $q \in [0.90..0.995]$ reveals the sensitivity envelope:
- At $p=1, q=0.95$:
  - **ARIMA:** 63.2% Detection Rate (12/19), 5.35 min Lead Time, FAR = 3.44/hr.
  - **SARIMAX:** 57.9% Detection Rate (11/19), 5.30 min Lead Time, FAR = 4.31/hr.
  - **Naive:** 52.6% Detection Rate (10/19), 4.88 min Lead Time, FAR = 2.58/hr.

### 7.3 EWMA Filtered Residual Anomaly Detection
Applying an Exponentially Weighted Moving Average to residuals ($S_t = \lambda |e_t| + (1 - \lambda) S_{t-1}$) with $\lambda = 0.3, q = 0.975$:
- **ARIMA & SARIMAX:** **57.9% Detection Rate (11/19)**, **3.53 min Lead Time**, FAR = 1.72/hr (2 false alarms).
- **Naive Baseline:** 42.1% Detection Rate (8/19), 3.65 min Lead Time, FAR = 1.72/hr (2 false alarms).
- EWMA filtering provides a **+15.8 percentage point advantage** over Naive while suppressing high-frequency sensor noise.

---

## 8. Pareto Frontier Verification

The Pareto trade-off curve between False Alarm Rate (FAR / hr) and Early Warning Detection Rate was recomputed and programmatically verified:
- **Non-Dominated Pareto Frontier:** Calculated strictly such that a configuration is Pareto-optimal iff no other parameter achieves both higher detection rate and lower false alarm rate.
- **Annotation Correction:** The callout arrow for ARIMA is verified to target the ARIMA marker at coordinates `(3.44, 63.2%)`, eliminating earlier graphic alignment ambiguities.

The verified figure is saved at `results/figures/warning_optimization_pareto.png`.

---

## 9. Conclusion & Operational Scope Qualifications

1. **Statistical Superiority:** Time series forecasting models (ARIMA and SARIMAX) model physical pressure-temperature dynamics with 47% lower error than persistence baselines.
2. **Early Detection Feasibility:** In honest cross-validation, ARIMA delivers 5.10 minutes of advance lead time prior to transient onset, outperforming the Naive baseline by +3.41 minutes.
3. **Engineering Scope Qualifications:** While 5 minutes of advance warning provides critical lead time for supervisory control, assertions regarding water hammer elimination or specific pressure surge reduction percentages represent engineering hypotheses that require verification through transient multiphase dynamic simulators (e.g. OLGA) and controlled offshore pilot trials.
