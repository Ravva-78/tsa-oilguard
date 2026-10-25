# Research Gap and Novelty Statement: OILGUARD-AC (Audited v2)

**Project:** OILGUARD — Forecasting-Based Early Warning of Abnormal Events in Oil Wells  
**Target Event:** Class 2 — Spurious Closure of Downhole Safety Valve (DHSV)  
**Dataset:** Petrobras 3W Benchmark v2.0.0 (Authentic Well Instances, N=19)  
**Date:** October 2026 (Audited v2)  

---

## 1. Context and Problem Statement

Downhole Safety Valve (DHSV) spurious closure (Petrobras 3W Class 2) is a critical subsea shut-in hazard. In current industrial practice, alarms trigger reactively after flow has collapsed, resulting in potential water hammer shocks, shut-in tubing pressure spikes, and unscheduled production loss. 

Recent research on the Petrobras 3W benchmark has applied machine learning algorithms—ranging from One-Class Classifiers (Fernandes et al., 2023) to Deep Learning (Vargas et al., 2019/2026)—to classify abnormal states. However, existing work suffers from three critical limitations:
1. **Target Ambiguity:** Models evaluate macro-classification accuracy across the entire file (including transient and steady-state fault periods) rather than measuring prospective advance lead time prior to transient onset ($T_{\text{transient}}$).
2. **The "Unconstrained False Alarm" Trap:** Optimization techniques tune thresholds to maximize detection without constraining normal-operation false alarms. In offshore control rooms, a warning system generating 4+ false alarms per hour induces alarm fatigue and will be silenced.
3. **Univariate Isolation:** Most forecasting detectors monitor downhole pressure (`P-TPT`) alone, neglecting the thermodynamic relationship between pressure drop and downhole temperature (`T-TPT`).

---

## 2. The Core Research Question

> **Primary Research Question:**  
> *Can multisensor forecasting residuals improve prospective early warning of spurious DHSV closure while respecting a predefined false-alarm budget on unseen real-world events?*

### Sub-Questions:
1. Does combining pressure forecast residuals with downhole temperature deviation evidence provide earlier or more reliable anomaly detection than univariate pressure forecasting?
2. How does dynamic exogenous integration in the forecasting equation (SARIMAX with lag-1 temperature) compare against post-hoc heuristic composite scoring (OILGUARD-AC)?
3. Does calibrating detection thresholds under an explicit false-alarm constraint on training data successfully bound false alarm rates on unseen holdout well events?
4. How do forecasting-based residual anomaly detectors compare against established non-forecasting density baselines (Local Outlier Factor, LOF) when both are subject to the same operational false-alarm constraints?

---

## 3. Evaluation Protocol Definitions: Resolving Methodological Ambiguity

To prevent conflating within-event time splits with cross-well generalization, this research explicitly defines and evaluates two distinct protocols:

### Protocol 1: Prospective Within-Event Split Evaluation (Well-Specific Baseline)
- **Industrial Context:** Each physical oil well has a unique operating baseline (reservoir pressure varies from 8.9 to 21.0 MPa across the 19 wells). An operator deploys a detector by establishing a well-specific baseline on historical normal telemetry before prospective monitoring begins.
- **Data Partition:** 60% Train (model fit), 20% Calibration (noise scale and threshold $\tau_k$), 20% Holdout Normal + Transient (prospective monitoring).
- **Threshold Calibration:** Threshold $\tau_k(q)$ is calibrated on event $k$'s own calibration split.

### Protocol 2: Genuine Cross-Well Leave-One-Event-Out (LOEO-CV) (Pooled Training Calibration)
- **Industrial Context:** Cold-start deployment on an uncalibrated well where NO baseline calibration data exists for that well.
- **Data Partition:** In fold $k$, event $k$ is completely held out.
- **Threshold Calibration:** Standardized calibration residuals from the other 18 training events are pooled ($H_{\text{train\_calib}} \approx 3.7\text{ hours}$). Threshold $\tau^*(q)$ is calibrated on the pooled training cohort to satisfy the operational budget and applied blindly to event $k$. Event $k$ contributes ZERO data to threshold, persistence, or hyperparameter selection.

---

## 4. Key Empirical Findings & Novelty Assessment

### Finding 1: Dynamic Exogenous Regression vs. Post-Hoc Residual Fusion
- **Observation:** In genuine cross-well LOEO-CV (Protocol 2) under a strict operational false-alarm budget ($\le 0.50\text{ FA/hr}$):
  - Dynamic Exogenous Regression (**SARIMAX**): **63.2% detection (12/19)**, **4.83 min mean lead time**, Realized FAR = **0.50 / hr** (2 FA total).
  - Balanced Multisensor Score (**OILGUARD-AC $w=0.5$**): **63.2% detection (12/19)**, **4.01 min mean lead time**, Realized FAR = **0.50 / hr** (2 FA total).
  - Heavy Pressure Score (**OILGUARD-AC $w=0.7$**): **52.6% detection (10/19)**, **4.67 min mean lead time**, Realized FAR = **0.75 / hr** (3 FA total).
  - Univariate ARIMA: **57.9% detection (11/19)**, **5.18 min mean lead time**, Realized FAR = **0.50 / hr** (2 FA total).
- **Physical Mechanism:** Downhole fluid has thermal inertia; fluid temperature changes more gradually than acoustic pressure drop during incipient valve throttling. Feeding temperature as a dynamic lag-1 exogenous regressor into SARIMAX conditions the autoregressive prediction with physical thermodynamic context *before* computing residuals. In contrast, post-hoc composite scoring $S_t = \sqrt{w z_P^2 + (1-w) z_T^2}$ can dilute the sharp pressure departure if temperature lags behind.

### Finding 2: Failure of Density-Based Detectors (LOF) under Operational Budgets
- In the literature (e.g. Fernandes et al., 2023), Local Outlier Factor (LOF) was hailed as the top-performing detector on 3W based on retrospective window classification F1-scores.
- **Our Audited Empirical Finding:** In prospective early warning, LOF fails operational constraints. Even with heavy persistence ($p=3$) and high quantile thresholds ($q=0.99$), LOF produces **1.51 to 2.01 false alarms per hour** on unseen holdout wells, violating operational budgets by 2×.
- **Why?** LOF measures k-NN density in raw $[P, T]$ space. Because baseline operating conditions vary across wells, normal holdout telemetry falls into low-density regions relative to calibration sets, generating frequent false alarms. In contrast, time-series forecasting models differenced residuals ($d \in \{0, 1\}$), providing intrinsic scale-invariance and drift adaptability.

### Finding 3: Calibration FAR vs. Realized Holdout FAR Divergence
- Because individual event calibration segments in 3W are short ($H_{\text{calib}} < 0.5\text{ h}$), setting a zero false-alarm constraint on calibration data does *not* guarantee that holdout FAR will remain below budget. Realized holdout FAR must always be reported separately from calibration FAR.

---

## 5. Scope of Scientific Claims

We do not claim that multisensor composite scoring is uniformly superior to univariate forecasting. Rather, the contribution is the **first rigorous, alarm-constrained evaluation framework on authentic Petrobras 3W Class 2 events**, establishing:
1. That dynamic regression (SARIMAX) is scientifically more defensible than heuristic residual combination;
2. That time-series forecasting models outperform non-forecasting density methods (LOF) by maintaining operational false-alarm constraints on unseen wells;
3. That classical TSA provides 4 to 5 minutes of advance warning, creating an operational window for diagnostic assessment and manual choke throttling.
