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

## 3. Evaluation Protocol Definitions & Cohort Verification

To prevent conflating within-event time splits with cross-well generalization, this research audits the evaluation unit and explicitly defines three protocols across the **19 authentic events** representing **5 distinct physical offshore wells**:
- `WELL-00011`: 12 events (63.2% of cohort)
- `WELL-00003`: 3 events (15.8% of cohort)
- `WELL-00012`: 2 events (10.5% of cohort)
- `WELL-00009`: 1 event (5.3% of cohort)
- `WELL-00013`: 1 event (5.3% of cohort)

### Protocol 1: Prospective Within-Event Split Evaluation (Well-Specific Baseline)
- **Operational Scenario:** Evaluates performance when historical baseline data from the specific well is available to condition the model and calibrate thresholds.
- **Data Partition:** 60% Train (model fit), 20% Calibration (noise scale and threshold $\tau_k$), 20% Holdout Normal + Transient (prospective monitoring).
- **Threshold Calibration:** Threshold $\tau_k(q)$ is calibrated on event $k$'s own calibration split ($H_{\text{calib}} \approx 0.2\text{ h}$).

### Protocol 2: Genuine Cross-Event Leave-One-Event-Out (LOEO-CV) (Pooled Training Calibration)
- **Operational Scenario:** Evaluates generalization to an unseen event when pooled historical data from other events is used for calibration.
- **Data Partition:** In fold $k$, event $k$ is completely held out.
- **Threshold Calibration:** Standardized calibration residuals from the other 18 training events are pooled ($H_{\text{train\_calib}} \approx 3.7\text{ hours}$). Threshold $\tau^*(q)$ is calibrated on the pooled training cohort and applied to event $k$. Event $k$ contributes zero data to fitting, normalization, threshold, or persistence selection.

### Protocol 3: Leave-One-Well-Out Cross-Validation (LOWO-CV) (Cross-Physical-Well Generalization)
- **Operational Scenario:** Cold-start deployment on an entirely unmonitored physical well structure, ensuring that repeated events from the dominant well (`WELL-00011`) cannot inform calibration for that well.
- **Data Partition:** In fold $w$, all events belonging to physical well $w$ are completely held out.
- **Threshold Calibration:** Thresholds are calibrated strictly on pooled data from events belonging to the other 4 physical wells.

---

## 4. Key Empirical Findings & Reconciled Results

All detectors are benchmarked at matched operational realized false-alarm rates ($\le 1.0\text{ FA/hr}$, with primary comparisons at $\le 0.75\text{ FA/hr}$ across 3.98 normal holdout hours):

| Method / Model | P1 Early Detection | P1 Mean / Med Lead | P1 Realized FAR | P2 Early Detection [95% CI] | P2 Mean / Med Lead | P2 Realized FAR | P3 (LOWO) Detection | Meets Budget? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMAX (exog T)** | 36.8% (7/19) | 4.0 / 4.3 min | 0.25 / hr | **63.2% (12/19)** [42.1%, 84.2%] | **4.8 / 4.8 min** | **0.50 / hr** (2 FA) | **73.7% (14/19)** | **YES** ($\le 1.0$) |
| **OILGUARD-AC ($w=0.85$)** | 47.4% (9/19) | 3.3 / 3.5 min | 0.50 / hr | **63.2% (12/19)** [42.1%, 84.2%] | **5.3 / 5.4 min** | **0.50 / hr** (2 FA) | **73.7% (14/19)** | **YES** ($\le 1.0$) |
| **OILGUARD-AC ($w=0.5$)** | **68.4% (13/19)** | **4.2 / 4.3 min** | **0.75 / hr** | **63.2% (12/19)** [42.1%, 84.2%] | **4.0 / 3.2 min** | **0.50 / hr** (2 FA) | **63.2% (12/19)** | **YES** ($\le 1.0$) |
| **Univariate ARIMA** | 31.6% (6/19) | 2.5 / 1.5 min | 0.25 / hr | **57.9% (11/19)** [36.8%, 78.9%] | **5.2 / 5.2 min** | **0.50 / hr** (2 FA) | **68.4% (13/19)** | **YES** ($\le 1.0$) |
| **OILGUARD-AC ($w=0.7$)** | 57.9% (11/19) | 2.8 / 2.5 min | 0.75 / hr | **52.6% (10/19)** [31.6%, 73.7%] | **4.7 / 4.4 min** | **0.75 / hr** (3 FA) | **63.2% (12/19)** | **YES** ($\le 1.0$) |
| **Naive Baseline** | 47.4% (9/19) | 4.5 / 4.5 min | 1.01 / hr | **47.4% (9/19)** [26.3%, 68.4%] | **2.3 / 2.2 min** | **1.01 / hr** (4 FA) | **57.9% (11/19)** | Borderline |
| **Local Outlier Factor (LOF)** | 57.9% (11/19) | 4.6 / 4.8 min | **1.51 / hr** | **57.9% (11/19)** [36.8%, 78.9%] | **4.5 / 4.8 min** | **1.51 / hr** (6 FA) | **57.9% (11/19)** | **NO (Exceeds Budget!)** |

### Finding 1: Dynamic Exogenous Regression vs. Post-Hoc Residual Fusion
- **Observation:** In cross-event LOEO-CV (Protocol 2), dynamic exogenous regression (**SARIMAX**) matches the best multisensor score (**63.2%**, 12/19 events) while achieving longer lead time (**4.8 min mean, 4.8 min median**) than balanced fusion ($w=0.5$: 4.0 min mean, 3.2 min median). Increasing temperature weight ($w=0.7$) degrades detection to 52.6% (10/19).
- **Physical Mechanism:** Thermal inertia causes downhole fluid temperature (`T-TPT`) to lag acoustic pressure drop (`P-TPT`) by 60–120 seconds during incipient valve throttling. Feeding temperature dynamically as an exogenous input in SARIMAX conditions the autoregressive prediction equation directly, capturing thermodynamic relationships without diluting early acoustic pressure departures.

### Finding 2: Failure of Density-Based Detectors (LOF) under Operational Budgets
- LOF achieves 57.9% detection but incurs **1.51 false alarms/hr** (6 false alarms) even at its strictest persistence setting ($p=3, q=0.99$). It cannot satisfy the 1.0/hr budget on unseen wells.
- **Mechanism:** Static baseline pressure varies across the 5 physical wells from 8.9 to 21.0 MPa. Raw Euclidean distance in $[P, T]$ space flags normal operating conditions of unseen wells as anomalies. Time-series forecasting models difference out static level shifts, though residuals remain subject to variance changes and operational drift.

### Finding 3: Calibration FAR vs. Realized Holdout FAR Divergence
- Because individual event calibration segments in 3W are brief ($H_{\text{calib}} < 0.5\text{ h}$), selecting a threshold that produces zero false alarms on calibration data does not guarantee that holdout FAR will remain below budget. Realized holdout FAR must be reported separately.

---

## 5. Scope of Scientific Claims

We do not claim that multisensor composite scoring is uniformly superior to univariate forecasting, nor do we assert that this is the first study of time series anomaly detection. Rather, the contribution is **an event-level evaluation framework that explicitly examines early-warning detection and false-alarm trade-offs on the selected real 3W Class 2 cohort**, establishing:
1. That dynamic regression (SARIMAX) is an effective formulation that matches or exceeds heuristic residual fusion;
2. That time-series forecasting models respect operational false-alarm constraints on unseen wells better than raw distance-based density methods (LOF);
3. That detected events had an observed mean lead time of approximately 4 to 5 minutes under the experimental definition. We do not claim this retrospective lead time proves sufficient time for safe manual or automated intervention, closed-loop choke control, or water hammer prevention. Full dynamic validation in multiphase simulators (e.g., OLGA) remains essential future work.
