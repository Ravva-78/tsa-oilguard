# Research Gap and Novelty Statement: OILGUARD-AC

**Project:** OILGUARD — Forecasting-Based Early Warning of Abnormal Events in Oil Wells  
**Proposed Method:** OILGUARD-AC (Alarm-Constrained, Multisensor Early Warning)  
**Date:** October 2026  

---

## 1. Context and Problem Statement

Downhole Safety Valve (DHSV) spurious closure (Petrobras 3W Class 2) is a critical subsea shut-in hazard. In current industrial practice, alarms trigger reactively after flow has collapsed, resulting in potential water hammer shocks, shut-in tubing pressure spikes, and unscheduled production loss. 

Recent research on the Petrobras 3W benchmark has applied machine learning algorithms—ranging from One-Class Classifiers (Fernandes et al., 2023) to Deep Learning (Vargas et al., 2019/2026)—to classify abnormal states. However, existing work suffers from three critical limitations:
1. **Target Ambiguity:** Models evaluate macro-classification accuracy across the entire file (including transient and steady-state fault periods) rather than measuring prospective advance lead time prior to transient onset ($T_{\text{transient}}$).
2. **The "Unconstrained False Alarm" Trap:** Optimization techniques tune thresholds to maximize detection without constraining normal-operation false alarms. In offshore control rooms, a warning system generating 4+ false alarms per hour induces alarm fatigue and will be silenced.
3. **Univariate Isolation:** Most forecasting detectors monitor downhole pressure (`P-TPT`) alone, neglecting the thermodynamic relationship between pressure drop and downhole temperature (`T-TPT`).

---

## 2. The Research Question

> **Primary Research Question:**  
> *Can calibrated, multisensor forecasting residuals improve early detection of spurious DHSV closure while meeting a predefined false-alarm budget on unseen real-world events?*

### Sub-Questions:
1. Does combining pressure forecast residuals with downhole temperature deviation evidence provide earlier or more reliable anomaly detection than univariate pressure forecasting?
2. Does calibrating detection thresholds under an explicit false-alarm constraint ($B_{\text{FAR}} \le 1.0\text{ or } 2.0\text{ alarms/hour}$) on training/calibration data successfully control false alarm rates on unseen holdout well events?
3. How does forecasting-based residual anomaly detection compare against established non-forecasting anomaly detectors (specifically Local Outlier Factor, identified by Fernandes et al., 2023 as the top performer on 3W)?

---

## 3. The Proposed Technical Contribution: OILGUARD-AC

We propose **OILGUARD-AC (Alarm-Constrained, Multisensor Early Warning for Spurious DHSV Closure)** consisting of three core technical mechanisms:

### Contribution 1: Multisensor Residual Coupling Score
Rather than monitoring univariate raw residual $e_P(t) = y_P(t) - \hat{y}_P(t)$ alone:
1. We compute normalized pressure residual:
   $$z_{P, t} = \frac{|e_{P, t}|}{\hat{\sigma}_{\text{calib}, P}}$$
2. We compute normalized temperature deviation from rolling calibration baseline:
   $$z_{T, t} = \frac{|T_t - \mu_{\text{calib}, T}|}{\hat{\sigma}_{\text{calib}, T}}$$
3. We compute a composite thermodynamic anomaly score:
   $$S_t = \sqrt{w_P z_{P, t}^2 + (1 - w_P) z_{T, t}^2}$$
   where $w_P \in [0.5, 1.0]$ balances pressure dominance with thermal corroboration.

### Contribution 2: False-Alarm-Budgeted Threshold Calibration
Given an operational nuisance-alarm budget $B_{\text{FAR}}$ (e.g., $B \le 1.0\text{ FA/hr}$ or $B \le 2.0\text{ FA/hr}$):
- Threshold $\tau$ is calibrated strictly on normal operating data such that:
  $$\text{FAR}_{\text{calib}}(\tau) \le B_{\text{FAR}}$$
- This replaces arbitrary percentile selection ($q=0.99$ or post-hoc $q=0.95$) with an operationally motivated constraint.
- The realized false alarm rate on holdout test data is evaluated to test calibration robustness.

### Contribution 3: Leakage-Free Event-Level Validation (LOEO-CV) with Established Comparator
- Evaluated strictly using Leave-One-Event-Out Cross-Validation (LOEO-CV) on all 19 authentic Class 2 events.
- Benchmarked against:
  1. Univariate ARIMA (pressure residual alone)
  2. Univariate SARIMAX (pressure residual with temperature exog)
  3. Naive Persistence Baseline
  4. Non-forecasting benchmark: **Local Outlier Factor (LOF)** fitted on normal calibration windows.

---

## 4. What Is Novel vs. What Is Prior Art

| Component | Status | Source / Prior Art |
| :--- | :--- | :--- |
| **ARIMA / SARIMAX Modeling** | Prior Art | Box-Jenkins (2015); standard time series literature. |
| **Petrobras 3W Benchmark** | Prior Art | Vargas et al. (2019, 2026). |
| **Local Outlier Factor (LOF)** | Prior Art | Breunig et al. (2000); applied to 3W by Fernandes et al. (2023). |
| **Multisensor Bivariate Residual Score** | **Proposed Novelty** | Evaluates whether thermodynamic coupling between `P-TPT` and `T-TPT` provides earlier incipient warning than pressure residuals alone on authentic 3W data. |
| **Alarm-Constrained Calibration Framework** | **Proposed Novelty** | Calibrating detection thresholds strictly under an explicit false-alarm rate budget ($B_{\text{FAR}}$) on training/calibration data, and evaluating realized FAR and lead times under LOEO-CV. |

---

## 5. Negative Finding & Scientific Honesty Policy

We explicitly commit to the following principles:
- **No Cherry-Picking:** All metrics will be evaluated across the verified 19 gold events.
- **Hypothesis Testing:** If OILGUARD-AC does *not* outperform univariate ARIMA or LOF under the same false-alarm constraints, we will report the negative finding with technical root-cause analysis (e.g., thermal sensor inertia vs acoustic pressure dynamics).
- **No Unverifiable Claims:** We will not claim water hammer prevention or SCADA deployment readiness; all findings will be characterized as empirical observations on the Petrobras 3W benchmark.
