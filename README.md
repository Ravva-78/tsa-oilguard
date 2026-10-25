# OILGUARD: Forecasting-Based Early Warning of Abnormal Events in Oil Wells

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

**OILGUARD** is a Time Series Analysis (TSA) research project investigating whether statistical forecasting residuals from **ARIMA** and **SARIMAX** models can provide reliable, actionable early warnings of abnormal events in offshore oil wells prior to transient onset, outperforming standard industrial persistence baselines and non-forecasting anomaly detectors (Local Outlier Factor, LOF).

The project evaluates authentic offshore well telemetry from the **Petrobras 3W benchmark dataset**, focusing on **Class 2: Spurious Closure of Downhole Safety Valve (DHSV)** across an audited gold cohort of **19 real-world events**.

---

## Research Gap & Contribution: OILGUARD-AC

Previous benchmark studies (Vargas et al. 2019/2026, Fernandes et al. 2023) primarily frame anomaly detection on the 3W dataset as macroscopic fault classification across entire event files, often tuning detection thresholds post-hoc without operational false-alarm constraints.

**OILGUARD-AC (Alarm-Constrained, Multisensor Early Warning)** addresses this gap with three technical contributions:
1. **Multisensor Bivariate Residual Scoring:** Combines standardized pressure forecast residuals (`P-TPT`) with downhole temperature deviations (`T-TPT`).
2. **False-Alarm-Budgeted Threshold Calibration:** Calibrates detection thresholds on normal operating data subject to an operational nuisance-alarm budget ($B_{\text{FAR}} \le 1.0\text{ alarms/hour}$).
3. **Rigorous Separation of Evaluation Protocols:**
   - **Protocol 1 (Prospective Within-Event Split):** Fits models on historical normal data ($T < T_{\text{calib}}$) and calibrates thresholds on well-specific calibration slices ($H_{\text{calib}} < 0.5\text{ h}$).
   - **Protocol 2 (Genuine Cross-Well LOEO-CV):** Holds out the entire event $k$ from model fitting and pool-calibrates thresholds strictly on the remaining 18 wells. Evaluated with 1,000-iteration bootstrap 95% Confidence Intervals.

---

## Key Experimental Results

### 1. Statistical Forecasting Accuracy (Test Split)
Evaluated across 19 authentic real-world offshore shut-in events:

| Model | MAE (MPa) | RMSE (MPa) | MAPE (%) | R² Score | Bias (MPa) | Performance vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA(p, d, q)** | **0.0163** | **0.0524** | **8.69%** | **0.9994** | +0.0075 | **47.1% Lower MAE** |
| **SARIMAX (+T-TPT)** | **0.0162** | **0.0523** | **8.63%** | **0.9994** | +0.0074 | **47.4% Lower MAE** |
| **Naive Baseline** | 0.0307 | 0.0781 | 18.22% | 0.9988 | +0.0194 | Reference Baseline |

### 2. Audited Matched-FAR Benchmark Comparison (N=19 Gold Events)
To ensure a scientifically fair comparison without threshold bias, models were evaluated at a matched operational realized false-alarm rate ceiling ($\text{FAR} \le 0.75\text{ FA/hr}$):

| Model / Method | Protocol 1 Config | P1 Early Detection | P1 Lead Time | P1 Realized FAR | Protocol 2 Config | P2 Early Detection | P2 Lead Time | P2 Realized FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMAX (exog T)** | $q=0.975, p=2$ | 36.8% (7/19) | 4.00 min | 0.25 / hr | $q=0.90, p=2$ | **63.2% (12/19)** | **4.83 min** | **0.50 / hr** |
| **OILGUARD-AC ($w=0.5$)** | $q=0.95, p=2$ | **68.4% (13/19)** | **4.21 min** | **0.75 / hr** | $q=0.95, p=2$ | **63.2% (12/19)** | **4.01 min** | **0.50 / hr** |
| **OILGUARD-AC ($w=0.7$)** | $q=0.90, p=3$ | 57.9% (11/19) | 2.79 min | 0.75 / hr | $q=0.90, p=2$ | 52.6% (10/19) | 4.67 min | 0.75 / hr |
| **Univariate ARIMA** | $q=0.975, p=2$ | 31.6% (6/19) | 2.47 min | 0.25 / hr | $q=0.90, p=2$ | 57.9% (11/19) | 5.18 min | 0.50 / hr |
| **Naive Baseline** | $q=0.99, p=2$ | 42.1% (8/19) | 3.31 min | 0.75 / hr | $q=0.90, p=3$ | 42.1% (8/19) | 3.46 min | 0.50 / hr |
| **Local Outlier Factor (LOF)** | $q=0.99, p=3$ | 57.9% (11/19) | 4.61 min | **1.51 / hr** | $q=0.99, p=3$ | 57.9% (11/19) | 4.48 min | **1.51 / hr (Violates Budget!)** |

### Key Scientific Lessons
1. **Dynamic Conditioning vs. Post-Hoc Fusion:** In cross-well LOEO-CV, SARIMAX (exogenous temperature) matches or outperforms post-hoc composite scoring ($w=0.7$). Due to thermal inertia, downhole temperature changes lag acoustic pressure drops by 60–120 seconds during incipient valve throttling. Conditioning pressure forecasts directly on temperature dynamics provides superior early warning.
2. **Failure of Non-Forecasting Density Baselines:** LOF achieves 57.9% detection but incurs 1.51 false alarms/hr (violating the 1.0/hr budget). Because baseline pressure varies across wells from 8.9 to 21.0 MPa, raw distance metrics mistake inter-well variance for anomalies.
3. **Calibration FAR vs. Realized Holdout FAR:** Calibrating on short event-specific slices ($<0.5\text{ h}$) leads to budget collapse ($K_{\text{allowed}} = 0$). Pooling 18 training wells in Protocol 2 creates a stable calibration baseline.
4. **Qualified Operational Lead Time:** The observed 4 to 5 minutes of early warning provides an actionable diagnostic window for control-room operators. Validation in high-fidelity multiphase transient simulators (e.g. OLGA) is recommended before field commissioning.

---

## Deliverables Directory

All completed project deliverables are organized as follows:

- 📓 **Jupyter Walkthrough Notebook:** [`notebooks/oilguard_walkthrough.ipynb`](notebooks/oilguard_walkthrough.ipynb)  
  *Fully executed reproducible walkthrough covering manifest audit, stationarity tests, forecasting, residual analysis, baseline diagnosis, and OILGUARD-AC benchmark evaluation.*
- 📄 **Word Technical Report:** [`report/OILGUARD_Technical_Report.docx`](report/OILGUARD_Technical_Report.docx)  
  *Publication-grade formatted Word document with embedded high-resolution figures, tables, and mathematical formulations.*
- 📝 **Markdown Technical Report:** [`report/OILGUARD_Technical_Report.md`](report/OILGUARD_Technical_Report.md)  
  *Git-viewable version of the technical report.*
- 📊 **PowerPoint Presentation:** [`report/OILGUARD_Presentation.pptx`](report/OILGUARD_Presentation.pptx)  
  *16:9 widescreen presentation slides covering industrial context, literature audit, forecasting results, OILGUARD-AC framework, and benchmark results.*
- 📰 **Blog Article:** [`report/OILGUARD_Blog_Post.md`](report/OILGUARD_Blog_Post.md)  
  *Engaging 1,250-word technical article explaining the project, the literature gap, the alarm-budget trade-off, and why forecasting beats density methods on unseen wells.*
- 📚 **Research Documentation:**
  - Literature Review: [`research/LITERATURE_REVIEW.md`](research/LITERATURE_REVIEW.md)
  - Research Gap & Novelty: [`research/RESEARCH_GAP_AND_NOVELTY.md`](research/RESEARCH_GAP_AND_NOVELTY.md)
  - Experiment Plan: [`research/EXPERIMENT_PLAN.md`](research/EXPERIMENT_PLAN.md)

---

## Project Architecture

```
TSA-OILGUARD/
├── config/
│   └── experiment.json               # Typed pipeline & model hyperparameters
├── data/
│   └── processed/                    # Scanned manifest.parquet & 19 per-event parquets
├── notebooks/
│   └── oilguard_walkthrough.ipynb     # Executed reproducible walkthrough
├── report/
│   ├── OILGUARD_Technical_Report.docx # Formatted Word report
│   ├── OILGUARD_Technical_Report.md   # Markdown technical report
│   ├── OILGUARD_Presentation.pptx     # 16:9 PowerPoint slides
│   └── OILGUARD_Blog_Post.md          # 1,250-word technical blog post
├── research/
│   ├── LITERATURE_REVIEW.md          # In-depth literature review & comparative table
│   ├── RESEARCH_GAP_AND_NOVELTY.md   # Research gap, hypothesis & novelty statement
│   └── EXPERIMENT_PLAN.md            # Technical experiment roadmap
├── results/
│   ├── figures/                      # High-res plots (Pareto front, LOEO benchmarks, trade-offs)
│   ├── metrics/                      # Forecast, warning, and optimization CSVs
│   └── tables/                       # Candidate audit, stationarity, and ARIMA orders
├── scripts/
│   ├── run_pipeline.py               # Main pipeline orchestration entry point
│   ├── optimize_warning.py           # Threshold grid search & EWMA optimization
│   └── run_oilguard_ac_experiments.py# OILGUARD-AC ablation, calibration & LOEO-CV runner
├── src/oilguard/                     # Modular Python package
│   ├── config.py                     # Configuration dataclasses & loader
│   ├── data.py                       # Data ingest, resampling & quality audits
│   ├── eda.py                        # Visualization generation routines
│   ├── evaluation.py                 # Forecast metrics (MAE, RMSE, MAPE, R2)
│   ├── forecasting.py                # Rolling ARIMA, SARIMAX, and Naive models
│   ├── manifest.py                   # 3W parquet scanner & metadata builder
│   ├── oilguard_ac.py                # Multisensor scoring, budgeted calibration & LOF
│   ├── pipeline.py                   # End-to-end execution pipeline
│   └── warning.py                    # Threshold calibration & onset logic
├── tests/
│   ├── test_core.py                  # Core pipeline unit tests (100% passing)
│   └── test_oilguard_ac.py           # OILGUARD-AC unit tests (100% passing)
├── .gitignore                        # Git exclusion rules
├── pyproject.toml                    # Pytest and ruff configuration
└── requirements.txt                  # Pinned dependencies
```

---

## Quick Start & Reproduction

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run All Unit Tests
```powershell
pytest tests/ -v
```

### 3. Run OILGUARD-AC Research Experiments
```powershell
python scripts/run_oilguard_ac_experiments.py
```
Outputs benchmark tables to `results/metrics/` and figures to `results/figures/oilguard_ac_benchmarks.png` and `results/figures/oilguard_ac_tradeoff.png`.

### 4. Run the Full Forecasting Pipeline
```powershell
python scripts/run_pipeline.py --config config/experiment.json
```

---

## Citation & References

1. **Vargas, R. E. V., et al. (2019).** *A realistic and public dataset with rare undesirable real events in oil wells.* Journal of Petroleum Science and Engineering, 181, 106223. [DOI: 10.1016/j.petrol.2019.106223](https://doi.org/10.1016/j.petrol.2019.106223)
2. **Vargas, R. E. V., et al. (2026).** *3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells.* Scientific Data, 13, 949. [DOI: 10.1038/s41597-026-07225-z](https://doi.org/10.1038/s41597-026-07225-z)
3. **Fernandes Jr., W., Komati, K. S., & Gazolli, K. (2023/2024).** *Anomaly detection in oil-producing wells: a comparative study of one-class classifiers in a multivariate time series dataset.* Journal of Petroleum Exploration and Production Technology, 14, 523–542. [DOI: 10.1007/s13202-023-01710-6](https://doi.org/10.1007/s13202-023-01710-6)
4. **Wang, Y., et al. (2024).** *Graph Structure Change-Based Anomaly Detection in Multivariate Time Series of Industrial Processes.* IEEE Transactions on Industrial Informatics, 20(6), 8440–8450. [DOI: 10.1109/TII.2023.3347000](https://doi.org/10.1109/TII.2023.3347000)
5. **Khan, A., Al Farid, F., Uddin, J., et al. (2026).** *Early-warning industrial fault detection based on physics-guided residual learning and calibrated CRNNs.* Scientific Reports, 16, 8812. [DOI: 10.1038/s41598-026-48227-6](https://doi.org/10.1038/s41598-026-48227-6)
