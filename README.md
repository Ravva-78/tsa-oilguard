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
3. **Leak-Free Leave-One-Event-Out Validation (LOEO-CV):** Thresholds and persistence parameters are selected strictly on 18 training events and evaluated on the 19th unseen holdout event, benchmarked against Local Outlier Factor (LOF) with 95% bootstrap confidence intervals.

---

## Key Experimental Results

### 1. Statistical Forecasting Accuracy (Test Split)
Evaluated across 19 authentic real-world offshore shut-in events:

| Model | MAE (MPa) | RMSE (MPa) | MAPE (%) | R² Score | Bias (MPa) | Performance vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA(p, d, q)** | **0.0163** | **0.0524** | **8.69%** | **0.9994** | +0.0075 | **47.1% Lower MAE** |
| **SARIMAX (+T-TPT)** | **0.0162** | **0.0523** | **8.63%** | **0.9994** | +0.0074 | **47.4% Lower MAE** |
| **Naive Baseline** | 0.0307 | 0.0781 | 18.22% | 0.9988 | +0.0194 | Reference Baseline |

### 2. Unbiased Benchmark Comparison (LOEO-CV, Budget $B_{\text{FAR}} \le 1.0\text{ FA/hr}$)
Evaluated with 1,000-iteration bootstrap 95% Confidence Intervals:

| Method / Model | Unbiased Detection Rate [95% CI] | Early Hits | Mean Lead Time [95% CI] | Realized FAR (/hr) [95% CI] | Budget Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Univariate SARIMAX** | **68.4%** [47.4%, 89.5%] | **13 / 19** | **5.40 min** [3.86, 6.86] | **0.75 / hr** [0.00, 1.52] | **YES** ($\le 1.0$) |
| **OILGUARD-AC (Multisensor, w=0.7)** | **47.4%** [26.3%, 68.4%] | 9 / 19 | **3.78 min** [2.07, 5.33] | **0.75 / hr** [0.00, 1.52] | **YES** ($\le 1.0$) |
| **Univariate ARIMA** | **42.1%** [21.1%, 68.4%] | 8 / 19 | **4.31 min** [2.43, 6.20] | **0.50 / hr** [0.00, 1.15] | **YES** ($\le 1.0$) |
| **Naive Persistence Baseline** | **47.4%** [26.3%, 68.4%] | 9 / 19 | **4.48 min** [2.50, 6.43] | **0.50 / hr** [0.00, 1.12] | **YES** ($\le 1.0$) |
| **Local Outlier Factor (LOF)** | **73.7%** [52.6%, 89.6%] | 14 / 19 | **5.54 min** [4.14, 6.86] | **2.01 / hr** [0.84, 3.01] | **NO (2× Budget Violation!)** |

*Key finding: Non-forecasting density methods (LOF) suffer from cross-well density shift and violate operational false alarm budgets by 2×. In contrast, time-series forecasting models dynamically adapt to baseline drift, satisfying the budget while delivering **5.4 minutes** of advance early warning.*

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
