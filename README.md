# OILGUARD: Forecasting-Based Early Warning of Abnormal Events in Oil Wells

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

**OILGUARD** is an academic Time Series Analysis (TSA) research project investigating whether statistical forecasting residuals from **ARIMA** and **SARIMAX** models can provide reliable, actionable early warnings of abnormal events in offshore oil wells prior to transient onset, outperforming standard industrial persistence baselines.

The project evaluates authentic offshore well telemetry from the **Petrobras 3W benchmark dataset**, focusing on **Class 2: Spurious Closure of Downhole Safety Valve (DHSV)** across an audited gold cohort of **19 real-world events**.

---

## Key Experimental Results

### 1. Statistical Forecasting Accuracy (Test Split)
Evaluated across 19 authentic real-world offshore shut-in events:

| Model | MAE (MPa) | RMSE (MPa) | MAPE (%) | R² Score | Bias (MPa) | Performance vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA(p, d, q)** | **0.0163** | **0.0524** | **8.69%** | **0.9994** | +0.0075 | **47.1% Lower MAE** |
| **SARIMAX (+T-TPT)** | **0.0162** | **0.0523** | **8.63%** | **0.9994** | +0.0074 | **47.4% Lower MAE** |
| **Naive Baseline** | 0.0307 | 0.0781 | 18.22% | 0.9988 | +0.0194 | Reference Baseline |

### 2. Unbiased Early Warning Performance (Leave-One-Event-Out CV)
To prevent optimization leakage, thresholds were calibrated strictly on training folds and evaluated on unseen holdout events:

| Model | CV Strategy | Denominator | Unbiased Detection Rate | Early Hits | Mean Lead Time | Median Lead Time | FAR (/hr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARIMA** | LOEO-CV | 19 | **42.1%** | 8 | **5.10 min** | **5.17 min** | 4.31 |
| **SARIMAX** | LOEO-CV | 19 | **36.8%** | 7 | 4.26 min | 4.50 min | 5.17 |
| **Naive Baseline** | LOEO-CV | 19 | 36.8% | 7 | 1.69 min | 1.17 min | 3.44 |

*Key finding: In honest cross-validation, ARIMA delivers **over 3× greater advance lead time** than the Naive baseline (5.10 min vs 1.69 min).*

### 3. Residual Optimization & Sensitivity Ceilings
Evaluating retrospective parameter grids and EWMA filtered residuals on the 19-event cohort:

| Technique & Configuration | ARIMA Detection Rate | Naive Baseline | ARIMA Lead Time | False Alarm Rate | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pareto Grid Search ($p=1, q=0.95$)** | **63.2%** (12/19) | 52.6% (10/19) | **5.35 min** | 3.44 / hr | **+10.6% Higher Detection** |
| **EWMA Filtered ($S_t, \lambda=0.3, q=0.975$)** | **57.9%** (11/19) | 42.1% (8/19) | **3.53 min** | 1.72 / hr | **+15.8% Higher Detection** |

---

## Deliverables Directory

All completed project deliverables are organized as follows:

- 📓 **Jupyter Walkthrough Notebook:** [`notebooks/oilguard_walkthrough.ipynb`](notebooks/oilguard_walkthrough.ipynb)  
  *Fully executed reproducible walkthrough covering manifest audit, stationarity tests, forecasting, residual analysis, baseline diagnosis, and optimization.*
- 📄 **Word Technical Report:** [`report/OILGUARD_Technical_Report.docx`](report/OILGUARD_Technical_Report.docx)  
  *Publication-grade formatted Word document with embedded high-resolution figures, tables, and mathematical formulations.*
- 📝 **Markdown Technical Report:** [`report/OILGUARD_Technical_Report.md`](report/OILGUARD_Technical_Report.md)  
  *Git-viewable version of the technical report.*
- 📊 **PowerPoint Presentation:** [`report/OILGUARD_Presentation.pptx`](report/OILGUARD_Presentation.pptx)  
  *16:9 widescreen presentation slides covering industrial context, dataset filtering, forecasting results, anomaly diagnosis, and optimization.*
- 📰 **Blog Article:** [`report/OILGUARD_Blog_Post.md`](report/OILGUARD_Blog_Post.md)  
  *Engaging 1,150-word technical article explaining the project, the noisy baseline paradox, and the real-world value of 5-minute early warnings.*

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
│   └── OILGUARD_Blog_Post.md          # 1,150-word technical blog post
├── results/
│   ├── figures/                      # High-res plots (Pareto front, EDA, Forecasts)
│   ├── metrics/                      # Forecast, warning, and optimization CSVs
│   └── tables/                       # Candidate audit, stationarity, and ARIMA orders
├── scripts/
│   ├── run_pipeline.py               # Main pipeline orchestration entry point
│   └── optimize_warning.py           # Threshold grid search & EWMA optimization
├── src/oilguard/                     # Modular Python package
│   ├── config.py                     # Configuration dataclasses & loader
│   ├── data.py                       # Data ingest, resampling & quality audits
│   ├── eda.py                        # Visualization generation routines
│   ├── evaluation.py                 # Forecast metrics (MAE, RMSE, MAPE, R2)
│   ├── forecasting.py                # Rolling ARIMA, SARIMAX, and Naive models
│   ├── manifest.py                   # 3W parquet scanner & metadata builder
│   ├── pipeline.py                   # End-to-end execution pipeline
│   └── warning.py                    # Threshold calibration & onset logic
├── tests/
│   └── test_core.py                  # Unit tests (100% passing)
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

### 2. Run Unit Tests
```powershell
pytest tests/ -v
```

### 3. Run Warning System Optimization
```powershell
python scripts/optimize_warning.py
```
Outputs the Pareto frontier figure to `results/figures/warning_optimization_pareto.png` and summary tables to `results/metrics/`.

### 4. Run the Full Pipeline
```powershell
python scripts/run_pipeline.py --config config/experiment.json
```

---

## Citation & References

- **Petrobras 3W Dataset:** Vargas, R. E. V. et al. (2019). *A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells.* Journal of Petroleum Science and Engineering.
- **Statistical Process Control:** Montgomery, D. C. (2009). *Introduction to Statistical Quality Control.* John Wiley & Sons.
- **Time Series Analysis:** Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control.* John Wiley & Sons.
