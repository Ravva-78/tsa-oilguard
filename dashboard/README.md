# OILGUARD Interactive Dashboard

A Streamlit dashboard for the OILGUARD-AC research project.

## Features
- Dark offshore command-center UI with interactive Plotly charts
- Audited benchmark snapshot and detection-vs-false-alarm trade-off
- Selectable nuisance-alarm budget
- Event-level explorer backed by `results/audited_v2/` CSVs
- Upload-and-replay interface for pressure telemetry using one-step-ahead ARIMA or SARIMAX forecasts
- Residual score, persistence-based warning markers, and downloadable replay CSV
- Explicit guardrails: no SCADA connection, no equipment control, not a certified safety system

## Install in your existing project

Copy `app.py` and `requirements-dashboard.txt` into your `TSA-OILGUARD` project root, beside `README.md` and the `results/` directory. Then run in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard looks for:
- `results/audited_v2/metrics/matched_far_comparison.csv`
- `results/audited_v2/metrics/per_event_outcomes.csv`
- `results/audited_v2/tables/cohort_manifest.csv`
- `results/audited_v2/tables/evaluation_definitions.csv`

If files are absent, reference cards still render but event-level exploration will be unavailable.

## Important
The embedded benchmark snapshot comes from the supplied OILGUARD audit summary and must be checked against `matched_far_comparison.csv` before presentation. The uploaded telemetry replay uses a user-selected threshold and is exploratory; it is not the same as the audited cross-well threshold calibration. Do not use it for real-world safety decisions.
