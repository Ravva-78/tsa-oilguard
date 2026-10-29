"""Populate the official BMSCE TSA AAT Report Template with complete OILGUARD project content.

Strictly preserves the template structure, headers, valuation report, and formatting,
ensuring all text, tables, figures, citations, and metadata are inserted in their exact
chronological sections.
"""

from __future__ import annotations

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
TEMPLATE_BACKUP = REPORT_DIR / "TSA AAT REPORT Template - Lab - Backup.docx"
TARGET_DOCX = REPORT_DIR / "TSA AAT REPORT Template - Lab.docx"
FIGURES_DIR = PROJECT_ROOT / "results" / "audited_v2" / "figures"
ACCURACY_FIG = PROJECT_ROOT / "results" / "figures" / "forecast_accuracy.png"
CUTAWAY_IMG = PROJECT_ROOT / "assets" / "images" / "subsea_wellbore_cutaway.jpg"


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_formatted_p(target_p, text: str, bold: bool = False, italic: bool = False, space_after: int = 6, align=WD_ALIGN_PARAGRAPH.LEFT):
    new_p = target_p.insert_paragraph_before()
    new_p.alignment = align
    new_p.paragraph_format.line_spacing = 1.15
    new_p.paragraph_format.space_after = Pt(space_after)
    r = new_p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    return new_p


def add_bullet_p(target_p, bold_prefix: str, text: str):
    new_p = target_p.insert_paragraph_before()
    new_p.paragraph_format.line_spacing = 1.15
    new_p.paragraph_format.space_after = Pt(4)
    new_p.paragraph_format.left_indent = Inches(0.25)
    r_bullet = new_p.add_run("•  ")
    r_bullet.font.name = "Times New Roman"
    r_bullet.font.size = Pt(12)
    r_bullet.font.bold = True
    
    r_pre = new_p.add_run(bold_prefix)
    r_pre.font.name = "Times New Roman"
    r_pre.font.size = Pt(12)
    r_pre.font.bold = True
    
    r_body = new_p.add_run(text)
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(12)
    return new_p


def add_image_p(target_p, img_path: Path, caption: str, width_in=5.0):
    if not img_path.exists():
        return
    p_img = target_p.insert_paragraph_before()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_img = p_img.add_run()
    r_img.add_picture(str(img_path), width=Inches(width_in))
    p_cap = target_p.insert_paragraph_before()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10)
    r_cap.font.italic = True


def find_p(doc, query: str):
    for p in doc.paragraphs:
        if query in p.text:
            return p
    raise ValueError(f"Paragraph containing '{query}' not found")


def main():
    if not TEMPLATE_BACKUP.exists():
        raise FileNotFoundError(f"Template backup missing at {TEMPLATE_BACKUP}")

    doc = docx.Document(str(TEMPLATE_BACKUP))

    # -------------------------------------------------------------------------
    # 1. COVER PAGE: Title, Application Domain, SDG, and Table 0
    # -------------------------------------------------------------------------
    p10 = doc.paragraphs[10]
    p10.runs[1].text = "\tOILGUARD: Forecasting-Based Early Warning of Abnormal Events in Oil Wells"
    p10.runs[1].font.name = "Times New Roman"
    p10.runs[1].font.size = Pt(13)
    p10.runs[1].bold = True

    p11 = doc.paragraphs[11]
    p11.runs[1].text = " Industrial (Offshore Petroleum Systems & Subsea Production Wells)"
    p11.runs[1].font.name = "Times New Roman"
    p11.runs[1].font.size = Pt(13)
    p11.runs[1].bold = True

    p12 = doc.paragraphs[12]
    p12.runs[1].text = "\tSDG 9 (Industry & Innovation), SDG 12 (Responsible Production), SDG 14 (Life Below Water)"
    p12.runs[1].font.name = "Times New Roman"
    p12.runs[1].font.size = Pt(12)
    p12.runs[1].bold = True

    # Populate Table 0 (Submitted by)
    t0 = doc.tables[0]
    t0.rows[0].cells[1].text = "Ravva Nagarjun & Naveen Kumar N"
    t0.rows[1].cells[1].text = "1BM21AI000 & 1BM21AI001"
    t0.rows[2].cells[1].text = "7th Semester, AIML"
    t0.rows[3].cells[1].text = "Ravva Nagarjun & Naveen Kumar N (Submitted Online)"
    for r in t0.rows:
        for c in r.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(11)

    # -------------------------------------------------------------------------
    # 2. ABSTRACT (Inserted before Section 2)
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "INTRODUCTION AND PROBLEM IDENTIFICATION")
    abstract_text = (
        "In offshore deepwater petroleum production, spurious closure of the Downhole Safety Valve "
        "(DHSV, designated as Class 2 in the Petrobras 3W benchmark) represents a high-consequence operational "
        "emergency. The sudden uncommanded shut-in of the emergency subsea barrier isolates the high-pressure "
        "reservoir upstream while triggering catastrophic downstream hydrodynamic pressure collapse and acoustic shock. "
        "Conventional Supervisory Control and Data Acquisition (SCADA) systems rely on static lower-bound pressure "
        "trip thresholds (P < P_trip) that trigger only after the valve is fully seated and flow has ceased, providing "
        "zero advance reaction time for control-room operators. This project presents OILGUARD, a time-series forecasting "
        "early warning framework that leverages rolling 1-step-ahead statistical models (ARIMA and multivariate SARIMAX "
        "with dynamic exogenous temperature) to detect incipient mechanical throttling minutes prior to catastrophic closure. "
        "Evaluated across an audited gold-standard cohort of 19 authentic offshore well events spanning 5 distinct physical "
        "wells from the Petrobras 3W dataset, the proposed statistical forecasting models achieve a 47.4% reduction in "
        "Mean Absolute Error (0.0162 MPa vs. 0.0307 MPa) over standard persistence baselines. When coupled with operational "
        "false-alarm constraints (<= 0.50 to 1.0 alarms/hour) under strict Leave-One-Event-Out (LOEO-CV) and Leave-One-Well-Out "
        "(LOWO-CV) cross-validation, the proposed approach achieves 63.2% to 73.7% event detection with an average advance "
        "lead time of 4.8 minutes, whereas popular non-forecasting density baselines (Local Outlier Factor) fail operational "
        "deployment due to severe cross-well baseline pressure drift."
    )
    add_formatted_p(target_p, abstract_text, space_after=12)

    # -------------------------------------------------------------------------
    # 3. SECTION 2: INTRODUCTION AND PROBLEM IDENTIFICATION
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "2.2 Problem Statement")
    add_formatted_p(
        target_p,
        "Offshore deepwater hydrocarbon extraction represents one of the most critical and hazardous frontiers of modern industrial process engineering. "
        "In deepwater fields such as Brazil's Santos Basin pre-salt province, wells are drilled through water columns exceeding 2,100 meters, penetrating seabed "
        "formations over 2,400 meters below the mudline. The multiphase production fluid (consisting of oil, gas, and formation water) flows through subsea tubing "
        "under extreme pressures (ranging from 8.9 to 21.0 MPa) and elevated temperatures (exceeding 100°C). Operating under such extreme thermodynamic regimes requires "
        "continuous, reliable cyber-physical surveillance to safeguard personnel, platform processing topsides, and marine coastal ecosystems."
    )

    target_p = find_p(doc, "2.3 Need for Forecasting")
    add_formatted_p(
        target_p,
        "The primary barrier protecting offshore wells is the Downhole Safety Valve (DHSV)—a hydraulic, fail-safe flapper valve positioned inside the tubing string "
        "hundreds of meters beneath the seabed. During normal operations, hydraulic control line pressure holds this flapper open at 90° to permit uninterrupted multiphase flow. "
        "However, mechanical latch fatigue, seal degradation, or control-line hydraulic bleeding can cause an uncommanded, spurious closure of the DHSV while the well is operating at full capacity. "
        "This event (classified as Class 2 in the Petrobras 3W benchmark) causes flowing hydrocarbons to violently collide with the sealed steel flapper. Severe water-hammer pressure shockwaves "
        "propagate upstream, while downstream pressure collapses instantaneously toward zero, risking subsea flowline cavitation, flange blowout, and emergency platform shutdowns."
    )

    target_p = find_p(doc, "2.4 Objectives")
    add_formatted_p(
        target_p,
        "Conventional industrial Supervisory Control and Data Acquisition (SCADA) architectures employ static lower-bound pressure trip thresholds (e.g., P < P_trip). "
        "The fatal limitation of this legacy approach is that downhole pressure crosses the trip threshold only AFTER the valve flapper has fully traveled into its seat. "
        "The shockwave has already initiated, and the fluid flow has completely stopped—providing an advance warning lead time of exactly zero seconds. "
        "Forecasting is essential because mechanical valves do not close instantaneously; rather, an incipient phase of flapper flutter and hydraulic throttling occurs minutes before "
        "complete shut-in. By developing dynamic statistical time-series models that predict expected normal pressure trajectories, incipient throttling can be detected as a statistical divergence "
        "(forecast residual spike) 4 to 5 minutes before the catastrophic transient, giving automated systems time to ramp down topside chokes and prevent water-hammer shock."
    )

    target_p = find_p(doc, "2.5 Scope of the Study")
    add_bullet_p(target_p, "Objective 1: ", "Conduct rigorous exploratory data analysis, stationarity testing, and autocorrelation structure profiling on high-frequency offshore downhole telemetry.")
    add_bullet_p(target_p, "Objective 2: ", "Build and validate a univariate baseline time series forecasting model (ARIMA) and persistence benchmark on downhole pressure telemetry.")
    add_bullet_p(target_p, "Objective 3: ", "Formulate an enhanced multivariate forecasting model (SARIMAX with dynamic exogenous temperature) and a bivariate residual fusion score (OILGUARD-AC).")
    add_bullet_p(target_p, "Objective 4: ", "Establish an alarm-constrained threshold calibration framework that strictly enforces offshore false-alarm budgets (B_FAR <= 1.0 alarms/hour) on historical data.")
    add_bullet_p(target_p, "Objective 5: ", "Execute leak-free Leave-One-Event-Out (LOEO-CV) and Leave-One-Well-Out (LOWO-CV) cross-validation across 19 authentic events from 5 physical offshore wells.")

    target_p = find_p(doc, "LITERATURE REVIEW AND RESEARCH GAP")
    add_formatted_p(
        target_p,
        "This study investigates high-frequency downhole pressure (P-TPT, MPa) and downhole temperature (T-TPT, °C) telemetry sampled at 1 Hz and aggregated to 10-second intervals "
        "from the certified gold cohort of 19 authentic offshore Class 2 events in the Petrobras 3W benchmark. The study aims to prospectively detect incipient valve throttling within an "
        "operational 10-minute pre-transient window [T_transient - 10 min, T_transient). The scope is strictly bounded to prospective early-warning anomaly detection; it does not "
        "attempt full reservoir multiphase fluid simulation or replace physical hydrodynamic flow-loop testing."
    )

    # -------------------------------------------------------------------------
    # 4. SECTION 3: LITERATURE REVIEW AND RESEARCH GAP
    # -------------------------------------------------------------------------
    # Populate existing Table 2 with 5 papers
    t2 = doc.tables[2]
    papers_data = [
        (
            "Paper 1: Vargas et al. (2019) [J. Pet. Sci. Eng.]",
            "Rare undesirable event detection; Petrobras 3W Dataset v1.0 (8 event classes, 1,984 files: authentic, simulated, hand-drawn).",
            "Supervised machine learning classification (Random Forest, Support Vector Machines, Multilayer Perceptron).",
            "Established the premier public offshore well benchmark with authentic sensor defects, providing standardized baseline evaluation.",
            "Treats anomaly detection as retrospective time-step classification; does not evaluate prospective lead time before transient onset or normal operational false-alarm rates.",
            "https://doi.org/10.1016/j.petrol.2019.106223"
        ),
        (
            "Paper 2: Vargas et al. (2026) [Nature Sci. Data]",
            "Benchmark curation and data quality standardization; Petrobras 3W Dataset v2.0.0 (2,228 instances, 1.87 GB).",
            "Standardized baseline classifiers, data quality pipelines, and enhanced expert annotation toolkits.",
            "Refined expert annotations, standardized metadata, and strictly separated authentic well files (WELL-*) from numerical/hand-drawn models.",
            "Focuses primarily on benchmark data publishing; leaves prospective early-warning triggering, lead time quantification, and operational nuisance alarm budgets unaddressed.",
            "https://doi.org/10.1038/s41597-026-07225-z"
        ),
        (
            "Paper 3: Fernandes Jr., Komati & Gazolli (2024) [J. Pet. Explor. Prod. Technol.]",
            "Multivariate anomaly detection in oil-producing wells; Petrobras 3W dataset.",
            "Unsupervised One-Class Classification: Local Outlier Factor (LOF), Isolation Forest (IF), One-Class SVM (OCSVM), and Autoencoders.",
            "Reported that LOF achieved highest performance (~87% F1-score) due to local density cluster modeling across multiple operating modes.",
            "Evaluates windowed time-step classification retrospectively; does not measure prospective lead time prior to transient onset, and incurs severe false-alarm penalties across distinct wells due to baseline density drift.",
            "https://doi.org/10.1007/s13202-023-01710-6"
        ),
        (
            "Paper 4: Wang et al. (2024) [IEEE Trans. Ind. Inform.]",
            "Multivariate time series anomaly detection in complex industrial processes; SWaT, WADI, and chemical process plant benchmarks.",
            "Graph Structure Change-Based Multivariate Anomaly Detection (GSC-MAD) combining 1-step prediction errors with dynamic graph neural networks.",
            "Effectively captures relational dependency shifts across inter-sensor topologies under multivariate operational drift.",
            "Requires complex deep learning architectures with high parameter count; uncalibrated for low-sample industrial event regimes (e.g., 19 real well events) and lacks operational false-alarm constraints.",
            "https://doi.org/10.1109/TII.2023.3347000"
        ),
        (
            "Paper 5: Khan, Al Farid, Uddin et al. (2026) [Scientific Reports]",
            "Early-warning industrial fault detection and valve stiction; Tennessee Eastman Process (TEP) chemical benchmark.",
            "Physics-guided twin residuals + Attention-based CRNN (1D-CNN + BiGRU) with Platt scaling probability calibration.",
            "Demonstrated that physics-informed twin residuals paired with calibrated probability thresholds improved fault detection (+17% NAB score vs. IF).",
            "Evaluated on chemical process simulation (TEP) rather than authentic multiphase petroleum telemetry; high parameter count vulnerable to severe overfitting on small industrial event cohorts.",
            "https://doi.org/10.1038/s41598-026-48227-6"
        )
    ]
    for r_idx, p_data in enumerate(papers_data, start=1):
        row = t2.rows[r_idx]
        for c_idx, val in enumerate(p_data):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            if c_idx == 0:
                set_cell_background(cell, "F1F5F9")
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9.5)
                    if c_idx == 0:
                        run.font.bold = True

    target_p = find_p(doc, "3.3 Proposed Approach")
    add_formatted_p(
        target_p,
        "A critical survey of existing literature reveals a profound methodological divide between macroscopic post-onset classification "
        "and genuine prospective early warning. The identified research gaps are threefold:\n"
        "1. Conflation of Classification and Warning: Existing studies on the Petrobras 3W benchmark evaluate windowed time-step F1-scores across entire event durations. "
        "Because macroscopic transient pressure drop is massive, models achieve high accuracy labeling time steps that occur AFTER the valve is already closed. "
        "No study evaluates prospective advance lead time prior to transient onset (T_transient).\n"
        "2. Absence of Operational Alarm Constraints: Published anomaly detectors report detection without measuring realized false-alarm rates during normal steady-state operation. "
        "In offshore SCADA operations, an uncalibrated detector generating multiple alarms per shift causes severe operator fatigue and is immediately decommissioned.\n"
        "3. Cross-Asset Generalization Failure: Unsupervised density models (such as LOF) model raw feature space density. Because operating pressure varies widely across physical completions "
        "(from 8.9 to 21.0 MPa), raw coordinate models suffer severe density drift when deployed across unseen wells, triggering continuous false alarms."
    )

    target_p = find_p(doc, "DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS")
    add_formatted_p(
        target_p,
        "To address these gaps, this project proposes OILGUARD: a leak-free, alarm-constrained early warning architecture combining statistical time series forecasting "
        "(SARIMAX with dynamic exogenous temperature) and bivariate residual scoring (OILGUARD-AC). The system operates recursively: rolling 1-step-ahead forecasts establish "
        "the expected physical trajectory; standardized forecast residuals (z_P) and temperature deviations (z_T) are coupled into an elliptical score S(t) = sqrt(w*z_P^2 + (1-w)*z_T^2); "
        "and alarms are latched only when persistent threshold breaches occur under a pre-calibrated operational false-alarm budget (B_FAR <= 1.0 FA/hr). "
        "The architecture is evaluated under strict Leave-One-Event-Out (LOEO-CV) and Leave-One-Well-Out (LOWO-CV) protocols to prove cross-well reliability."
    )
    add_image_p(target_p, CUTAWAY_IMG, "Figure 3.1: Subsea wellbore cutaway showing mudline (1,850m), Subsea Christmas Tree (XT), and DHSV flapper barrier.")

    # -------------------------------------------------------------------------
    # 5. SECTION 4: DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "4.2 Variables and Data Description")
    add_formatted_p(
        target_p,
        "The empirical evaluation is conducted on the authentic offshore production recordings of the Petrobras 3W Benchmark Dataset v2.0.0, "
        "curated and published by Petróleo Brasileiro S.A. (Petrobras). The dataset represents the premier publicly accessible repository of authentic multiphase "
        "deepwater telemetry capturing rare, catastrophic undesirable events in subsea oil wells."
    )

    target_p = find_p(doc, "4.3 Data Preparation")
    add_formatted_p(
        target_p,
        "The time-series telemetry comprises continuous sensor streams sampled at 1 Hz from permanent downhole gauges:\n"
        "• Target Variable (P-TPT): Downhole pressure measured at the Transdutor de Pressão e Temperatura (TPT) located directly upstream/downstream of the completion, scaled to Megapascals (MPa).\n"
        "• Dynamic Exogenous Variable (T-TPT): Downhole temperature measured at the TPT gauge, recorded in degrees Celsius (°C).\n"
        "• Ground Truth State Labels: Annotated by expert offshore petroleum engineers into Class 0 (steady-state normal production), Class 102 (incipient transient degradation), and Class 2 (steady spurious shut-in state)."
    )

    target_p = find_p(doc, "4.4 Exploratory Data Analysis")
    add_formatted_p(
        target_p,
        "The data preparation pipeline enforces strict operational realism and data hygiene:\n"
        "1. Uniform Resampling: High-frequency 1 Hz streams are resampled to 10-second uniform intervals, attenuating turbulent high-frequency acoustic sensor jitter while preserving mechanical throttling signatures.\n"
        "2. Missing Data Handling: Forward-fill interpolation is bounded to a maximum of 6 consecutive missing samples (60 seconds); any longer sensor dropouts trigger event disqualification.\n"
        "3. Forensic Candidate Screening: All 22 candidate Class 2 instances in the 3W dataset were forensically audited. Three corrupted files were removed: WELL-00010_20171218180131 (unusable load failure; missing pressure sensor), "
        "WELL-00011_20140515083000 (unrecorded sensor gap of 1,608 seconds > 10 min), and WELL-00002_20131104004101 (zero normal variance flatline sensor failure, sigma = 0.0 MPa). "
        "This establishes an audited gold-standard cohort of exactly 19 authentic events originating from 5 distinct physical wells (WELL-00003: 3 events, WELL-00009: 1 event, WELL-00011: 12 events, WELL-00012: 2 events, WELL-00013: 1 event).\n"
        "4. Chronological Splitting: Each event is split strictly chronologically: 60% training (for ARIMA/SARIMAX parameter fitting), 20% calibration (for threshold estimation under false-alarm constraints), and 20% holdout testing."
    )

    target_p = find_p(doc, "BASELINE FORECASTING MODEL")
    add_formatted_p(
        target_p,
        "Exploratory Data Analysis (EDA) reveals critical temporal properties governing offshore pressure telemetry:\n"
        "• Trend and Seasonality: Normal production displays steady mean pressure governed by reservoir drive and surface choke backpressure, with minor cyclical fluctuations from multiphase slugging.\n"
        "• Autocorrelation Function (ACF): ACF of raw pressure displays very slow linear decay across lags 1 to 60, indicating strong non-stationarity and integrated stochastic behavior.\n"
        "• Partial Autocorrelation Function (PACF): PACF cuts off sharply after lag 1, confirming that downhole pressure dynamics follow a dominant first-order autoregressive process AR(1) with first-order integration d=1.\n"
        "• Stationarity Testing: Statistical stationarity was tested using the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests. Raw downhole pressure is non-stationary (ADF test statistic: -2.14, p-value: 0.228; KPSS p-value < 0.01). "
        "Upon first-order differencing (d=1), the series becomes strictly stationary (ADF test statistic: -14.62, p-value < 0.0001; KPSS test statistic: 0.084, p-value > 0.10), establishing d=1 for all forecasting models."
    )

    # -------------------------------------------------------------------------
    # 6. SECTION 5: BASELINE FORECASTING MODEL
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "5.2 Model Methodology")
    add_formatted_p(
        target_p,
        "Two baseline forecasting architectures are selected for benchmark comparison:\n"
        "1. Industrial Naive Persistence Baseline: Predicts that the next pressure reading equals the current reading: P_hat_t = P_{t-1}. "
        "This represents the universal operational baseline used across offshore SCADA systems.\n"
        "2. Univariate ARIMA(1, 1, 1): The classical Box-Jenkins autoregressive integrated moving average model, modeling pressure dynamics exclusively from historical pressure observations."
    )

    target_p = find_p(doc, "5.3 Implementation")
    add_formatted_p(
        target_p,
        "The mathematical formulation of the baseline ARIMA(1, 1, 1) model is defined by:\n"
        "    (1 - phi_1 B)(1 - B)^d P_t = (1 + theta_1 B) epsilon_t\n"
        "where B is the backshift operator (B^k X_t = X_{t-k}), phi_1 is the autoregressive coefficient, theta_1 is the moving average coefficient, "
        "d=1 is the differencing order, and epsilon_t ~ N(0, sigma^2) represents Gaussian white noise. Parameter estimation is performed via maximum likelihood estimation (MLE) "
        "over the 60% training partition, with model orders verified through Akaike Information Criterion (AIC) minimization."
    )

    target_p = find_p(doc, "5.4 Baseline Results")
    add_formatted_p(
        target_p,
        "The baseline models are implemented in Python using statsmodels (v0.15.0). A rolling 1-step-ahead recursive forecasting generator operates prospectively across holdout test slices:\n"
        "    from statsmodels.tsa.arima.model import ARIMA\n"
        "    model = ARIMA(train_p, order=(1, 1, 1), trend='n')\n"
        "    res = model.fit()\n"
        "    forecast = res.forecast(steps=1)\n"
        "    residual = actual_p - forecast\n"
        "Residuals are computed at each time step without future data leakage."
    )

    target_p = find_p(doc, "PROPOSED MODEL AND ENHANCEMENT")
    add_formatted_p(
        target_p,
        "Evaluating rolling 1-step-ahead forecasts over untouched holdout test splits across all 19 gold events demonstrates that statistical modeling captures authentic pressure trajectories with remarkable precision:\n"
        "• ARIMA(1, 1, 1): Mean Absolute Error (MAE) = 0.0163 MPa, Root Mean Squared Error (RMSE) = 0.0524 MPa, Mean Absolute Percentage Error (MAPE) = 8.69%, and R^2 = 0.9994.\n"
        "• Naive Baseline: MAE = 0.0307 MPa, RMSE = 0.0781 MPa, MAPE = 18.22%, and R^2 = 0.9988.\n"
        "The statistical ARIMA model reduces MAE by 47.1% and MAPE by more than half compared to persistence.\n\n"
        "Discovery of the 'Noisy Baseline Paradox': When residual thresholding was first applied to early warning without false-alarm tracking, the Naive model appeared to detect 7 of 19 events with 4.5 minutes lead time. "
        "Forensic analysis revealed that because the Naive baseline merely lags 1-Hz sensor streams, its residuals are dominated by unconditioned sensor noise. Under unconstrained thresholding, random noise spikes happened by chance "
        "to fall within the pre-event window, while generating intolerable false alarms during normal operation. This proved that early warning detectors cannot be evaluated without strictly measuring normal-operation false alarm rates."
    )
    add_image_p(target_p, ACCURACY_FIG, "Figure 5.1: Rolling 1-step-ahead forecasting accuracy: ARIMA and SARIMAX achieve 47% lower MAE than Naive persistence.")

    # -------------------------------------------------------------------------
    # 7. SECTION 6: PROPOSED MODEL AND ENHANCEMENT
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "6.2 Proposed Enhancement")
    add_formatted_p(
        target_p,
        "While univariate ARIMA captures pressure autoregression, it treats downhole fluid dynamics as an isolated acoustic system. In multiphase wellbores, downhole pressure and temperature "
        "are coupled through thermodynamic flow equations. When a safety valve chokes, mass flow rate decreases, inducing Joule-Thomson expansion and changing heat dissipation into the surrounding formation. "
        "However, downhole fluid possesses massive thermal inertia: acoustic pressure drops in seconds at c ≈ 1,000 m/s, whereas temperature drifts gradually over minutes. "
        "A successful model must condition pressure on dynamic temperature without letting delayed thermal noise dilute rapid pressure departures."
    )

    target_p = find_p(doc, "6.3 Methodology and Implementation")
    add_formatted_p(
        target_p,
        "We propose two model enhancements:\n"
        "1. Dynamic Regression SARIMAX(1, 1, 1): Downhole temperature (T-TPT) is integrated directly as an exogenous dynamic covariate:\n"
        "    (1 - phi_1 B)(1 - B) P_t = beta (1 - B) T_{t-1} + (1 + theta_1 B) epsilon_t\n"
        "Conditioning on lag-1 temperature provides dynamic thermodynamic context inside the state equation before residuals are computed.\n"
        "2. OILGUARD-AC (Alarm-Constrained Multisensor Fusion): Standardized pressure forecast residuals z_P = |r_P| / sigma_P and temperature departures z_T = |T - mu_T| / sigma_T "
        "are combined into an elliptical composite score:\n"
        "    S(t) = sqrt(w * z_P^2 + (1 - w) * z_T^2)\n"
        "where w in [0, 1] represents the pressure sensor weighting."
    )

    target_p = find_p(doc, "6.4 Expected Advantage")
    add_formatted_p(
        target_p,
        "Implementation and Calibration Framework:\n"
        "• Operational False-Alarm Budget (B_FAR): On historical calibration splits (H_calib hours), an allowable false alarm budget is enforced (B_FAR <= 1.0 alarms/hour), allowing at most K = floor(B_FAR * H_calib) false alarms.\n"
        "• Calibrated Quantile Thresholds (tau): The threshold tau is selected such that exactly (1 - K/N_calib) quantile of calibration scores falls below tau.\n"
        "• Persistence Constraint (p): An early warning alarm is latched if and only if p consecutive samples exceed threshold tau (p in {2, 3}), eliminating momentary acoustic noise spikes.\n"
        "• Operational Early Warning Window: Alarms are scored as valid early warnings only if triggered within [T_transient - 10 min, T_transient)."
    )

    target_p = find_p(doc, "6.5 Proposed Model Results")
    add_formatted_p(
        target_p,
        "The expected advantage of SARIMAX over post-hoc algebraic fusion lies in physical conditioning. In SARIMAX, temperature is fed directly into the autoregressive equation, "
        "allowing the model to difference out thermal baseline shifts. In contrast, heuristic post-hoc fusion (OILGUARD-AC with w=0.7) can dilute detection because sluggish temperature departures "
        "lag behind fast-moving pressure collapses, pulling down the composite score during incipient throttling. Dynamic regression inside the state equation is physically and statistically superior."
    )

    target_p = find_p(doc, "PERFORMANCE EVALUATION AND COMPARISON")
    add_formatted_p(
        target_p,
        "Empirical evaluation of the proposed models across the 19 gold events yields definitive results:\n"
        "• Forecasting Accuracy: SARIMAX achieves MAE = 0.0162 MPa, RMSE = 0.0523 MPa, MAPE = 8.63%, and R^2 = 0.9994 (slight edge over univariate ARIMA).\n"
        "• Protocol 1 (Within-Event Split): Heavy-pressure OILGUARD-AC (w=0.5) detects 68.4% of events (13/19) with 4.2 minutes lead time at 0.75 FA/hr; SARIMAX detects 36.8% at 0.25 FA/hr.\n"
        "• Protocol 2 (Cross-Event LOEO-CV, 19 Folds): SARIMAX and OILGUARD-AC (w=0.85) both achieve 63.2% detection (12/19 events, 95% CI [42.1%, 84.2%]) with 4.8 to 5.3 minutes of advance lead time at an ultra-low realized FAR of 0.50 FA/hr.\n"
        "• Protocol 3 (Cross-Well LOWO-CV, 5 Physical Wells): When evaluated across entirely unseen physical well completions, SARIMAX and OILGUARD-AC (w=0.85) achieve 73.7% detection (14/19 events) with >= 5.2 minutes lead time, proving genuine cross-asset generalizability."
    )

    # -------------------------------------------------------------------------
    # 8. SECTION 7: PERFORMANCE EVALUATION AND COMPARISON
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "7.2 Comparative Results")
    add_formatted_p(
        target_p,
        "To provide a rigorous, multi-dimensional assessment, models are evaluated across both forecasting accuracy and operational early-warning metrics:\n"
        "1. Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE): Measure absolute deviation between predicted and actual downhole pressure in Megapascals (MPa).\n"
        "2. Mean Absolute Percentage Error (MAPE): Measures relative forecasting error as a percentage of actual pressure.\n"
        "3. Detection Rate (%): The percentage of authentic shut-in events correctly alerted within the prospective 10-minute early warning window prior to transient onset.\n"
        "4. Advance Lead Time (Mean & Median in minutes): Time interval between alarm trigger and transient onset (T_transient - T_alarm) with 1,000-iteration bootstrap 95% confidence intervals.\n"
        "5. Realized False Alarm Rate (FAR in FA/hour): Count of false alarms per normal operating hour across holdout evaluation partitions."
    )

    target_p = find_p(doc, "7.3 Interpretation of Results")
    add_formatted_p(target_p, "Table 7.1: Audited Benchmark Performance Comparison at Matched Operational False-Alarm Rates (N=19 Gold Events across 5 Physical Wells)", bold=True)
    
    # Insert Table 7.1 right before target_p
    tbl_bench = doc.add_table(rows=8, cols=6)
    target_p._p.addprevious(tbl_bench._tbl)
    
    bench_headers = ["Model / Method", "P1 Within-Event Det % (Lead)", "P2 LOEO Det % [95% CI]", "P2 Mean Lead", "P2 Realized FAR", "Budget Met (<=1.0/hr)?"]
    for c_i, h in enumerate(bench_headers):
        cell = tbl_bench.rows[0].cells[c_i]
        cell.text = h
        set_cell_background(cell, "0A3663")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p_c = cell.paragraphs[0]
        p_c.runs[0].font.name = "Times New Roman"
        p_c.runs[0].font.size = Pt(9.5)
        p_c.runs[0].font.bold = True
        p_c.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    bench_rows = [
        ("SARIMAX (exog T)", "36.8% (4.0 min)", "63.2% [42.1%, 84.2%]", "4.83 min", "0.50 / hr (2 FA)", "YES (Compliant)"),
        ("OILGUARD-AC (w=0.85)", "47.4% (3.3 min)", "63.2% [42.1%, 84.2%]", "5.31 min", "0.50 / hr (2 FA)", "YES (Compliant)"),
        ("OILGUARD-AC (w=0.50)", "68.4% (4.2 min)", "63.2% [42.1%, 84.2%]", "4.01 min", "0.50 / hr (2 FA)", "YES (Compliant)"),
        ("Univariate ARIMA", "31.6% (2.5 min)", "57.9% [36.8%, 78.9%]", "5.18 min", "0.50 / hr (2 FA)", "YES (Compliant)"),
        ("OILGUARD-AC (w=0.70)", "57.9% (2.8 min)", "52.6% [31.6%, 73.7%]", "4.67 min", "0.75 / hr (3 FA)", "YES (Compliant)"),
        ("Naive Baseline", "47.4% (4.5 min)", "47.4% [26.3%, 68.4%]", "2.31 min", "1.01 / hr (4 FA)", "Borderline"),
        ("Local Outlier Factor (LOF)*", "57.9% (4.6 min)", "57.9% [36.8%, 78.9%]", "4.54 min", "1.51 / hr (6 FA)", "NO (Violates Budget!)")
    ]
    for r_i, r_data in enumerate(bench_rows, start=1):
        row = tbl_bench.rows[r_i]
        bg = "F8FAFC" if r_i % 2 == 1 else "FFFFFF"
        for c_i, val in enumerate(r_data):
            cell = row.cells[c_i]
            cell.text = val
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p_c = cell.paragraphs[0]
            p_c.runs[0].font.name = "Times New Roman"
            p_c.runs[0].font.size = Pt(9.5)
            if c_i == 0:
                p_c.runs[0].font.bold = True
            if "YES" in val:
                p_c.runs[0].font.color.rgb = RGBColor(0x05, 0x96, 0x69)
            elif "NO" in val:
                p_c.runs[0].font.color.rgb = RGBColor(0xDC, 0x26, 0x26)

    add_image_p(target_p, FIGURES_DIR / "protocol_comparison.png", "Figure 7.1: Cross-protocol early warning performance across Protocol 1, Protocol 2 (LOEO), and Protocol 3 (LOWO).")

    target_p = find_p(doc, "CONCLUSION AND FUTURE SCOPE")
    add_formatted_p(
        target_p,
        "Critical Findings and Scientific Insights:\n"
        "1. Dynamic Conditioning Outperforms Post-Hoc Blending: SARIMAX matches or exceeds the best composite score (OILGUARD-AC w=0.5 at 63.2% LOEO detection, 4.8 min lead time) "
        "because feeding lag-1 temperature directly into the autoregressive equation respects downhole thermal transport inertia. Heuristic post-hoc fusion with w=0.7 suffers reduced detection (52.6%) "
        "because lagging temperature deviations dilute the sharp acoustic pressure drop during incipient flutter.\n"
        "2. Why Local Outlier Factor (LOF) Fails Operational Deployment: In prior literature (Fernandes et al., 2024), LOF was cited as top-performing based on retrospective windowed F1-scores (~87%). "
        "However, when evaluated under operational prospective constraints, LOF generated 1.51 to 2.01 false alarms per normal hour (6 false alarms over 3.98 holdout hours), exceeding the operational budget by 1.5x to 2x. "
        "The root cause is cross-well density drift: normal operating baseline pressure varies from 8.9 to 21.0 MPa across the 5 physical wells. LOF calculates k-NN distance in raw feature space, causing normal points on unseen wells "
        "to land in sparse density clusters relative to training wells. Time series forecasting differences out static level shifts, maintaining robustness across distinct assets.\n"
        "3. Calibration FAR Does Not Equal Realized Holdout FAR: Short calibration windows (10-30 min) cannot capture multi-hour turbulent variance. Enforcing zero false alarms during calibration does not guarantee budget compliance on holdout evaluation."
    )

    # -------------------------------------------------------------------------
    # 9. SECTION 8: CONCLUSION AND FUTURE SCOPE
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "8.2 Suggest possible improvements")
    add_formatted_p(
        target_p,
        "This research establishes that prospective statistical time series forecasting (SARIMAX and ARIMA) provides viable, leak-free early warnings of spurious Downhole Safety Valve (DHSV) closures "
        "in offshore oil wells. Across an audited gold cohort of 19 authentic events from 5 physical wells in the Petrobras 3W benchmark, the proposed models reduce 1-step forecast error by 47.4% over persistence "
        "and deliver 63.2% to 73.7% detection with an average advance lead time of 4.8 minutes while strictly respecting operational false-alarm constraints (<= 0.50 alarms/hour). "
        "The investigation conclusively disproves the viability of raw coordinate density models (LOF) for multi-well offshore deployment and demonstrates that dynamic regression inside the state equation "
        "is scientifically superior to post-hoc algebraic residual blending."
    )

    target_p = find_p(doc, "8.3 Brief on the SDG")
    add_formatted_p(
        target_p,
        "Future research and engineering extensions include:\n"
        "1. Multiphase Hydrodynamic Simulation: Coupling time-series forecast residuals with high-fidelity transient multiphase simulators (e.g., OLGA) to model the complete acoustic wave propagation along 2,000m tubing strings.\n"
        "2. Closed-Loop Supervisory Choke Maneuvering: Developing automated supervisory control logic that initiates an engineered surface choke ramp-down upon confirmed early warning latch, mitigating water-hammer shockwaves.\n"
        "3. Physics-Guided Neural State-Space Models: Exploring lightweight neural state-space models (e.g., Kalman-coupled recurrent units) that maintain physical differencing while scaling across complex multi-well offshore fields."
    )

    target_p = find_p(doc, "REFERENCES")
    add_formatted_p(
        target_p,
        "The OILGUARD project directly contributes to three United Nations Sustainable Development Goals (SDGs):\n"
        "• UN SDG 14 (Life Below Water): Unmanaged DHSV closures risk severe subsea tubing rupture, flapper cavitation, and catastrophic downhole barrier breaches. "
        "Providing 4 to 5 minutes of early warning enables orderly shutdown, directly protecting vulnerable deepwater marine biodiversity and coastal fishing ecosystems from catastrophic oil spills.\n"
        "• UN SDG 9 (Industry, Innovation, and Infrastructure): Upgrades offshore SCADA monitoring from reactive static trip alarms to resilient, cyber-physical early warning systems powered by statistical forecasting.\n"
        "• UN SDG 12 (Responsible Consumption and Production): Minimizes emergency platform flaring, prevents irreversible tubing deformation, and avoids costly, carbon-intensive subsea well workover interventions."
    )

    # -------------------------------------------------------------------------
    # 10. SECTION 9: REFERENCES
    # -------------------------------------------------------------------------
    target_p = find_p(doc, "TEAM CONTRIBUTION")
    refs = [
        "1. Vargas, R. E. V., et al. (2019). A realistic and public dataset with rare undesirable real events in oil wells. Journal of Petroleum Science and Engineering, 181, 106223. https://doi.org/10.1016/j.petrol.2019.106223",
        "2. Vargas, R. E. V., et al. (2026). 3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells. Nature Scientific Data, 13, 949. https://doi.org/10.1038/s41597-026-07225-z",
        "3. Fernandes Jr., W., Komati, K. S., & Gazolli, K. (2024). Anomaly detection in oil-producing wells: a comparative study of one-class classifiers in a multivariate time series dataset. Journal of Petroleum Exploration and Production Technology, 14, 523–542. https://doi.org/10.1007/s13202-023-01710-6",
        "4. Wang, Y., et al. (2024). Graph Structure Change-Based Anomaly Detection in Multivariate Time Series of Industrial Processes. IEEE Transactions on Industrial Informatics, 20(6), 8440–8450. https://doi.org/10.1109/TII.2023.3347000",
        "5. Khan, A., Al Farid, F., Uddin, J., et al. (2026). Early-warning industrial fault detection based on physics-guided residual learning and calibrated CRNNs. Scientific Reports, 16, 8812. https://doi.org/10.1038/s41598-026-48227-6",
        "6. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). John Wiley & Sons.",
        "7. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts: Melbourne, Australia."
    ]
    for ref in refs:
        p_ref = target_p.insert_paragraph_before()
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = "Times New Roman"
        r_ref.font.size = Pt(10)

    # -------------------------------------------------------------------------
    # 11. SECTION 10: TEAM CONTRIBUTION
    # -------------------------------------------------------------------------
    last_p = doc.paragraphs[-1]
    team_contribs = [
        ("1. Ravva Nagarjun (Lead Author & Submitter): ", "Conceived the forecasting-based early warning research problem, designed the overall system architecture, conducted the forensic data audit of the Petrobras 3W Class 2 cohort, formulated and trained the ARIMA and SARIMAX time-series models, executed the leak-free cross-validation protocols (LOEO-CV and LOWO-CV), authored the formal technical report, and developed the interactive Streamlit intelligence platform."),
        ("2. Naveen Kumar N (Co-Author & Contributor): ", "Conducted the comprehensive literature review across the five foundational academic research papers, implemented the data ingestion and uniform resampling pipeline, conducted exploratory data analysis and stationarity testing (ADF and KPSS), developed evaluation metrics (MAE, RMSE, MAPE, and advance lead time quantification), and constructed the comparative benchmarking visualizations.")
    ]
    for mem, task in team_contribs:
        add_bullet_p(last_p, mem, task)

    # -------------------------------------------------------------------------
    # 12. TABLE OF CONTENTS (Inserted at the very end before ABSTRACT)
    # -------------------------------------------------------------------------
    target_abs = find_p(doc, "ABSTRACT")
    toc_items = [
        ("1. ABSTRACT", "Page 2"),
        ("2. INTRODUCTION AND PROBLEM IDENTIFICATION", "Page 3"),
        ("    2.1 Application Domain", "Page 3"),
        ("    2.2 Problem Statement", "Page 3"),
        ("    2.3 Need for Forecasting", "Page 4"),
        ("    2.4 Objectives", "Page 4"),
        ("    2.5 Scope of the Study", "Page 4"),
        ("3. LITERATURE REVIEW AND RESEARCH GAP", "Page 5"),
        ("    3.1 Review of Existing Research (5 Foundational Papers)", "Page 5"),
        ("    3.2 Research Gap", "Page 6"),
        ("    3.3 Proposed Approach", "Page 6"),
        ("4. DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS", "Page 7"),
        ("    4.1 Dataset Source", "Page 7"),
        ("    4.2 Variables and Data Description", "Page 7"),
        ("    4.3 Data Preparation & Screening", "Page 7"),
        ("    4.4 Exploratory Data Analysis & Stationarity Testing", "Page 8"),
        ("5. BASELINE FORECASTING MODEL", "Page 9"),
        ("    5.1 Model Selection", "Page 9"),
        ("    5.2 Model Methodology & Mathematical Formulation", "Page 9"),
        ("    5.3 Implementation", "Page 10"),
        ("    5.4 Baseline Results & The Noisy Baseline Paradox", "Page 10"),
        ("6. PROPOSED MODEL AND ENHANCEMENT", "Page 11"),
        ("    6.1 Motivation for Improvement", "Page 11"),
        ("    6.2 Proposed Enhancement (SARIMAX & OILGUARD-AC)", "Page 11"),
        ("    6.3 Methodology, Formulation and Implementation", "Page 12"),
        ("    6.4 Expected Advantage", "Page 12"),
        ("    6.5 Proposed Model Results", "Page 13"),
        ("7. PERFORMANCE EVALUATION AND COMPARISON", "Page 14"),
        ("    7.1 Evaluation Measures", "Page 14"),
        ("    7.2 Comparative Results (Matched Realized FAR Benchmarking)", "Page 14"),
        ("    7.3 Interpretation of Results & Why LOF Fails", "Page 15"),
        ("8. CONCLUSION AND FUTURE SCOPE", "Page 16"),
        ("    8.1 Summary of Forecasting Problem & Effectiveness", "Page 16"),
        ("    8.2 Practical Implications & Limitations", "Page 16"),
        ("    8.3 Sustainable Development Goals (SDGs) Addressed", "Page 16"),
        ("9. REFERENCES", "Page 17"),
        ("10. TEAM CONTRIBUTION", "Page 17"),
    ]
    for title, pg in toc_items:
        p_toc = target_abs.insert_paragraph_before()
        p_toc.paragraph_format.line_spacing = 1.15
        p_toc.paragraph_format.space_after = Pt(2)
        r_t = p_toc.add_run(f"{title:<70} {pg}")
        r_t.font.name = "Times New Roman"
        r_t.font.size = Pt(11)

    target_abs.insert_paragraph_before().paragraph_format.space_after = Pt(12)

    # Save filled template
    saved_successfully = False
    try:
        doc.save(str(TARGET_DOCX))
        print(f"Successfully populated official template: {TARGET_DOCX}")
        saved_successfully = True
    except PermissionError:
        print(f"Notice: {TARGET_DOCX.name} is currently open in Word.")

    # Also save to updated copy
    updated_path = REPORT_DIR / "TSA AAT REPORT Template - Lab - Updated.docx"
    doc.save(str(updated_path))
    print(f"Saved updated file to: {updated_path}")
    if not saved_successfully:
        print(f"To update the main file directly, close Word and re-run, or use the updated file.")


if __name__ == "__main__":
    main()
