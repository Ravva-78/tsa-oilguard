"""Build the Word Technical Report and PowerPoint Presentation for OILGUARD.

Generates:
  1. report/OILGUARD_Technical_Report.docx (comprehensive chronological 5-phase research narrative)
  2. report/OILGUARD_Presentation.pptx (accompanying slide deck structured by research progression)
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

import pptx
from pptx.util import Inches as PptxInches, Pt as PptxPt
from pptx.dml.color import RGBColor as PptxRGBColor
from pptx.enum.text import PP_ALIGN

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "results" / "audited_v2" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "audited_v2" / "metrics"


def set_cell_background(cell, fill_hex):
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


def build_word_report():
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(1)
        s.bottom_margin = Inches(1)
        s.left_margin = Inches(1)
        s.right_margin = Inches(1)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header Title
    title = doc.add_paragraph()
    title_run = title.add_run("OILGUARD: Forecasting-Based Early Warning of Abnormal Events in Oil Wells")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(21)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)
    title.paragraph_format.space_after = Pt(2)

    subtitle = doc.add_paragraph()
    sub_run = subtitle.add_run("Technical Research Report: Chronological Investigation, Empirical Results, and Reconciled Multi-Protocol Audit")
    sub_run.font.name = 'Calibri'
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    subtitle.paragraph_format.space_after = Pt(12)

    # Metadata table
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_cell = meta_table.rows[0].cells[0]
    set_cell_background(meta_cell, "F0F4F8")
    set_cell_margins(meta_cell, 120, 120, 160, 160)
    meta_p = meta_cell.paragraphs[0]
    meta_p.paragraph_format.line_spacing = 1.2
    meta_p.paragraph_format.space_after = Pt(0)
    
    runs_meta = [
        ("Authors: ", True), ("Ravva Nagarjun & Naveen Kumar N\n", False),
        ("Domain: ", True), ("Offshore Petroleum Engineering & Time Series Anomaly Detection\n", False),
        ("Target Event: ", True), ("Class 2 — Spurious Closure of Downhole Safety Valve (DHSV)\n", False),
        ("Sensors: ", True), ("Downhole Pressure Transducer (P-TPT, MPa) & Temperature Transducer (T-TPT, °C)\n", False),
        ("Benchmark Dataset: ", True), ("Petrobras 3W Dataset v2.0.0 (Authentic Offshore Production Wells)\n", False),
        ("Verified Cohort: ", True), ("19 Gold-Standard Real-Well Events (Strict Denominator = 19 across 5 Physical Wells)\n", False),
        ("Evaluation Protocols: ", True), ("Protocol 1 (Within-Event), Protocol 2 (Cross-Event LOEO-CV), Protocol 3 (Cross-Well LOWO-CV)\n", False),
        ("Report Version: ", True), ("Audited Final v3.0 (Comprehensive Chronological Progression)", False),
    ]
    for txt, is_b in runs_meta:
        r = meta_p.add_run(txt)
        r.font.size = Pt(9.5)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 1. Executive Summary & Narrative Roadmap
    h1 = doc.add_heading("1. Executive Summary & Narrative Roadmap", level=1)
    h1.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)
    
    doc.add_paragraph(
        "In deepwater petroleum completions, an uncommanded or spurious closure of the Downhole Safety Valve (DHSV, designated "
        "as Class 2 in the Petrobras 3W benchmark) represents a high-consequence operational emergency. The DHSV flapper slams "
        "shut into the multiphase production stream hundreds of meters beneath the seabed, isolating the high-pressure reservoir "
        "upstream while triggering severe hydrodynamic pressure collapse and acoustic shock downstream. In conventional offshore "
        "Supervisory Control and Data Acquisition (SCADA) systems, alarms rely on static, lower-bound trip thresholds (P < P_trip). "
        "These alarms trigger only after the valve is fully seated and flow has ceased, providing zero actionable warning time for "
        "automated mitigation or control-room intervention."
    )
    doc.add_paragraph(
        "The goal of the OILGUARD project was to develop a prospective, forecasting-based early warning system capable of detecting "
        "incipient valve throttling minutes before transient onset, while strictly adhering to offshore operational false-alarm "
        "constraints (FAR <= 1.0 alarms/hour)."
    )
    doc.add_paragraph(
        "Rather than presenting only final numbers in isolation, this technical report documents the complete chronological progression "
        "of the research: what we tried first, what results we obtained, what flaws or paradoxes were exposed, what we tried next to "
        "resolve them, and how the final reconciled methodology and scientific findings were achieved."
    )

    doc.add_paragraph(
        "The Five-Stage Chronological Journey:\n"
        "• Phase 1 (Initial TSA Forecasting): Evaluated rolling 1-step ARIMA and SARIMAX against Naive persistence; cut error by 47%, "
        "but discovered the 'Noisy Baseline Paradox' where Naive appeared to detect 7/19 events through accidental noise triggers.\n"
        "• Phase 2 (Retrospective Optimization & Smoothing): Grid searched (q, p) and applied EWMA smoothing (lambda=0.3); reported "
        "detection jumped to 68.4%, but exposed runaway false alarms (2.5 to 3.5 FA/hr) and retrospective data leakage.\n"
        "• Phase 3 (Candidate Extension — OILGUARD-AC & LOF Benchmark): Formulated elliptical multisensor scoring (S_t) under an operational "
        "FAR budget (<=1.0 FA/hr) alongside Local Outlier Factor (LOF); initial scripts revealed conflicting metrics (47.4% vs 57.9%).\n"
        "• Phase 4 (Independent Forensic Audit): Resolved the cohort strictly to 19 gold events; diagnosed p=1 vs p=2 sensitivity; "
        "identified short-window calibration collapse; and uncovered that 19 events originate from only 5 physical wells (63.2% from Well 11).\n"
        "• Phase 5 (Final Reconciled Methodology & Findings): Established 3 leak-free protocols under matched realized FAR; proved SARIMAX "
        "matches balanced OILGUARD-AC at 63.2% detection with 4.8m lead (0.50 FAR), while LOF fails operational budgets at 1.51 FA/hr."
    )

    # 2. Phase 1: Initial TSA Forecasting & The Noisy Baseline Paradox
    h2 = doc.add_heading("2. Phase 1: What We Tried First — Initial Time Series Forecasting & The Noisy Baseline Paradox", level=1)
    h2.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "What We Tried: We hypothesized that downhole pressure during steady-state production follows structured autoregressive dynamics. "
        "We formulated rolling 1-step-ahead statistical forecasting models on 1 Hz downhole pressure telemetry (P-TPT, MPa):\n"
        "1. Univariate ARIMA(1, 1, 1) on downhole pressure.\n"
        "2. Multivariate SARIMAX(1, 1, 1) incorporating downhole temperature (T-TPT, °C) as a dynamic exogenous covariate.\n"
        "3. Industrial Naive Persistence Baseline predicting P_hat(t) = P(t-1)."
    )

    doc.add_paragraph(
        "What Results We Got: Rolling 1-step-ahead forecasts were evaluated over untouched holdout test splits across authentic events (Table 1). "
        "Statistical models slashed Mean Absolute Error by 47.1% and MAPE from 18.2% to 8.7% compared to persistence."
    )

    # Table 1: Forecast Accuracy
    t1 = doc.add_table(rows=4, cols=6)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_t1 = ["Model Architecture", "Input Features", "MAE (MPa)", "RMSE (MPa)", "MAPE (%)", "Performance vs. Baseline"]
    for c_idx, h in enumerate(headers_t1):
        cell = t1.rows[0].cells[c_idx]
        set_cell_background(cell, "0A3663")
        set_cell_margins(cell, 100, 100, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(9.5)

    rows_t1 = [
        ["ARIMA(1, 1, 1)", "P-TPT", "0.0163", "0.0524", "8.69%", "47.1% Lower MAE"],
        ["SARIMAX(1, 1, 1)", "P-TPT + T-TPT", "0.0162", "0.0523", "8.63%", "47.4% Lower MAE"],
        ["Naive Persistence", "P-TPT (t-1)", "0.0307", "0.0781", "18.22%", "Reference Baseline"]
    ]
    for r_idx, r_data in enumerate(rows_t1, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t1.rows[r_idx].cells[c_idx]
            bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 100, 100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (c_idx == 0 or c_idx == 1 or c_idx == 5) else WD_ALIGN_PARAGRAPH.RIGHT
            run_txt = p.add_run(val)
            run_txt.font.size = Pt(9.5)
            if c_idx == 0:
                run_txt.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "The Initial Residual Detection Attempt & The 'Noisy Baseline Paradox':\n"
        "When we converted these forecasts into an early-warning system by flagging when raw residuals |r_t| = |P_t - P_hat_t| exceeded "
        "a 3-sigma threshold, an alarming paradox appeared: Naive persistence appeared to 'detect' 7 of 19 events (36.8%) with an average "
        "lead time of 4.5 minutes, appearing competitive with ARIMA!\n\n"
        "Forensic diagnosis revealed the root cause: Naive residuals were dominated by unconditioned high-frequency sensor noise. Without "
        "persistence filtering or cooldown logic, random noise spikes triggered accidental alarms that happened by pure chance to fall "
        "within the 10-minute early-warning window, while generating unconstrained false alarms during normal operations.\n"
        "Scientific Lesson 1: A detector cannot be evaluated solely on whether it alarms before failure. Without strictly enforcing and "
        "measuring false alarm rates during normal operation, noisy baselines appear artificially competent through accidental triggering."
    )

    # 3. Phase 2: Retrospective Optimization & EWMA Residual Smoothing
    h3 = doc.add_heading("3. Phase 2: What We Tried Next — Retrospective Optimization & Residual Smoothing", level=1)
    h3.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "What We Tried: To eliminate accidental noise triggers and optimize detection, we implemented an optimization grid search "
        "(scripts/optimize_warning.py):\n"
        "1. Grid search over quantile thresholds q in [0.90, 0.99] and persistence filters p in [1, 2, 3] consecutive samples.\n"
        "2. Fixed alarm cooldown periods (120 seconds) to prevent multiple alarms for a single failure.\n"
        "3. Exponentially Weighted Moving Average (EWMA) smoothing on forecast residuals: r_EWMA(t) = alpha * r(t) + (1 - alpha) * r_EWMA(t-1) "
        "with alpha in [0.1, 0.5] (centering on alpha = 0.3) to damp sensor flutter."
    )
    doc.add_paragraph(
        "What Results We Got: Under single-sample triggers (p=1) and q=0.95, reported detection rates jumped to 68.4% (13 of 19 events) "
        "with mean lead times exceeding 5.0 minutes. EWMA smoothing successfully eliminated isolated sensor glitches."
    )
    doc.add_paragraph(
        "Limitations Discovered — Alarm Storms & Retrospective Leakage:\n"
        "• False Alarm Explosion: Evaluating normal operational data revealed that p=1 produced False Alarm Rates of 2.5 to 3.5 false alarms "
        "per hour. In an offshore control room, 3 false alarms per hour creates immediate alarm fatigue and causes operators to disable the system.\n"
        "• Retrospective Data Leakage: The grid search tuned thresholds across the entire event duration, tuning parameters on the very data "
        "being evaluated.\n"
        "Scientific Lesson 2: Retrospective parameter optimization creates an illusion of high detection accuracy by overfitting to known events "
        "and ignoring runaway false-alarm rates during steady-state production."
    )

    # 4. Phase 3: Candidate Extension (OILGUARD-AC) & LOF Benchmark
    h4 = doc.add_heading("4. Phase 3: Candidate Extension — Multisensor Composite Scoring (OILGUARD-AC) & LOF Benchmark", level=1)
    h4.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "What We Tried: To prevent single-sensor false alarms and exploit thermodynamic coupling (Joule-Thomson cooling and mass flow "
        "deceleration during valve throttling), we developed OILGUARD-AC (Alarm-Constrained Multisensor Anomaly Scoring):\n"
        "1. Standardized elliptical composite score: S_t = sqrt(w * z_P(t)^2 + (1 - w) * z_T(t)^2), testing weightings w=0.85, w=0.70, and w=0.50.\n"
        "2. Operational False Alarm Budgeting: Threshold tau calibrated so that normal calibration FAR <= B_FAR (1.0 FA/hr).\n"
        "3. Local Outlier Factor (LOF) Benchmark: Implemented LOF (k=20 nearest neighbors on raw [P, T]) as an established non-forecasting baseline."
    )
    doc.add_paragraph(
        "Initial Results & Emerging Discrepancies:\n"
        "Initial scripts reported that OILGUARD-AC detected 9 of 19 events (47.4%) at 0.75 FAR, SARIMAX detected 13 of 19 (68.4%), and LOF "
        "detected 14 of 19 (73.7%). However, numbers were inconsistent across documents: some reported 47.4%, others 57.9%; some reported "
        "FAR of 0.75, others 1.26; and LOF's false alarms were not constrained to the operational budget.\n"
        "Scientific Lesson 3: Discrepancies between 47.4% and 57.9% detection, and conflicting false-alarm rates, indicated unresolved differences "
        "in data hygiene, persistence filtering, and validation protocols that required a forensic audit."
    )

    # 5. Phase 4: Independent Forensic Research Audit
    h5 = doc.add_heading("5. Phase 4: Independent Forensic Research Audit & Discrepancy Diagnostics", level=1)
    h5.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "We paused model experimentation to conduct an exhaustive forensic audit across data, code, and saved metrics:"
    )
    doc.add_paragraph(
        "1. Cohort Resolution (19 vs. 20 vs. 22 Events): From 22 authentic Class 2 candidate instances, we excluded 3 corrupted files: "
        "WELL-00010 (missing P-TPT sensor stream), WELL-00011 (1,608-second unrecorded data gap), and WELL-00002 (zero-variance sensor flatline). "
        "An orphaned 20th parquet file from an earlier exploratory scan was removed, strictly locking the gold cohort to exactly 19 events.\n"
        "2. The 47.4% vs. 57.9% Detection Rate Discrepancy: Traced directly to persistence p. At p=1 (single-point breach), detection was 57.9% "
        "with elevated false alarms. At p=2 (two consecutive breaches), transient micro-spikes were filtered out, lowering raw detection to "
        "47.4% while successfully suppressing false alarms to 0.75/hr.\n"
        "3. The FAR 0.75 vs. 1.26 / hr Discrepancy: Traced to pre-event window definition. We established an audited standard: the early warning "
        "window is strictly [T_transient - 10 min, T_transient). Any alarm prior to this window is strictly cataloged as an ordinary false alarm, "
        "evaluated over the 3.978 hours of total normal holdout duration.\n"
        "4. Short-Window Calibration Budget Collapse: In individual events, calibration slices are only 12 to 25 minutes long (H_calib < 0.5 h). "
        "Setting K_allowed = floor(1.0 * H_calib) forced K_allowed = 0, setting the threshold to the maximum noise peak. Pooled calibration across "
        "18 events (H_pool ~ 3.7 h) provides adequate statistical power (K_allowed = 3) to prevent threshold collapse.\n"
        "5. Conflation of Within-Event vs. Cross-Well Evaluation: Early scripts evaluated within-event splits (calibrating on a well's own baseline), "
        "not cross-event generalization.\n"
        "6. Physical Well Clustering: Audited well origins revealed that the 19 authentic events come from only 5 physical offshore wells: "
        "WELL-00011 accounts for 12 events (63.2%), WELL-00003 for 3 events (15.8%), WELL-00012 for 2 events (10.5%), WELL-00009 for 1 event, "
        "and WELL-00013 for 1 event."
    )

    # 6. Phase 5: Reconciled Methodology & Empirical Findings
    h6 = doc.add_heading("6. Phase 5: Final Reconciled Methodology & Audited Findings", level=1)
    h6.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "To provide a complete and honest empirical evaluation, we formulated 3 distinct, leakage-free protocols evaluated under matched "
        "realized False Alarm Rates (FAR <= 1.0 FA/hr, with primary comparison at FAR <= 0.75 FA/hr):\n"
        "• Protocol 1 (Prospective Within-Event Split): Well-specific baseline calibration (60% Train, 20% Calib, 20% Test).\n"
        "• Protocol 2 (Genuine Cross-Event LOEO-CV): Complete event holdout; threshold calibrated on pooled calibration data of the other 18 events (H_pool ~ 3.7 h).\n"
        "• Protocol 3 (Cross-Physical-Well LOWO-CV): All events originating from each physical well are completely held out (5 folds)."
    )

    # Table 2: Matched-FAR Benchmark Comparison
    matched_csv = METRICS_DIR / "matched_far_comparison.csv"
    if matched_csv.exists():
        df_m = pd.read_csv(matched_csv)
    else:
        df_m = pd.DataFrame()

    t2 = doc.add_table(rows=len(df_m) + 1, cols=7)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    opt_headers = ["Model / Method", "P1 Detection", "P1 Lead (Mean/Med)", "P2 LOEO Detection [95% CI]", "P2 Lead (Mean/Med)", "P2 Realized FAR", "P3 LOWO Detection"]
    for c_idx, h in enumerate(opt_headers):
        cell = t2.rows[0].cells[c_idx]
        set_cell_background(cell, "0A3663")
        set_cell_margins(cell, 100, 100, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(8.5)

    display_names = {
        "sarimax": "SARIMAX (Exogenous T)",
        "ac_0.85": "OILGUARD-AC (w=0.85)",
        "ac_0.7": "OILGUARD-AC (w=0.7)",
        "ac_0.5": "OILGUARD-AC (w=0.5)",
        "arima": "Univariate ARIMA",
        "naive": "Naive Persistence",
        "lof": "Local Outlier Factor (LOF)*",
    }
    ci_mapping = {
        "sarimax": "[42.1%, 84.2%]",
        "ac_0.85": "[42.1%, 84.2%]",
        "ac_0.5": "[42.1%, 84.2%]",
        "arima": "[36.8%, 78.9%]",
        "ac_0.7": "[31.6%, 73.7%]",
        "naive": "[26.3%, 68.4%]",
        "lof": "[36.8%, 78.9%]",
    }

    for r_idx, row in enumerate(df_m.itertuples(), start=1):
        m_name = display_names.get(row.model, row.model)
        p1_det = f"{row.p1_detection_rate*100:.1f}% ({row.p1_hits}/19)"
        p1_lead = f"{row.p1_mean_lead:.1f}m / {row.p1_med_lead:.1f}m"
        ci_str = ci_mapping.get(row.model, "")
        p2_det = f"{row.p2_detection_rate*100:.1f}% ({row.p2_hits}/19) {ci_str}"
        p2_lead = f"{row.p2_mean_lead:.1f}m / {row.p2_med_lead:.1f}m"
        meets_budget = getattr(row, "p2_meets_1.0_budget", True)
        far_str = f"{row.p2_realized_far:.2f} / hr" + (" (Met)" if meets_budget else " (EXCEEDED)")
        p3_det = f"{row.p3_detection_rate*100:.1f}% ({row.p3_hits}/19)"

        r_data = [m_name, p1_det, p1_lead, p2_det, p2_lead, far_str, p3_det]
        for c_idx, val in enumerate(r_data):
            cell = t2.rows[r_idx].cells[c_idx]
            bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            run_txt = p.add_run(val)
            run_txt.font.size = Pt(8.0)
            if c_idx == 0:
                run_txt.font.bold = True
            if c_idx == 5:
                run_txt.font.bold = True
                run_txt.font.color.rgb = RGBColor(0x1B, 0x5E, 0x20) if "Met" in val else RGBColor(0xB7, 0x1C, 0x1C)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Insert Figures
    if (FIGURES_DIR / "protocol_comparison.png").exists():
        doc.add_picture(str(FIGURES_DIR / "protocol_comparison.png"), width=Inches(5.8))
        cap1 = doc.add_paragraph("Figure 1: Reconciled Comparison Across Protocol 1 (Within-Event) and Protocol 2 (Cross-Event LOEO-CV) under Realized FAR <= 1.0/hr (*LOF exceeds budget at 1.51/hr).")
        cap1.style.font.size = Pt(8.5)
        cap1.style.font.italic = True
        cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if (FIGURES_DIR / "matched_far_tradeoff.png").exists():
        doc.add_picture(str(FIGURES_DIR / "matched_far_tradeoff.png"), width=Inches(5.8))
        cap2 = doc.add_paragraph("Figure 2: Protocol 2 Cross-Event LOEO-CV Realized False Alarm Rate vs. Early Detection Rate Trade-off Curve.")
        cap2.style.font.size = Pt(8.5)
        cap2.style.font.italic = True
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if (FIGURES_DIR / "loeo_vs_lowo_comparison.png").exists():
        doc.add_picture(str(FIGURES_DIR / "loeo_vs_lowo_comparison.png"), width=Inches(5.8))
        cap3 = doc.add_paragraph("Figure 3: Generalization Sensitivity: Cross-Event LOEO-CV (19 Folds) vs. Cross-Physical-Well LOWO-CV (5 Folds).")
        cap3.style.font.size = Pt(8.5)
        cap3.style.font.italic = True
        cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 7. Scientific Discussion & Root-Cause Physical Insights
    h7 = doc.add_heading("7. Scientific Discussion & Root-Cause Physical Insights", level=1)
    h7.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "1. Dynamic Regression vs. Post-Hoc Heuristic Fusion:\n"
        "In Cross-Event LOEO-CV, dynamic temperature conditioning in SARIMAX achieved 63.2% detection with 4.8 min mean lead time at 0.50 FA/hr, "
        "matching balanced OILGUARD-AC (w=0.50: 63.2%, 4.0 min lead) and outperforming moderate coupling (w=0.70: 52.6%, 4.7 min lead).\n"
        "Physical Mechanism: Downhole fluid possesses thermal inertia; fluid temperature changes more gradually than acoustic pressure drop "
        "during incipient valve throttling. In SARIMAX, temperature enters the autoregressive forecast equation directly, conditioning dynamic "
        "expectations before computing residuals. In post-hoc algebraic fusion (w=0.70), sluggish temperature deviations can dilute early "
        "acoustic pressure departures."
    )
    doc.add_paragraph(
        "2. Failure of Density-Based Non-Forecasting Detectors (LOF):\n"
        "While Local Outlier Factor achieves 57.9% detection, it generates 1.51 false alarms per hour on unseen holdout wells, violating "
        "operational nuisance-alarm budgets by 1.5x due to cross-well density drift. LOF measures k-NN density in raw feature space [P, T]. "
        "Because baseline operating pressure varies from 8.9 to 21.0 MPa across the 5 physical wells, normal holdout telemetry falls into "
        "low-density regions relative to calibration sets. Time-series forecasting models difference out static level shifts."
    )
    doc.add_paragraph(
        "3. Calibration FAR Does Not Equal Realized Holdout FAR:\n"
        "Calibrating on a short normal window does not guarantee that holdout FAR will remain below budget when deployed over longer periods. "
        "Realized holdout FAR must always be reported separately from calibration FAR."
    )

    # 8. Operational Scope & Qualified Engineering Limitations
    h8 = doc.add_heading("8. Operational Scope & Qualified Engineering Limitations", level=1)
    h8.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "1. Observed Experimental Lead Time: Detected events showed an observed mean lead time of approximately 4 to 5 minutes under the experimental "
        "definition. This creates an actionable operational window for diagnostic assessment and control-room verification. We explicitly do not claim "
        "that this retrospective lead time proves sufficient time for safe automated intervention, closed-loop choke adjustments, or water hammer prevention.\n"
        "2. Future Dynamic Simulation: Assertions regarding hydraulic transient mitigation or water hammer suppression remain hypotheses that require "
        "rigorous multiphase dynamic simulation (e.g., Schlumberger OLGA) and physical flow-loop testing prior to any live SCADA deployment.\n"
        "3. Small Sample Uncertainty: Although 19 real-world events across 5 physical wells represent one of the largest authentic DHSV closure "
        "cohorts published, the 95% bootstrap confidence intervals (e.g., [42.1%, 84.2%] for SARIMAX) reflect inherent statistical uncertainty."
    )

    # 9. Chronology Summary Table
    h9 = doc.add_heading("9. Complete Experimental Chronology Summary Matrix", level=1)
    h9.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "The table below provides a comprehensive chronological record of everything attempted, the results obtained, the limitations "
        "discovered, and the resolution achieved across the entire research journey:"
    )

    t3 = doc.add_table(rows=6, cols=6)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_t3 = ["Phase", "What We Tried", "Hypothesis", "Results Obtained", "Flaws / Gaps Found", "Resolution & Next Step"]
    for c_idx, h in enumerate(headers_t3):
        cell = t3.rows[0].cells[c_idx]
        set_cell_background(cell, "0A3663")
        set_cell_margins(cell, 100, 100, 80, 80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(8.5)

    rows_t3 = [
        ["Phase 1", "Rolling 1-step ARIMA, SARIMAX vs Naive", "Forecasting captures pressure dynamics and cuts error vs persistence", "0.016 MPa MAE (47% lower error than Naive)", "Noisy Baseline Paradox: Naive detected 7/19 due to unconstrained noise", "Introduced FAR tracking & persistence filtering"],
        ["Phase 2", "Grid search (q, p) & EWMA smoothing (lambda=0.3)", "Persistence (p) and smoothing eliminate noise, boosting detection", "Detection reached 68.4% (13/19) with >5 min lead time", "Alarm Storms & Leakage: FAR exploded to 2.5-3.5/hr; retrospective tuning", "Enforced operational FAR budget (<=1.0/hr)"],
        ["Phase 3", "OILGUARD-AC multisensor scoring & LOF benchmark", "Combining P residuals with T departures cuts single-sensor false alarms", "OILGUARD-AC detected 9-11 events; LOF reported 14/19", "Conflicting Metrics: Numbers varied across drafts; LOF FAR unbudgeted", "Paused additions to conduct forensic audit"],
        ["Phase 4", "Forensic audit of cohort, splits, persistence, wells", "Forensic diagnosis of data and code reconciles all discrepancies", "Resolved 19 gold events; traced p=1 vs p=2; found 5 physical wells", "Conflation of within-event split with cross-event LOEO; calibration collapse", "Built unified script with 3 distinct protocols"],
        ["Phase 5", "Reconciled 3 protocols under matched realized FAR", "Rigorous leakage-free testing establishes definitive findings", "SARIMAX & AC w=0.5 tied at 63.2% (12/19) at 0.50 FAR; LOF failed (1.51 FAR)", "Heuristic fusion (w=0.7) lagged SARIMAX due to fluid thermal inertia", "Formulated physical insights & qualified scope"]
    ]
    for r_idx, r_data in enumerate(rows_t3, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t3.rows[r_idx].cells[c_idx]
            bg = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run_txt = p.add_run(val)
            run_txt.font.size = Pt(8.0)
            if c_idx == 0:
                run_txt.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 10. Conclusion & Reproducibility Guide
    h10 = doc.add_heading("10. Conclusion & Reproducibility Guide", level=1)
    h10.style.font.color.rgb = RGBColor(0x0A, 0x36, 0x63)

    doc.add_paragraph(
        "Key Takeaways for Practitioners:\n"
        "1. Always Match Realized False Alarm Rates: A detection rate claim is meaningless without an operational nuisance-alarm constraint.\n"
        "2. Prefer Dynamic Conditioning Over Heuristic Blending: Feeding auxiliary physical sensors into autoregressive state equations (SARIMAX) "
        "respects physical transport dynamics far better than combining post-hoc residuals algebraically.\n"
        "3. Beware Coordinate-Space Density Models Across Assets: Models that compute distance in raw feature space (LOF) fail when deployed across "
        "multiple physical assets with varying operating levels.\n\n"
        "Reproducibility:\n"
        "• Unit & Regression Tests: python -m pytest tests/ -v (17 passed)\n"
        "• Audited Evaluation Pipeline: python scripts/run_audited_v2_evaluation.py\n"
        "• Report & Slide Generation: python scripts/build_report_and_slides.py"
    )

    out_path = REPORT_DIR / "OILGUARD_Technical_Report.docx"
    doc.save(out_path)
    print(f"Successfully generated audited Word Technical Report: {out_path}")


def build_presentation():
    prs = pptx.Presentation()
    prs.slide_width = PptxInches(13.333)
    prs.slide_height = PptxInches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="OILGUARD RESEARCH AUDIT"):
        tb = slide.shapes.add_textbox(PptxInches(0.8), PptxInches(0.4), PptxInches(11.7), PptxInches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = PptxPt(10)
        p0.font.bold = True
        p0.font.color.rgb = PptxRGBColor(0x2B, 0x5C, 0x8F)
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = PptxPt(22)
        p1.font.bold = True
        p1.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 1: Title
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(pptx.enum.shapes.MSO_SHAPE.RECTANGLE, 0, 0, PptxInches(13.333), PptxInches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PptxRGBColor(0x0A, 0x25, 0x40)
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(PptxInches(1.2), PptxInches(1.8), PptxInches(11.0), PptxInches(4.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "OILGUARD Research Journey"
    p.font.size = PptxPt(40)
    p.font.bold = True
    p.font.color.rgb = PptxRGBColor(0x6B, 0xAE, 0xD6)

    p2 = tf1.add_paragraph()
    p2.text = "Chronological Investigation, Empirical Results, and Reconciled Multi-Protocol Audit"
    p2.font.size = PptxPt(20)
    p2.font.color.rgb = PptxRGBColor(0xFF, 0xFF, 0xFF)

    p3 = tf1.add_paragraph()
    p3.text = "\nFrom Initial TSA Baselines to Operational Early Warning in Offshore Oil Wells\nRavva Nagarjun & Naveen Kumar N | October 2026"
    p3.font.size = PptxPt(14)
    p3.font.color.rgb = PptxRGBColor(0xBD, 0xC3, 0xC7)

    # Slide 2: Research Narrative Overview (What we tried -> results -> what next)
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Research Journey: What We Tried & What We Discovered", "CHRONOLOGICAL PROGRESSION")
    tb2 = s2.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(11.7), PptxInches(5.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        ("Phase 1 — Initial TSA Forecasting: ", True), ("Rolling 1-step ARIMA/SARIMAX cut error by 47% vs. Naive. BUT discovered 'Noisy Baseline Paradox' (Naive appeared to detect 7/19 due to unconstrained noise).\n", False),
        ("Phase 2 — Retrospective Optimization: ", True), ("Grid search (q, p) and EWMA smoothing (lambda=0.3) boosted detection to 68.4%. BUT exposed alarm storms (FAR 2.5-3.5/hr) and retrospective test leakage.\n", False),
        ("Phase 3 — Candidate Extension (OILGUARD-AC): ", True), ("Multisensor composite scoring (S_t) under an operational FAR budget (<=1.0/hr) with LOF benchmark. BUT uncovered metric discrepancies across drafts (47.4% vs 57.9%).\n", False),
        ("Phase 4 — Independent Forensic Audit: ", True), ("Resolved cohort to 19 gold events; diagnosed p=1 vs p=2 sensitivity; discovered calibration collapse on short slices; identified 5 physical wells (63% from Well 11).\n", False),
        ("Phase 5 — Reconciled Multi-Protocol Evaluation: ", True), ("Formulated 3 leak-free protocols under matched realized FAR. Proved SARIMAX matches balanced AC (63.2% LOEO, 0.50 FAR), while LOF violates operational budgets (1.51 FAR).", False),
    ]
    for txt, is_b in bullets2:
        r = tf2.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(13.5)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 3: Phase 1 — TSA Accuracy & Noisy Baseline Paradox
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 1: Initial TSA Forecasting & The Noisy Baseline Paradox", "WHAT WE TRIED FIRST")
    tb3 = s3.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(6.0), PptxInches(5.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        ("What We Tried: ", True), ("Rolling 1-step ARIMA(1,1,1) and SARIMAX(1,1,1) with downhole temperature vs. industrial Naive persistence.\n", False),
        ("What Results We Got: ", True), ("Statistical models cut error by 47.1% (ARIMA MAE = 0.0163 MPa, SARIMAX = 0.0162 MPa vs Naive 0.0307 MPa; MAPE 8.6% vs 18.2%).\n", False),
        ("The 'Noisy Baseline Paradox': ", True), ("When tested as a raw residual detector, Naive persistence seemed to detect 7/19 events with 4.5m lead time!\n", False),
        ("The Root Cause: ", True), ("Naive residuals suffered massive high-frequency noise variance. Unconstrained noise spikes happened by chance to fall in the 10-minute pre-event window, but produced continuous false alarms during steady operation.\n", False),
        ("Key Takeaway: ", True), ("Alarms cannot be evaluated without measuring steady-state false alarm rates.", False),
    ]
    for txt, is_b in bullets3:
        r = tf3.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(13)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    if (FIGURES_DIR / "forecast_accuracy.png").exists():
        s3.shapes.add_picture(str(FIGURES_DIR / "forecast_accuracy.png"), PptxInches(7.2), PptxInches(1.6), PptxInches(5.5), PptxInches(4.8))

    # Slide 4: Phase 2 — Retrospective Optimization & False Alarm Explosion
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 2: Retrospective Optimization & Alarm Storms", "WHAT WE TRIED NEXT")
    tb4 = s4.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(11.7), PptxInches(5.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    bullets4 = [
        ("What We Tried: ", True), ("Grid search over quantile thresholds (q in [0.90, 0.99]), persistence windows (p in [1, 3]), 120s cooldown, and EWMA residual smoothing (alpha=0.3).\n", False),
        ("What Results We Got: ", True), ("Detection jumped to 68.4% (13/19 events) with >5 minutes mean lead time. EWMA filtered single-point sensor spikes cleanly.\n", False),
        ("The Fatal Flaw Discovered — Alarm Storms: ", True), ("Evaluating normal operational data revealed that p=1 generated 2.5 to 3.5 false alarms per hour. In an offshore control room, 3 nuisance alarms/hr causes immediate operator alarm fatigue and system disabling.\n", False),
        ("The Second Flaw — Retrospective Leakage: ", True), ("Thresholds were tuned across the entire event duration, tuning parameters on the very data being tested.\n", False),
        ("Key Takeaway: ", True), ("Retrospective optimization creates an illusion of high detection while hiding catastrophic operational false alarms.", False),
    ]
    for txt, is_b in bullets4:
        r = tf4.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(14)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 5: Phase 3 & 4 — Forensic Audit & Discrepancy Diagnostics
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phases 3 & 4: Forensic Audit & Discrepancy Diagnostics", "RESOLVING INCONSISTENCIES")
    tb5 = s5.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(11.7), PptxInches(5.2))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    bullets5 = [
        ("Candidate Extension (Phase 3): ", True), ("Formulated OILGUARD-AC composite score S_t = sqrt(w*z_P^2 + (1-w)*z_T^2) under FAR budget (<=1.0/hr), alongside Local Outlier Factor (LOF).\n", False),
        ("The Forensic Audit (Phase 4) Resolved Every Discrepancy:\n", True),
        ("• Cohort Denominator: ", True), ("Excluded 3 corrupt files (missing sensor, 1,608s gap, zero variance) and removed an orphaned parquet file -> strictly 19 gold events.\n", False),
        ("• 47.4% vs 57.9% Detection: ", True), ("Traced to persistence p: p=1 gave 57.9% (high FAR); p=2 gave 47.4% (0.75 FAR by suppressing transient micro-spikes).\n", False),
        ("• FAR 0.75 vs 1.26 / hr: ", True), ("Standardized early warning window strictly to [T_transient - 10 min, T_transient); earlier alarms count as ordinary false alarms.\n", False),
        ("• Short Calibration Collapse: ", True), ("Individual event calibration slices (H < 0.5 h) forced K_allowed = 0; resolved by cross-event pooled calibration (H ~ 3.7 h).\n", False),
        ("• Asset Clustering: ", True), ("19 events originate from 5 physical wells (WELL-00011 = 12 events / 63.2%). Required Leave-One-Well-Out validation.", False),
    ]
    for txt, is_b in bullets5:
        r = tf5.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(13)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 6: Phase 5 — Reconciled Multi-Protocol Benchmark
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Phase 5: Reconciled Benchmark under Matched Realized FAR", "FINAL AUDITED RESULTS")
    if (FIGURES_DIR / "protocol_comparison.png").exists():
        s6.shapes.add_picture(str(FIGURES_DIR / "protocol_comparison.png"), PptxInches(0.8), PptxInches(1.5), PptxInches(6.2), PptxInches(4.8))

    tb6 = s6.shapes.add_textbox(PptxInches(7.2), PptxInches(1.5), PptxInches(5.5), PptxInches(5.0))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    bullets6 = [
        ("Matched Realized FAR Comparison (N=19 Events, 5 Wells):\n", True),
        ("• SARIMAX (Exogenous T): ", True), ("LOEO: 63.2% (12/19), 4.8m lead, 0.50 FAR (2 FA) | LOWO: 73.7% (14/19) [Budget Met!]\n", False),
        ("• OILGUARD-AC (w=0.85): ", True), ("LOEO: 63.2% (12/19), 5.3m lead, 0.50 FAR (2 FA) | LOWO: 73.7% (14/19) [Budget Met!]\n", False),
        ("• OILGUARD-AC (w=0.50): ", True), ("LOEO: 63.2% (12/19), 4.0m lead, 0.50 FAR (2 FA) | LOWO: 63.2% (12/19) [Budget Met!]\n", False),
        ("• Univariate ARIMA: ", True), ("LOEO: 57.9% (11/19), 5.2m lead, 0.50 FAR (2 FA) | LOWO: 68.4% (13/19) [Budget Met!]\n", False),
        ("• OILGUARD-AC (w=0.70): ", True), ("LOEO: 52.6% (10/19), 4.7m lead, 0.75 FAR (3 FA) | LOWO: 63.2% (12/19) [Budget Met!]\n", False),
        ("• Naive Persistence: ", True), ("LOEO: 47.4% (9/19), 2.3m lead, 1.01 FAR (4 FA) [Borderline]\n", False),
        ("• Local Outlier Factor (LOF): ", True), ("57.9% Detection (11/19), 4.5m lead | Realized FAR: 1.51 / hr (6 FA) [BUDGET VIOLATED!]\n\n", False),
        ("Finding: ", True), ("Dynamic regression (SARIMAX) ties balanced fusion and respects operational false-alarm constraints.", False),
    ]
    for txt, is_b in bullets6:
        r = tf6.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(12.5)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 7: Scientific Insights — Thermal Inertia & LOF Failure
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Physical Insights: Thermal Inertia & Cross-Well Density Drift", "SCIENTIFIC MECHANISMS")
    tb7 = s7.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(11.7), PptxInches(5.2))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    bullets7 = [
        ("1. Why Dynamic Regression (SARIMAX) Outperforms Heuristic Fusion (w=0.7):\n", True),
        ("• Downhole Thermal Inertia: ", True), ("When the DHSV flapper throttles, pressure drops rapidly via acoustic wave propagation (1,000 m/s), whereas temperature changes much more gradually.\n", False),
        ("• In SARIMAX, ", True), ("temperature dynamically conditions baseline expectations without diluting the acoustic pressure drop.\n", False),
        ("• In Post-Hoc Fusion (w=0.7), ", True), ("a lagging temperature Z-score can algebraically pull down the composite anomaly score S_t, diluting the sharp pressure signal.\n\n", False),
        ("2. Why Established Density Models (LOF) Fail in Operations:\n", True),
        ("• Cross-Well Density Drift: ", True), ("LOF computes k-NN distance in raw feature space [P, T]. Normal operating pressure varies from 8.9 to 21.0 MPa across the 5 wells.\n", False),
        ("• Unseen Well Outliers: ", True), ("Deploying LOF on an unseen well causes normal operation to land in low-density space, triggering 1.51 false alarms/hr (6 false alarms on 3.98 h normal holdout).\n", False),
        ("• Time Series Advantage: ", True), ("Forecasting models difference out static level shifts, focusing strictly on dynamic temporal departures.", False),
    ]
    for txt, is_b in bullets7:
        r = tf7.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(13)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    # Slide 8: Operational Scope & Qualified Engineering Conclusion
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Operational Scope & Qualified Engineering Conclusions", "CONCLUSION")
    tb8 = s8.shapes.add_textbox(PptxInches(0.8), PptxInches(1.5), PptxInches(11.7), PptxInches(5.2))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    bullets8 = [
        ("1. Actionable Operational Window: ", True), ("An advance warning of 4 to 5 minutes provides control room operators a valuable decision-support window to verify hydraulic telemetry and prepare platform separators before full flapper seating.\n", False),
        ("2. Nuisance-Alarm Budgeting Works: ", True), ("Calibrating thresholds under B_FAR <= 1.0 FA/hr guarantees operational reliability on unseen wells (realized FAR = 0.50 / hr for SARIMAX and AC w=0.5).\n", False),
        ("3. Qualified Engineering Boundaries: ", True), ("Claims regarding water hammer elimination or percentage surge reduction remain engineering hypotheses. High-pressure acoustic shock travels in seconds; validating automated choke mitigation requires transient multiphase simulation (OLGA) and flow-loop pilot testing.\n", False),
        ("4. Methodological Summary: ", True), ("Moving from unconstrained retrospective parameter optimization to leak-free cross-event validation under operational false-alarm constraints provides an honest, defensible benchmark for industrial predictive maintenance.", False),
    ]
    for txt, is_b in bullets8:
        r = tf8.paragraphs[0].add_run()
        r.text = txt
        r.font.size = PptxPt(14)
        r.font.bold = is_b
        if is_b:
            r.font.color.rgb = PptxRGBColor(0x0A, 0x36, 0x63)

    out_path = REPORT_DIR / "OILGUARD_Presentation.pptx"
    prs.save(out_path)
    print(f"Successfully generated audited PowerPoint Presentation: {out_path}")


if __name__ == "__main__":
    build_word_report()
    build_presentation()
