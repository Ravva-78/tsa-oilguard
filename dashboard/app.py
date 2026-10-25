from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

st.set_page_config(page_title="OILGUARD | Early Warning", page_icon="🌊", layout="wide")
ROOT = Path(__file__).resolve().parent
if not (ROOT / "results").exists() and (ROOT.parent / "results").exists():
    ROOT = ROOT.parent
METRICS = ROOT / "results" / "audited_v2" / "metrics"
TABLES = ROOT / "results" / "audited_v2" / "tables"

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:radial-gradient(ellipse at 10% 0%,rgba(13,148,136,.14),transparent 35%),radial-gradient(ellipse at 90% 5%,rgba(14,116,144,.17),transparent 32%),#07131f;color:#e6f0f6}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1d2a,#081521);border-right:1px solid #24404e}
.block-container{padding-top:1.4rem;max-width:1600px}
h1,h2,h3{color:#f3f8fb;letter-spacing:-.035em}
.eyebrow,.kicker{color:#59d6c1;font-family:'DM Mono',monospace;letter-spacing:.13em;text-transform:uppercase;font-size:.72rem}
.hero{padding:.7rem 0 1rem}.hero-title{font-size:clamp(2.1rem,4vw,3.5rem);font-weight:800;line-height:1.02;margin:.35rem 0 .65rem}.hero-sub{color:#9fb7c5;max-width:900px;line-height:1.65}
.pill{display:inline-block;border:1px solid #28685f;color:#8ef0df;background:#0d94881a;border-radius:999px;padding:.3rem .65rem;font-size:.72rem;margin:.8rem .35rem 0 0}
.panel{background:linear-gradient(145deg,#122a39f2,#0a1c29f2);border:1px solid #294452;border-radius:16px;padding:1rem 1.1rem}.metric-label{color:#9fb7c5;font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}.metric-value{color:#f3f8fb;font-size:1.8rem;font-weight:800;margin:.2rem 0}.metric-foot,.note{color:#8fa8b7;font-size:.8rem;line-height:1.55}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#122a39,#0a1c29);border:1px solid #294452;padding:.9rem;border-radius:14px}div[data-testid="stMetricLabel"]{color:#9fb7c5}div[data-testid="stMetricValue"]{color:#f3f8fb}
.stTabs [data-baseweb="tab"]{color:#a9bfca}.stTabs [aria-selected="true"]{color:#71e6d3!important}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #28685f;background:#0d948826;color:#e1fbf6;font-weight:700}
hr{border-color:#24404e}
</style>
""", unsafe_allow_html=True)

# Audited-v2 reference snapshot from the project's supplied audit summary.
# Verify against matched_far_comparison.csv before publishing these values externally.
REFERENCE = pd.DataFrame([
    {"Method":"SARIMAX + temperature", "Detection %":63.2, "Hits":"12/19", "Mean lead (min)":4.83, "FAR / hr":0.50},
    {"Method":"OILGUARD-AC (w=0.5)", "Detection %":63.2, "Hits":"12/19", "Mean lead (min)":4.01, "FAR / hr":0.50},
    {"Method":"Univariate ARIMA", "Detection %":57.9, "Hits":"11/19", "Mean lead (min)":5.18, "FAR / hr":0.50},
    {"Method":"OILGUARD-AC (w=0.7)", "Detection %":52.6, "Hits":"10/19", "Mean lead (min)":4.67, "FAR / hr":0.75},
    {"Method":"Naive persistence", "Detection %":42.1, "Hits":"8/19", "Mean lead (min)":3.50, "FAR / hr":0.50},
    {"Method":"Local Outlier Factor", "Detection %":57.9, "Hits":"11/19", "Mean lead (min)":4.50, "FAR / hr":1.51},
])

def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

def style_fig(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,19,31,.2)",
        font=dict(color="#c5d7e1",family="Manrope"), margin=dict(l=15,r=15,t=35,b=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,197,214,.13)"), yaxis=dict(gridcolor="rgba(148,197,214,.13)"))
    return fig

def robust_scale(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    mad = (x - x.median()).abs().median()
    s = 1.4826 * mad
    if not np.isfinite(s) or s < 1e-9: s = x.std()
    return float(s) if np.isfinite(s) and s > 1e-9 else 1.0

def replay_forecast(frame, pressure, temperature, model, train_pct, order, threshold, persistence):
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels missing; install it with pip install statsmodels")
    y = pd.to_numeric(frame[pressure], errors="coerce").interpolate(limit_direction="both")
    n = len(y); split = int(n * train_pct / 100)
    if n < 50 or split < 30 or split >= n-3: raise ValueError("Upload at least 50 rows of numeric telemetry.")
    preds = np.full(n, np.nan)
    if model.startswith("SARIMAX"):
        if not temperature: raise ValueError("Choose a temperature sensor for SARIMAX.")
        temp = pd.to_numeric(frame[temperature], errors="coerce").interpolate(limit_direction="both")
        exog = temp.shift(1).bfill().to_frame("T_lag1")
        fit = SARIMAX(y.iloc[:split], exog=exog.iloc[:split], order=order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
        train_resid = y.iloc[:split].to_numpy() - np.asarray(fit.fittedvalues)
        result = fit
        for i in range(split,n):
            xnext = exog.iloc[[i]]
            preds[i] = float(result.get_forecast(1, exog=xnext).predicted_mean.iloc[0])
            result = result.append(endog=y.iloc[[i]], exog=xnext, refit=False)
    else:
        result = ARIMA(y.iloc[:split], order=order).fit()
        train_resid = y.iloc[:split].to_numpy() - np.asarray(result.fittedvalues)
        for i in range(split,n):
            preds[i] = float(result.forecast(1).iloc[0])
            result = result.append(y.iloc[[i]], refit=False)
    resid = y.to_numpy() - preds
    cal_start = int(split * .75)
    cal = train_resid[cal_start:]
    cal = cal[np.isfinite(cal)]
    scale = robust_scale(cal) if len(cal) else robust_scale(train_resid)
    score = np.abs(resid) / scale
    warning = np.zeros(n,dtype=bool); run = 0
    for i in range(split,n):
        run = run+1 if np.isfinite(score[i]) and score[i] >= threshold else 0
        if run >= persistence: warning[i] = True
    out = frame.copy(); out["row_index"] = np.arange(n); out["pressure_actual"] = y
    out["pressure_forecast"] = preds; out["forecast_residual"] = resid
    out["normalized_score"] = score; out["warning"] = warning
    return out, split

with st.sidebar:
    st.markdown('<div class="eyebrow">OILGUARD / CONTROL DECK</div>', unsafe_allow_html=True)
    st.markdown("## 🌊 OILGUARD")
    st.caption("Alarm-constrained early warning")
    page = st.radio("Navigation", ["Command Center","Telemetry Replay","Model Benchmarks","Event Explorer","About & Method"], label_visibility="collapsed")
    st.markdown("---")
    budget = st.select_slider("Nuisance alarm budget", options=[0.25,0.5,0.75,1.0,1.5,2.0], value=1.0, format_func=lambda x:f"{x:.2f} alarms / normal hr")
    st.caption("Petrobras 3W · Class 2 · Spurious DHSV closure")
    st.markdown('<div class="note">Research prototype only. No SCADA connection or control actuation.</div>', unsafe_allow_html=True)

matched = read_csv(METRICS/"matched_far_comparison.csv")
outcomes = read_csv(METRICS/"per_event_outcomes.csv")
manifest = read_csv(TABLES/"cohort_manifest.csv")
definitions = read_csv(TABLES/"evaluation_definitions.csv")

st.markdown('''<div class="hero"><div class="eyebrow">OFFSHORE INTELLIGENCE SYSTEM · CLASS 2 EVENT MONITORING</div><div class="hero-title">See the signal before the shutdown.</div><div class="hero-sub">An interactive research cockpit for pressure forecasting, temperature-aware anomaly evidence, and false-alarm-aware early warning for spurious downhole safety valve closure.</div><span class="pill">P-TPT · Pressure</span><span class="pill">T-TPT · Temperature</span><span class="pill">LOEO-CV · Cross-well evaluation</span><span class="pill">Alarm budget aware</span></div>''', unsafe_allow_html=True)

if page == "Command Center":
    st.markdown('<div class="kicker">01 / Command center</div>',unsafe_allow_html=True)
    st.markdown("### Fleet-level early-warning overview")
    st.caption("Retrospective research metrics from the supplied audited summary; not live telemetry.")
    a,b,c,d = st.columns(4)
    a.metric("Audited event cohort","19 events","5 distinct physical wells")
    b.metric("Top reported detection","63.2%","12 of 19 events")
    c.metric("SARIMAX mean lead","4.83 min","Temperature as exogenous input")
    d.metric("Reference realized FAR","0.50 / hr","2 alarms / 3.9778 normal hours")
    left,right = st.columns([1.15,1])
    with left:
        st.markdown("#### Early-warning detection by method")
        q=REFERENCE.sort_values("Detection %")
        fig=go.Figure(go.Bar(x=q["Detection %"],y=q["Method"],orientation="h",text=q["Detection %"].map(lambda x:f"{x:.1f}%"),textposition="outside",marker_color=["#48d7bd" if x=="SARIMAX + temperature" else "#527b99" for x in q["Method"]]))
        fig.update_layout(xaxis_title="Detected events (%)",yaxis_title="",xaxis_range=[0,78])
        st.plotly_chart(style_fig(fig,390),use_container_width=True)
    with right:
        st.markdown("#### Detection vs nuisance alarms")
        fig=px.scatter(REFERENCE,x="FAR / hr",y="Detection %",size="Mean lead (min)",color="Method",hover_name="Method",hover_data=["Hits","Mean lead (min)"])
        fig.add_vline(x=budget,line_dash="dash",line_color="#ffbf69",annotation_text=f"Budget {budget:g}/hr")
        fig.update_layout(showlegend=False,xaxis_range=[0,2],yaxis_range=[0,80])
        st.plotly_chart(style_fig(fig,390),use_container_width=True)
        st.markdown('<div class="note">Bubble size represents mean lead time. Points to the right of the selected budget exceed that target in this reference comparison.</div>',unsafe_allow_html=True)
    view=REFERENCE.copy(); view["Budget status"]=np.where(view["FAR / hr"]<=budget,"Within selected budget","Exceeds selected budget")
    st.markdown("#### Model snapshot")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.info("SARIMAX and OILGUARD-AC (w=0.5) both report 63.2% detection in the cross-well comparison. SARIMAX has the longer reported mean lead time; the multisensor score is not uniformly superior.")

elif page == "Telemetry Replay":
    st.markdown('<div class="kicker">02 / Telemetry replay lab</div>',unsafe_allow_html=True)
    st.markdown("### Upload a well telemetry CSV")
    st.markdown('<div class="note">The app fits on an initial training segment and replays the later sequence one step at a time. It is an exploratory prototype, not a calibrated operational alarm or SCADA interface.</div>',unsafe_allow_html=True)
    upload=st.file_uploader("Telemetry file",type=["csv"],help="Suggested columns: timestamp, P-TPT, T-TPT. At least 50 rows are recommended.")
    if upload is None:
        st.markdown('<div class="panel"><div class="eyebrow">INPUT CONTRACT</div><h3>One row per observation</h3><p class="note">Use numeric pressure and optional temperature columns. No upload means no live prediction is being made. For a first test, use a copy of a single well telemetry CSV, not the entire 1.8 GB dataset.</p></div>',unsafe_allow_html=True)
    else:
        raw=pd.read_csv(upload); nums=[c for c in raw.columns if pd.to_numeric(raw[c],errors="coerce").notna().sum()>=max(20,int(len(raw)*.5))]
        if len(raw)<50 or not nums: st.error("Please upload at least 50 rows with usable numeric sensor columns.")
        else:
            pdefault=next((c for c in nums if "p-tpt" in c.lower() or "pressure" in c.lower()),nums[0])
            tdefault=next((c for c in nums if "t-tpt" in c.lower() or "temp" in c.lower()),"None")
            x1,x2,x3=st.columns(3)
            pressure=x1.selectbox("Pressure sensor",nums,index=nums.index(pdefault))
            tchoices=["None"]+[c for c in nums if c!=pressure]
            temperature=x2.selectbox("Temperature sensor",tchoices,index=tchoices.index(tdefault) if tdefault in tchoices else 0)
            model=x3.selectbox("Forecasting method",["Univariate ARIMA","SARIMAX + lagged temperature"])
            y1,y2,y3,y4=st.columns(4)
            train=y1.slider("Initial training %",50,80,60,5)
            p=y2.number_input("AR order p",0,4,1); diff=y3.number_input("Differencing d",0,2,0); q=y4.number_input("MA order q",0,4,1)
            z1,z2=st.columns(2); threshold=z1.slider("Normalized residual threshold",1.0,8.0,3.5,.25); persist=z2.slider("Consecutive points to trigger",1,10,3)
            st.caption(f"{len(raw):,} rows · {len(raw.columns)} columns · {upload.name}")
            if st.button("▶ Run prospective replay",type="primary",use_container_width=True):
                try:
                    with st.spinner("Fitting forecaster and replaying holdout observations…"):
                        result,split=replay_forecast(raw,pressure,None if temperature=="None" else temperature,model,train,(int(p),int(diff),int(q)),threshold,persist)
                    st.session_state["oilguard_result"]=result; st.session_state["oilguard_split"]=split
                    st.session_state["oilguard_meta"]={"model":model,"pressure":pressure,"threshold":threshold,"persistence":persist}
                except Exception as e: st.error(f"Replay failed: {e}")
            result=st.session_state.get("oilguard_result")
            if result is not None:
                split=st.session_state["oilguard_split"]; meta=st.session_state["oilguard_meta"]; hold=result.iloc[split:]
                valid=hold["pressure_forecast"].notna(); mae=np.mean(np.abs(hold.loc[valid,"pressure_actual"]-hold.loc[valid,"pressure_forecast"])) if valid.any() else np.nan
                k1,k2,k3,k4=st.columns(4); k1.metric("Replay rows",f"{len(hold):,}"); k2.metric("Warning-active samples",f"{int(hold.warning.sum()):,}"); k3.metric("Holdout MAE",f"{mae:.4f}" if np.isfinite(mae) else "N/A"); k4.metric("Threshold / persistence",f"{meta['threshold']:.2f} / {meta['persistence']}")
                fig=go.Figure(); fig.add_trace(go.Scatter(x=result.row_index,y=result.pressure_actual,name="Observed pressure",line=dict(color="#b8cbd7",width=1.5))); fig.add_trace(go.Scatter(x=result.row_index,y=result.pressure_forecast,name="One-step forecast",line=dict(color="#48d7bd",width=1.7,dash="dot")))
                warnings=result[result.warning]; fig.add_trace(go.Scatter(x=warnings.row_index,y=warnings.pressure_actual,name="Warning points",mode="markers",marker=dict(color="#ff7e79",size=8,symbol="diamond"))); fig.add_vline(x=split,line_dash="dash",line_color="#ffbf69",annotation_text="Prospective replay starts"); fig.update_layout(xaxis_title="Row index",yaxis_title=pressure)
                st.plotly_chart(style_fig(fig,420),use_container_width=True)
                score=go.Figure(); score.add_trace(go.Scatter(x=result.row_index,y=result.normalized_score,name="Normalized residual score",line=dict(color="#64a9ff"))); score.add_hline(y=meta["threshold"],line_dash="dash",line_color="#ffbf69",annotation_text="Threshold"); score.add_trace(go.Scatter(x=warnings.row_index,y=warnings.normalized_score,name="Warning",mode="markers",marker=dict(color="#ff7e79",size=7))); score.update_layout(xaxis_title="Row index",yaxis_title="Normalized absolute residual")
                st.plotly_chart(style_fig(score,300),use_container_width=True)
                if len(warnings): st.warning(f"{len(warnings)} warning-active samples found. This is not proof of an impending DHSV closure.")
                else: st.success("No score crossed the configured persistence rule during this replay.")
                st.download_button("Download replay results CSV",result.to_csv(index=False).encode(),"oilguard_replay_results.csv","text/csv")
                st.dataframe(result.iloc[max(split-5,0):min(split+25,len(result))],use_container_width=True,hide_index=True)

elif page == "Model Benchmarks":
    st.markdown('<div class="kicker">03 / Model benchmarks</div>',unsafe_allow_html=True)
    st.markdown("### Compare methods under the selected nuisance-alarm budget")
    left,right=st.columns([1.2,1])
    with left:
        fig=px.bar(REFERENCE.sort_values("Detection %",ascending=False),x="Method",y="Detection %",color="FAR / hr",color_continuous_scale=["#35cdb7","#ffbf69","#ff817a"],hover_data=["Hits","Mean lead (min)"])
        fig.update_layout(xaxis_tickangle=-25,yaxis_range=[0,80]); st.plotly_chart(style_fig(fig,420),use_container_width=True)
    with right:
        fig=px.scatter(REFERENCE,x="FAR / hr",y="Mean lead (min)",size="Detection %",color="Method",hover_name="Method")
        fig.add_vline(x=budget,line_dash="dash",line_color="#ffbf69"); fig.update_layout(showlegend=False); st.plotly_chart(style_fig(fig,420),use_container_width=True)
    st.markdown("#### Project's audited comparison CSV")
    if matched is not None: st.dataframe(matched,use_container_width=True,hide_index=True)
    else: st.warning("Could not find results/audited_v2/metrics/matched_far_comparison.csv beside app.py. Built-in reference summary is displayed instead.")
    with st.expander("How to interpret the alarm budget"):
        st.write("Calibration FAR and realized holdout FAR are different. A method is budget-compliant only if its realized FAR meets the target under the reported protocol. Detection rates should be compared at matched FAR, not at arbitrarily chosen thresholds.")

elif page == "Event Explorer":
    st.markdown('<div class="kicker">04 / Event explorer</div>',unsafe_allow_html=True)
    st.markdown("### Inspect event-level outcomes")
    if outcomes is None: st.warning("Place this app in the OILGUARD project root so it can load results/audited_v2/metrics/per_event_outcomes.csv.")
    else:
        query=st.text_input("Filter by event ID or well ID")
        filtered=outcomes.copy()
        if query:
            mask=filtered.astype(str).apply(lambda c:c.str.contains(query,case=False,na=False)).any(axis=1); filtered=filtered[mask]
        st.caption(f"Showing {len(filtered)} of {len(outcomes)} event outcome rows")
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.download_button("Download filtered outcomes",filtered.to_csv(index=False).encode(),"oilguard_filtered_event_outcomes.csv","text/csv")
    if manifest is not None:
        with st.expander("Cohort manifest and exclusions"): st.dataframe(manifest,use_container_width=True,hide_index=True)
    if definitions is not None:
        with st.expander("Evaluation definitions"): st.dataframe(definitions,use_container_width=True,hide_index=True)

else:
    st.markdown('<div class="kicker">05 / About & method</div>',unsafe_allow_html=True)
    l,r=st.columns(2)
    with l:
        st.markdown("### Research workflow")
        st.markdown("1. **Observe** pressure (`P-TPT`) and temperature (`T-TPT`) telemetry.\n2. **Forecast** the next pressure observation using ARIMA or SARIMAX.\n3. **Score** the normalized absolute one-step residual.\n4. **Persist** threshold exceedances before raising a warning marker.\n5. **Evaluate** detection, lead time and false alarms per normal operating hour.")
    with r:
        st.markdown("### Safety and scientific guardrails")
        st.markdown("- This is a research prototype, not a certified safety instrument.\n- Uploaded telemetry is processed by the local Streamlit app; there is no SCADA connection.\n- Upload-replay thresholds are user-selected and are not the audited cross-well thresholds.\n- A high score does not prove a DHSV closure will occur.\n- No automatic choke adjustment or shutdown command is implemented.")
    st.markdown("### Run locally")
    st.code(".\\.venv\\Scripts\\python.exe -m pip install -r requirements-dashboard.txt\n.\\.venv\\Scripts\\python.exe -m streamlit run app.py",language="powershell")
    st.markdown("### Expected project layout")
    st.code("TSA-OILGUARD/\n├── app.py\n├── requirements-dashboard.txt\n└── results/audited_v2/\n    ├── metrics/matched_far_comparison.csv\n    ├── metrics/per_event_outcomes.csv\n    └── tables/cohort_manifest.csv",language="text")
    st.warning("Before presenting, verify embedded benchmark reference values against the final audited CSV. Do not report uploaded replay results as audited research metrics.")

st.markdown("---")
st.markdown('<div class="note">OILGUARD-AC · Research dashboard · Petrobras 3W Class 2 · Retrospective benchmark analytics only · Not for operational safety decisions.</div>',unsafe_allow_html=True)
