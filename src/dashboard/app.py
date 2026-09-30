"""
app.py
------
Streamlit Monitoring Dashboard for Smart Factory Predictive Maintenance.

Launch:
    streamlit run src/dashboard/app.py

Author : Smart Factory PdM Team
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Factory PdM Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #09090e 0%, #15102a 50%, #03001e 100%);
    padding: 2.2rem 2.8rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.45);
    border: 1px solid rgba(102,126,234,0.15);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.8px;
    background: linear-gradient(90deg, #ffffff 30%, #a3b8ff 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0.4rem 0 0 0;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2rem 0 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px dashed rgba(102,126,234,0.25);
    letter-spacing: -0.3px;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0b0f19, #131a2b);
    border: 1px solid rgba(102,126,234,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",
    "secondary": "#a855f7",
    "success": "#22c55e",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#06b6d4",
    "gradient": ["#6366f1", "#a855f7", "#ec4899", "#3b82f6", "#14b8a6"],
    "sensors": ["#6366f1", "#ec4899", "#22c55e", "#f59e0b"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False),
)


# ── Data Loading Helpers ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_telemetry():
    path = DATA_RAW / "PdM_telemetry.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    # Shift to last 2 years
    df = shift_to_last_two_years(df)
    return df

@st.cache_data(ttl=300)
def load_failures():
    path = DATA_RAW / "PdM_failures.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = shift_to_last_two_years(df)
    return df

@st.cache_data(ttl=300)
def load_errors():
    path = DATA_RAW / "PdM_errors.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = shift_to_last_two_years(df)
    return df

@st.cache_data(ttl=300)
def load_machines():
    path = DATA_RAW / "PdM_machines.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=300)
def load_maint():
    path = DATA_RAW / "PdM_maint.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = shift_to_last_two_years(df)
    return df

@st.cache_data(ttl=300)
def load_metrics():
    path = METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=300)
def load_best_model_meta():
    path = METRICS_DIR / "best_model_meta.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(ttl=300)
def load_validation_report():
    path = METRICS_DIR / "validation_merged.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def shift_to_last_two_years(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to shift date range of DataFrame to 2024-07-19 -> 2026-07-18."""
    if "datetime" not in df.columns:
        return df
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # First year (shifted from original 2015 dataset)
    df_y1 = df.copy()
    df_y1["datetime"] = df_y1["datetime"] + pd.Timedelta(days=3487)
    
    # Second year (shifted from original 2015 dataset)
    df_y2 = df.copy()
    df_y2["datetime"] = df_y2["datetime"] + pd.Timedelta(days=3852)
    
    combined = pd.concat([df_y1, df_y2], ignore_index=True)
    if "machineID" in combined.columns:
        combined = combined.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    else:
        combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def get_pipeline_status():
    """Check which pipeline outputs exist."""
    checks = {
        "Raw Data": all((DATA_RAW / f"PdM_{n}.csv").exists() for n in ["telemetry","errors","failures","maint","machines"]),
        "Merged Data": (DATA_INTERIM / "merged_dataset.csv").exists(),
        "Cleaned Data": (DATA_INTERIM / "cleaned_dataset.csv").exists(),
        "Features": (DATA_PROCESSED / "features_engineered.csv").exists(),
        "EDA Charts": len(list(FIGURES_DIR.glob("*.png"))) >= 10 if FIGURES_DIR.exists() else False,
        "Model Metrics": (METRICS_DIR / "model_comparison.csv").exists(),
        "Best Model": (MODELS_DIR / "best_model.joblib").exists(),
    }
    return checks


@st.cache_data(ttl=300)
def compute_machine_health_scores(telemetry, failures, errors):
    """Calculate an interactive health score (0-100) per machine."""
    if telemetry is None:
        return {}
    latest_telem = telemetry.groupby("machineID").tail(100)
    
    # Base stats for deviations
    means = telemetry[["volt", "rotate", "pressure", "vibration"]].mean()
    stds = telemetry[["volt", "rotate", "pressure", "vibration"]].std()
    
    health_scores = {}
    max_date = telemetry["datetime"].max()
    
    for machine_id, group in latest_telem.groupby("machineID"):
        volt_dev = ((group["volt"] - means["volt"]).abs() / stds["volt"]).mean()
        rot_dev = ((group["rotate"] - means["rotate"]).abs() / stds["rotate"]).mean()
        pres_dev = ((group["pressure"] - means["pressure"]).abs() / stds["pressure"]).mean()
        vib_dev = ((group["vibration"] - means["vibration"]).abs() / stds["vibration"]).mean()
        
        avg_dev = (volt_dev + rot_dev + pres_dev + vib_dev) / 4.0
        score = max(15, min(100, int(100 - (avg_dev - 0.7) * 45)))
        
        if errors is not None:
            # Drop score further if there are recent errors in the last 30 days
            recent_err_cnt = len(errors[
                (errors["machineID"] == machine_id) & 
                (errors["datetime"] > (max_date - pd.Timedelta(days=30)))
            ])
            score = max(10, score - recent_err_cnt * 12)
            
        health_scores[machine_id] = score
        
    return health_scores


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏭 Smart Factory Predictive Maintenance</h1>
    <p>Premium Fleet Health Observability & Machine Diagnostics Dashboard • Last 2 Years Timeline (2024-2026)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Navigation")
    page = st.radio(
        "Select View",
        ["🏠 Overview", "🏭 Fleet Health Index", "📊 Sensor Analytics", "🔮 Failure Prediction Playground", 
         "🤖 Model Performance", "🔍 Data Quality", "📋 Pipeline Status"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 📁 Project Info")
    st.markdown(f"**Version:** 3.0.0")
    st.markdown(f"**Timeline:** July 2024 - July 2026")
    st.markdown(f"**Last Refresh:** {datetime.now().strftime('%H:%M:%S')}")

    if st.button("🔄 Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    telemetry = load_telemetry()
    failures = load_failures()
    errors = load_errors()
    machines = load_machines()
    metrics_df = load_metrics()

    # KPI Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)

    n_machines = len(machines) if machines is not None else 0
    n_records = len(telemetry) if telemetry is not None else 0
    n_failures = len(failures) if failures is not None else 0
    n_errors = len(errors) if errors is not None else 0

    best_f1 = "N/A"
    if metrics_df is not None and "f1_score" in metrics_df.columns:
        best_f1 = f"{metrics_df['f1_score'].max():.4f}"

    with col1:
        st.metric("🖥️ Machines", f"{n_machines}")
    with col2:
        st.metric("📈 Telemetry (2 Yrs)", f"{n_records:,}")
    with col3:
        st.metric("⚠️ Failure Events", f"{n_failures}")
    with col4:
        st.metric("🔴 Error Events", f"{n_errors:,}")
    with col5:
        st.metric("🏆 Best F1 (XGBoost)", best_f1)

    st.markdown("---")

    # Pipeline Status + Failure Timeline
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-title">Pipeline Health</div>', unsafe_allow_html=True)
        status = get_pipeline_status()
        for stage, ok in status.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} **{stage}**")

    with right:
        st.markdown('<div class="section-title">Failure Events Over Time</div>', unsafe_allow_html=True)
        if failures is not None and len(failures) > 0:
            fail_monthly = failures.copy()
            fail_monthly["month"] = fail_monthly["datetime"].dt.to_period("M").astype(str)
            monthly_counts = fail_monthly.groupby("month").size().reset_index(name="count")
            fig = px.bar(
                monthly_counts, x="month", y="count",
                color_discrete_sequence=[COLORS["danger"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Monthly Failure Count", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No failure data available.")

    # Machine Fleet Overview
    if machines is not None:
        st.markdown('<div class="section-title">Machine Fleet Overview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            model_counts = machines["model"].value_counts().reset_index()
            model_counts.columns = ["Model", "Count"]
            fig = px.pie(
                model_counts, values="Count", names="Model",
                color_discrete_sequence=COLORS["gradient"],
                hole=0.45,
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Machine Model Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                machines, x="age", nbins=15,
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, title="Machine Age Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FLEET HEALTH INDEX
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Fleet Health Index":
    st.markdown('<div class="section-title">Industrial Fleet Health Matrix</div>', unsafe_allow_html=True)
    
    telemetry = load_telemetry()
    failures = load_failures()
    errors = load_errors()
    
    if telemetry is None:
        st.error("Telemetry data not found. Run the pipeline first.")
    else:
        # Calculate health scores
        health_scores = compute_machine_health_scores(telemetry, failures, errors)
        
        # Fleet average
        avg_health = np.mean(list(health_scores.values()))
        
        # Categorize
        healthy = [m for m, s in health_scores.items() if s >= 80]
        at_risk = [m for m, s in health_scores.items() if 50 <= s < 80]
        critical = [m for m, s in health_scores.items() if s < 50]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fleet Average Health", f"{avg_health:.1f}%")
        with col2:
            st.metric("Healthy Machines (≥80%)", f"{len(healthy)} / 100")
        with col3:
            st.metric("At-Risk Machines (50-79%)", f"{len(at_risk)} / 100", delta=f"{len(at_risk)} check", delta_color="inverse")
        with col4:
            st.metric("Critical Machines (<50%)", f"{len(critical)} / 100", delta=f"{len(critical)} urgent", delta_color="inverse")
            
        st.markdown("---")
        st.write("### 🖥️ Real-time Machine Status Grid")
        st.write("This interactive grid displays the operational status of all 100 factory machines based on anomaly score indexes:")
        
        # Render a beautiful grid of 10 columns
        grid_cols = st.columns(10)
        for i in range(1, 101):
            col_idx = (i - 1) % 10
            score = health_scores.get(i, 100)
            
            if score >= 80:
                color_bg = "rgba(34, 197, 94, 0.12)"
                color_border = COLORS["success"]
                status_text = "Healthy"
            elif score >= 50:
                color_bg = "rgba(245, 158, 11, 0.12)"
                color_border = COLORS["warning"]
                status_text = "At Risk"
            else:
                color_bg = "rgba(239, 68, 68, 0.12)"
                color_border = COLORS["danger"]
                status_text = "Critical"
                
            with grid_cols[col_idx]:
                st.markdown(
                    f"""<div style="background:{color_bg}; border: 1px solid {color_border}; 
                    border-radius: 12px; padding: 0.75rem 0.5rem; text-align: center; margin-bottom: 0.9rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.15); transition: transform 0.2s;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px;">M-{i:03d}</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; margin: 0.1rem 0;">{score}%</div>
                    <div style="font-size: 0.65rem; font-weight: 700; color: {color_border}; text-transform: uppercase;">{status_text}</div>
                    </div>""",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENSOR ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sensor Analytics":
    telemetry = load_telemetry()

    if telemetry is None:
        st.error("Telemetry data not found. Place PdM_telemetry.csv in data/raw/")
    else:
        machines_list = sorted(telemetry["machineID"].unique())

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_machine = st.selectbox("Select Machine", list(machines_list), index=0)
        with col2:
            sensor = st.selectbox("Primary Sensor", ["volt", "rotate", "pressure", "vibration"])
        with col3:
            n_points = st.slider("Data Points", 100, 5000, 500, step=100)

        machine_data = telemetry[telemetry["machineID"] == selected_machine].tail(n_points)

        # Sensor Time Series
        st.markdown('<div class="section-title">Sensor Time Series</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Voltage", "Rotation", "Pressure", "Vibration"])
        sensors = ["volt", "rotate", "pressure", "vibration"]
        for i, s in enumerate(sensors):
            row, col = divmod(i, 2)
            fig.add_trace(
                go.Scatter(
                    x=machine_data["datetime"], y=machine_data[s],
                    mode="lines", name=s.capitalize(),
                    line=dict(color=COLORS["sensors"][i], width=1.5),
                ),
                row=row+1, col=col+1,
            )
        fig.update_layout(height=500, showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Sensor Statistics
        st.markdown('<div class="section-title">Sensor Statistics</div>', unsafe_allow_html=True)
        stats = machine_data[sensors].describe().T
        stats["cv%"] = (stats["std"] / stats["mean"] * 100).round(2)
        st.dataframe(stats.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)

        # Correlation Heatmap
        st.markdown('<div class="section-title">Sensor Correlation Matrix</div>', unsafe_allow_html=True)
        corr = machine_data[sensors].corr()
        fig = px.imshow(
            corr, text_auto=".3f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Pearson Correlation")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAILURE PREDICTION PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Failure Prediction Playground":
    st.markdown('<div class="section-title">XGBoost Fail-Safe Prediction Scenario Analysis</div>', unsafe_allow_html=True)
    
    # Load model and test dataset
    best_meta = load_best_model_meta()
    model_path = MODELS_DIR / "best_model.joblib"
    test_path = DATA_PROCESSED / "test_data.csv"
    
    if not model_path.exists() or not test_path.exists():
        st.warning("⚠️ Champion model or Processed Test Data not found. Please run the pipeline first to generate them.")
        st.info("Run `python run_pipeline.py` in the project directory.")
    else:
        import joblib
        
        @st.cache_resource
        def load_pipeline_model(path):
            return joblib.load(path)
            
        @st.cache_data
        def load_test_data(path):
            return pd.read_csv(path)
            
        model = load_pipeline_model(model_path)
        test_df = load_test_data(test_path)
        
        # User selections
        col1, col2 = st.columns(2)
        with col1:
            selected_m = st.selectbox("Select Machine to Inspect", sorted(test_df["machineID"].unique()))
        
        machine_test = test_df[test_df["machineID"] == selected_m]
        
        with col2:
            # Let them select a timestamp
            timestamps = machine_test["datetime"].tolist()
            selected_ts = st.selectbox("Select Timestamp Record", list(timestamps), index=len(timestamps)-1)
            
        selected_row = machine_test[machine_test["datetime"] == selected_ts].iloc[0]
        
        st.markdown("### 🎛️ Adjust Telemetry Inputs (Scenario Simulation)")
        st.write("Modify the live telemetry values below to observe how the model's prediction changes in real-time:")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v_val = st.slider("Voltage (volt)", float(test_df["volt"].min()), float(test_df["volt"].max()), float(selected_row["volt"]), step=0.1)
        with c2:
            r_val = st.slider("Rotation (rotate)", float(test_df["rotate"].min()), float(test_df["rotate"].max()), float(selected_row["rotate"]), step=1.0)
        with c3:
            p_val = st.slider("Pressure (pressure)", float(test_df["pressure"].min()), float(test_df["pressure"].max()), float(selected_row["pressure"]), step=0.1)
        with c4:
            vib_val = st.slider("Vibration (vibration)", float(test_df["vibration"].min()), float(test_df["vibration"].max()), float(selected_row["vibration"]), step=0.1)
            
        # Update features in a copy of the row
        input_row = selected_row.copy()
        
        # Core values
        input_row["volt"] = v_val
        input_row["rotate"] = r_val
        input_row["pressure"] = p_val
        input_row["vibration"] = vib_val
        
        # Recalculate interaction metrics
        input_row["volt_rotate_ratio"] = v_val / (r_val + 1e-6)
        input_row["pressure_vibration_ratio"] = p_val / (vib_val + 1e-6)
        
        # Build the exact feature matrix the model expects
        feature_cols = [c for c in test_df.columns if c not in ["datetime", "machineID", "failure_label"]]
        
        input_df = pd.DataFrame([input_row[feature_cols]])
        
        # Run prediction
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        
        # Display Prediction Result
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("#### 🚨 Prediction Result")
            if pred == 1:
                st.markdown(
                    f'<div style="padding: 1.5rem; background-color: rgba(239, 68, 68, 0.12); border-left: 5px solid {COLORS["danger"]}; border-radius: 12px;">'
                    f'<h3 style="color: {COLORS["danger"]}; margin: 0 0 0.5rem 0; font-weight: 700;">⚠️ HIGH RISK: Failure Predicted</h3>'
                    f'<p style="margin: 0; color: #e2e8f0; font-size: 0.95rem;">The machine is highly likely to experience a component failure within the next 24 hours. Scheduled maintenance recommended immediately.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="padding: 1.5rem; background-color: rgba(34, 197, 94, 0.12); border-left: 5px solid {COLORS["success"]}; border-radius: 12px;">'
                    f'<h3 style="color: {COLORS["success"]}; margin: 0 0 0.5rem 0; font-weight: 700;">✅ SAFE: Normal Operation</h3>'
                    f'<p style="margin: 0; color: #e2e8f0; font-size: 0.95rem;">No failure predicted in the next 24 hours. The machine is operating within safe physical thresholds.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Show original failure label for comparison
            actual_label = int(selected_row["failure_label"])
            act_text = "Failure" if actual_label == 1 else "No Failure"
            act_color = COLORS["danger"] if actual_label == 1 else COLORS["success"]
            st.markdown(f"<p style='margin-top: 1rem; color: #94a3b8;'>Actual Record Label: <b style='color:{act_color}'>{act_text}</b></p>", unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("#### 📊 Failure Probability Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of Failure (%)", 'font': {'size': 14, 'color': '#94a3b8'}},
                number={'font': {'color': '#f8fafc', 'size': 36}, 'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': COLORS["primary"]},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    metrics_df = load_metrics()
    best_meta = load_best_model_meta()

    if metrics_df is None:
        st.warning("⚠️ Model metrics not found. Run the pipeline first: `python run_pipeline.py`")
        st.info("Once the pipeline completes, model_comparison.csv will appear in reports/metrics/")
    else:
        # Best Model Badge
        if best_meta:
            st.success(f"🏆 **Selected Best Model: {best_meta['model_name']}** (selected by {best_meta['primary_metric']})")

        # Metric Comparison Radar Chart
        st.markdown('<div class="section-title">XGBoost Model Diagnostics</div>', unsafe_allow_html=True)
        metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        available_metrics = [m for m in metric_cols if m in metrics_df.columns]

        fig = go.Figure()
        for i, row in metrics_df.iterrows():
            values = [row[m] for m in available_metrics]
            values.append(values[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill="toself",
                name=row["model"],
                line=dict(color=COLORS["gradient"][i % len(COLORS["gradient"])]),
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
            **PLOTLY_LAYOUT, height=480, title="Multi-Metric Radar Diagnostics",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar Chart Comparison
        st.markdown('<div class="section-title">Model Metrics Bar Chart</div>', unsafe_allow_html=True)
        melted = metrics_df.melt(id_vars=["model"], value_vars=available_metrics, var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=COLORS["gradient"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title="All Metrics for Selected Champion Model")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics Table
        st.markdown('<div class="section-title">Detailed Metrics Table</div>', unsafe_allow_html=True)
        display_df = metrics_df.copy()
        for col in available_metrics:
            display_df[col] = display_df[col].map("{:.4f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Saved Model Artifacts
    st.markdown('<div class="section-title">Model Artifacts</div>', unsafe_allow_html=True)
    if MODELS_DIR.exists():
        model_files = list(MODELS_DIR.glob("*.joblib"))
        if model_files:
            for mf in model_files:
                size_kb = mf.stat().st_size / 1024
                st.markdown(f"📦 `{mf.name}` — {size_kb:.1f} KB")
        else:
            st.info("No saved model files found yet.")
    else:
        st.info("Models directory does not exist yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality":
    report = load_validation_report()
    telemetry = load_telemetry()

    if report:
        st.markdown('<div class="section-title">Validation Report Summary</div>', unsafe_allow_html=True)
        summary = report.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Checks", summary.get("total_checks", 0))
        with c2:
            st.metric("✅ Passed", summary.get("passed", 0))
        with c3:
            st.metric("❌ Failed", summary.get("failed", 0))
        with c4:
            st.metric("⚠️ Warnings", summary.get("warnings", 0))

        # Detailed Results Table
        results = report.get("results", [])
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.info("No validation report found. Run the pipeline to generate one.")

    # ── DRIFT DETECTION ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Real-time Data Drift Detection</div>', unsafe_allow_html=True)
    train_path = DATA_PROCESSED / "train_data.csv"
    test_path = DATA_PROCESSED / "test_data.csv"
    
    if not train_path.exists() or not test_path.exists():
        st.info("Run the pipeline first to generate train/test datasets for drift analysis.")
    else:
        # Load datasets
        @st.cache_data
        def load_drift_data():
            ref = pd.read_csv(train_path)
            tar = pd.read_csv(test_path)
            return ref, tar
            
        ref_df, tar_df = load_drift_data()
        
        st.write("#### Simulate Production Environment Drift")
        st.write("Modify production telemetry values using the sliders to see statistical tests trigger drift alerts:")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            vib_shift = st.slider("Simulate Vibration Shift (%)", -50.0, 100.0, 0.0, step=5.0)
        with col_s2:
            rot_shift = st.slider("Simulate Rotation Shift (%)", -50.0, 100.0, 0.0, step=5.0)
            
        # Apply simulated drift
        simulated_tar = tar_df.copy()
        if vib_shift != 0.0:
            simulated_tar["vibration"] = simulated_tar["vibration"] * (1.0 + vib_shift / 100.0)
        if rot_shift != 0.0:
            simulated_tar["rotate"] = simulated_tar["rotate"] * (1.0 + rot_shift / 100.0)
            
        # Initialize detector
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector(ref_df)
        drift_report = detector.detect_drift(simulated_tar)
        
        # Display report
        drift_rows = []
        for feature, metrics in drift_report.items():
            status = "🔴 DRIFT DETECTED" if metrics["drift_detected"] else "✅ STABLE"
            drift_rows.append({
                "Feature": feature.upper(),
                "Status": status,
                "KS Stat": f"{metrics['ks_stat']:.4f}",
                "P-Value": f"{metrics['p_val']:.4e}",
                "Wasserstein Dist": f"{metrics['wasserstein_distance']:.4f}",
                "Baseline Mean": f"{metrics['ref_mean']:.2f}",
                "Simulated Mean": f"{metrics['tar_mean']:.2f}"
            })
            
        drift_rep_df = pd.DataFrame(drift_rows)
        st.dataframe(drift_rep_df, use_container_width=True, hide_index=True)
        
        # Plot overlapping distribution
        selected_feat = st.selectbox("Select Feature to Visualize Distribution Overlap", ["volt", "rotate", "pressure", "vibration"])
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ref_df[selected_feat], name="Baseline (Train)",
            opacity=0.6, marker_color=COLORS["primary"],
            nbinsx=50, histnorm="probability density"
        ))
        fig.add_trace(go.Histogram(
            x=simulated_tar[selected_feat], name="Production (Simulated)",
            opacity=0.6, marker_color=COLORS["danger"] if drift_report[selected_feat]["drift_detected"] else COLORS["success"],
            nbinsx=50, histnorm="probability density"
        ))
        
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution Comparison: Baseline vs Production ({selected_feat.upper()})",
            height=380,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

    # Null Analysis on raw telemetry
    if telemetry is not None:
        st.markdown('<div class="section-title">Raw Telemetry — Null Analysis</div>', unsafe_allow_html=True)
        null_counts = telemetry.isnull().sum()
        null_pct = (null_counts / len(telemetry) * 100).round(3)
        null_df = pd.DataFrame({"Column": null_counts.index, "Null Count": null_counts.values, "Null %": null_pct.values})
        null_df = null_df[null_df["Null Count"] > 0]

        if null_df.empty:
            st.success("✅ No null values in raw telemetry data!")
        else:
            fig = px.bar(null_df, x="Column", y="Null %", color_discrete_sequence=[COLORS["warning"]])
            fig.update_layout(**PLOTLY_LAYOUT, height=350, title="Columns with Missing Values")
            st.plotly_chart(fig, use_container_width=True)

        # Data Types
        st.markdown('<div class="section-title">Data Type Summary</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": telemetry.columns,
            "Type": telemetry.dtypes.astype(str).values,
            "Non-Null Count": telemetry.notnull().sum().values,
            "Unique Values": telemetry.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Pipeline Status":
    st.markdown('<div class="section-title">Pipeline Stage Status</div>', unsafe_allow_html=True)

    status = get_pipeline_status()
    stages = [
        ("1️⃣", "Data Loading & Validation", "Raw Data"),
        ("2️⃣", "Dataset Merging", "Merged Data"),
        ("3️⃣", "Data Cleaning", "Cleaned Data"),
        ("4️⃣", "Feature Engineering", "Features"),
        ("5️⃣", "Exploratory Data Analysis", "EDA Charts"),
        ("6️⃣", "Model Training & Evaluation", "Model Metrics"),
        ("7️⃣", "Best Model Persistence", "Best Model"),
    ]

    for icon, name, key in stages:
        ok = status.get(key, False)
        badge = "✅ Complete" if ok else "❌ Not Run"
        color = COLORS["success"] if ok else COLORS["danger"]
        st.markdown(
            f"""<div style="background: linear-gradient(145deg, #0b0f19, #131a2b);
            border-left: 4px solid {color}; border-radius: 8px; padding: 1rem 1.5rem;
            margin-bottom: 0.5rem;">
            <strong>{icon} {name}</strong>
            <span style="float:right; color:{color}; font-weight:600;">{badge}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    completed = sum(status.values())
    total = len(status)
    pct = completed / total * 100

    st.markdown("---")
    st.progress(completed / total)
    st.markdown(f"**Pipeline Completion: {completed}/{total} stages ({pct:.0f}%)**")

    # Log file viewer
    st.markdown('<div class="section-title">Recent Pipeline Logs</div>', unsafe_allow_html=True)
    if LOGS_DIR.exists():
        log_files = sorted(LOGS_DIR.glob("*.log"), reverse=True)
        if log_files:
            selected_log = st.selectbox("Select Log File", [f.name for f in log_files])
            log_path = LOGS_DIR / selected_log
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            n_lines = st.slider("Lines to display", 10, min(500, len(lines)), 50)
            st.code("".join(lines[-n_lines:]), language="log")
        else:
            st.info("No log files found.")
    else:
        st.info("Logs directory not found.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.8rem;">'
    'Smart Factory Predictive Maintenance Premium Observability Dashboard v3.0 • '
    'Built with Streamlit & Plotly • '
    f'© {datetime.now().year} Smart Factory PdM Team</p>',
    unsafe_allow_html=True,
)
