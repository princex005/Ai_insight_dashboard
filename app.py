import os
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import read_dataset, basic_clean, infer_column_types, safe_fillna
from dashboard_generator import build_figures, export_html
from ai_agent import plan_chart, simple_insight_text

# -------------------------------
# Streamlit config + optional CSS
# -------------------------------
st.set_page_config(
    page_title="DataLens AI — Insights for Everyone",
    page_icon="📈",
    layout="wide",
)

css_path = Path("styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

APP_DIR = Path(__file__).parent
SAMPLE_PATH = APP_DIR / "sample_data" / "startup_sales.csv"

# -------------------------------
# Forecast helper (Prophet with fallback)
# -------------------------------
def quick_forecast(df, date_col, value_col, periods=12):
    """
    Try Prophet for time-series forecast. If unavailable, use a simple moving-average extension.
    Returns a dataframe that includes forecast columns when Prophet is available.
    """
    try:
        from prophet import Prophet
        d = df[[date_col, value_col]].dropna().copy()
        if d.empty:
            return d
        d = d.sort_values(date_col).rename(columns={date_col: "ds", value_col: "y"})
        m = Prophet(
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.fit(d)
        future = m.make_future_dataframe(periods=periods, freq="MS")
        fc = m.predict(future)
        out = d.merge(fc[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="right")
        out.rename(columns={"ds": date_col, "y": value_col}, inplace=True)
        return out
    except Exception:
        # Fallback: naive moving-average extension (keeps the app usable even without Prophet)
        d = df[[date_col, value_col]].dropna().sort_values(date_col).copy()
        if d.empty:
            return d
        # guess a step; if zero, default to ~monthly
        try:
            step = (d[date_col].iloc[-1] - d[date_col].iloc[0]) / max(len(d) - 1, 1)
        except Exception:
            step = pd.Timedelta(days=30)
        avg = d[value_col].rolling(3, min_periods=1).mean().iloc[-1]
        rows = []
        last_date = d[date_col].iloc[-1]
        for _ in range(periods):
            last_date = last_date + (step if step != 0 else pd.Timedelta(days=30))
            rows.append({date_col: last_date, value_col: avg})
        return pd.concat([d, pd.DataFrame(rows)], ignore_index=True)

# -------------------------------
# Header
# -------------------------------
st.title("📈 DataLens AI — Insights for Everyone")
st.caption(
    "Upload a dataset, type a short prompt or chat in plain English, and auto-generate visuals. "
    "Ideal for students, small businesses, and startups."
)

# -------------------------------
# Sidebar (upload + options)
# -------------------------------
with st.sidebar:
    if os.getenv("OPENAI_API_KEY"):
        st.success("🧠 AI assist: ON")
    else:
        st.info("AI assist: OFF (set OPENAI_API_KEY in Settings → Secrets for richer AI)")

    st.header("1) Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

    st.markdown("Or try a sample dataset:")
    if st.button("Load startup_sales.csv (sample)"):
        try:
            file = io.BytesIO(SAMPLE_PATH.read_bytes())
            file.name = "startup_sales.csv"
        except Exception as e:
            st.error(f"Couldn't load sample file: {e}")

    st.header("2) Options")
    fill_missing = st.checkbox("Fill missing numeric values (median)", value=True)
    export_button = st.button("Export HTML report")

    st.header("About")
    st.write("Runs locally or in the cloud. Your uploaded data stays within the app session.")

# -------------------------------
# Guard: need a file
# -------------------------------
if not file:
    st.info("Upload a CSV/XLSX or click the sample button in the sidebar.")
    st.stop()

# -------------------------------
# Read & clean
# -------------------------------
df_raw = read_dataset(file)
df = basic_clean(df_raw.copy())
if fill_missing:
    df = safe_fillna(df, numeric_only=True)

# -------------------------------
# Data preview + dtypes
# -------------------------------
st.subheader("Data preview")
st.dataframe(df.head(100), use_container_width=True)

numerics, cats, dts = infer_column_types(df)
with st.expander("Column types & basic stats"):
    st.write("**Numeric columns**:", numerics)
    st.write("**Categorical columns**:", cats)
    st.write("**Datetime columns**:", dts)
    st.write("**Shape**:", df.shape)
    st.write("**Missing values (per column)**:")
    st.write(df.isna().sum())

# -------------------------------
# Quick Filters (apply to a working copy)
# -------------------------------
df_work = df.copy()

# Categorical filters
if cats:
    choose_cats = st.multiselect("Filter by one or more categorical columns (optional)", cats, default=[])
    for c in choose_cats:
        vals = st.multiselect(f"Keep values in '{c}'", sorted(df_work[c].dropna().unique().tolist()))
        if vals:
            df_work = df_work[df_work[c].isin(vals)]

# Date filter (single date column support for now)
if dts:
    date_col = dts[0]
    min_d, max_d = df_work[date_col].min(), df_work[date_col].max()
    if pd.notna(min_d) and pd.notna(max_d) and min_d != max_d:
        dr = st.date_input("Date range", value=(min_d, max_d))
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            df_work = df_work[(df_work[date_col] >= start) & (df_work[date_col] <= end)]

# Aggregation switch
agg_choice = st.radio("Aggregation for bar/heatmap", ["sum", "mean"], horizontal=True)

# -------------------------------
# KPI Cards (simple generic)
# -------------------------------
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)
try:
    if numerics:
        total_val = float(df_work[numerics[0]].sum())
        avg_val = float(df_work[numerics[0]].mean())
        max_val = float(df_work[numerics[0]].max())
        col1.metric(f"Total {numerics[0]}", f"{total_val:,.0f}")
        col2.metric(f"Average {numerics[0]}", f"{avg_val:,.2f}")
        col3.metric(f"Max {numerics[0]}", f"{max_val:,.2f}")
    else:
        col1.metric("Rows", len(df_work))
        col2.metric("Columns", df_work.shape[1])
        col3.metric("Missing cells", int(df_work.isna().sum().sum()))
except Exception:
    pass

# -------------------------------
# Modes: Prompt mode (classic) or Chat mode (new)
# -------------------------------
mode = st.radio("Choose mode", ["Prompt mode", "Chat mode"], horizontal=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"/"assistant", "text": str}]

user_messages = []  # prompts to execute

if mode == "Prompt mode":
    prompt = st.text_input(
        "Describe what you want to see (e.g., 'bar average age by gender' or 'pie percentage by Investment_Avenues')"
    )
    if prompt:
        user_messages = [prompt]  # multi-instruction handled below with ';'

else:
    st.subheader("Ask DataLens AI 💬")
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.chat_message("user").write(m["text"])
        else:
            st.chat_message("assistant").write(m["text"])

    chat_in = st.chat_input("Ask something like 'forecast revenue' or 'pie share by category'...")
    if chat_in:
        st.session_state.chat_history.append({"role": "user", "text": chat_in})
        user_messages = [chat_in]

# -------------------------------
# Turn inputs into instructions & render
# -------------------------------
instructions = []
if user_messages:
    for m in user_messages:
        # if user selected mean, lightly nudge the text with "average"
        if agg_choice == "mean" and "average" not in m.lower():
            m = m + "; average"
        instructions.extend([p.strip() for p in m.split(";") if p.strip()])

all_figs = []

for instr in instructions:
    chart_guess, cleaned = plan_chart(instr)

    # Special case: forecast
    if chart_guess == "forecast":
        nums, cats2, dts2 = infer_column_types(df_work)
        if dts2 and nums:
            dc, vc = dts2[0], nums[0]
            fc = quick_forecast(df_work, dc, vc, periods=12)
            if {"yhat", "yhat_lower", "yhat_upper"}.issubset(set(fc.columns)):
                fig = px.line(fc, x=dc, y=["yhat", "yhat_lower", "yhat_upper"], title=f"Forecast for {vc} over {dc}")
            else:
                # fallback plot (moving-average)
                fig = px.line(fc, x=dc, y=vc, title=f"Forecast (simple) for {vc} over {dc}")
            all_figs.append({"title": fig.layout.title.text, "fig": fig})
        continue

    # Normal charts via your generator
    these = build_figures(df_work, cleaned)
    all_figs.extend(these)

# Draw charts
if not all_figs:
    st.warning("I couldn't infer a chart. Try: 'bar average age by gender', 'pie percentage by Investment_Avenues', or 'forecast Mutual_Funds'.")
else:
    st.subheader("Dashboard")
    for item in all_figs:
        st.plotly_chart(item["fig"], use_container_width=True)

# -------------------------------
# Automatic insights (simple text for demo)
# -------------------------------
with st.expander("🔍 Automatic insights"):
    if all_figs:
        st.write(simple_insight_text(all_figs[0]["title"], df_work.head(12)))
    else:
        st.write("Generate at least one chart to see insights here.")

# -------------------------------
# Export report
# -------------------------------
if export_button:
    if not all_figs:
        st.error("No charts to export. Provide a prompt and ensure charts are visible first.")
    else:
        out = export_html(all_figs, "ai_insight_report.html", title="AI Insight Report")
        with open(out, "rb") as f:
            st.download_button(
                "Download report",
                f,
                file_name="ai_insight_report.html",
                mime="text/html",
            )
            st.success("Report ready! Click the download button above.")

# -------------------------------
# Tips
# -------------------------------
st.markdown("---")
st.write("💡 **Tips**:")
st.write("- Separate multiple requests with ';' like: `bar average age by gender; pie percentage by Investment_Avenues`.")
st.write("- Use Chat mode to type natural questions like `forecast revenue` or `show correlation between Debentures and Equity_Market`.")
st.write("- The 'Aggregation' switch controls sum vs mean for bars/heatmaps.")
st.write("- If your dates are strings, the app will try to parse them automatically.")
