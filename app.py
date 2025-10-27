# app.py  — final version

import os
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from utils import read_dataset, basic_clean, infer_column_types, safe_fillna
from dashboard_generator import build_figures, export_html

# -------------------------------
# Look & feel (safe if files exist)
# -------------------------------
st.set_page_config(
    page_title="DataLens AI — Insights for Everyone",
    page_icon="📈",
    layout="wide",
)

# Optional custom styles (only loads if styles.css exists)
css_path = Path("styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Resolve repo/app directory (so sample path works locally & in Streamlit Cloud)
APP_DIR = Path(__file__).parent
SAMPLE_PATH = APP_DIR / "sample_data" / "startup_sales.csv"

# -------------------------------
# Header
# -------------------------------
st.title("📈 DataLens AI — Insights for Everyone")
st.caption(
    "Upload a dataset, type a short prompt, and auto-generate visuals. "
    "Ideal for students, small businesses, and startups."
)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    # AI status indicator (uses Streamlit Secrets → OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        st.success("🧠 AI assist: ON")
    else:
        st.warning("AI assist: OFF (set OPENAI_API_KEY in Settings → Secrets)")

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
# Prompt
# -------------------------------
prompt = st.text_input(
    "Describe what you want to see (e.g., 'show total revenue by month' or "
    "'scatter profit vs marketing_spend by region')"
)

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
# Data preview
# -------------------------------
st.subheader("Data preview")
st.dataframe(df.head(100), use_container_width=True)

# Column type summary
numerics, cats, dts = infer_column_types(df)
with st.expander("Column types & basic stats"):
    st.write("**Numeric columns**:", numerics)
    st.write("**Categorical columns**:", cats)
    st.write("**Datetime columns**:", dts)
    st.write("**Shape**:", df.shape)
    st.write("**Missing values (per column)**:")
    st.write(df.isna().sum())

# -------------------------------
# Build figures from prompt (AI if available, else heuristics via build_figures)
# -------------------------------
effective_prompt = (prompt or "").strip() or "show total of first numeric column by first category"
figs = build_figures(df, effective_prompt)

if not figs:
    st.warning(
        "I couldn't infer a chart from the prompt. Try mentioning columns or chart types "
        "like 'bar', 'line', 'scatter', 'histogram', 'box', or 'heatmap'."
    )
else:
    st.subheader("Dashboard")
    for item in figs:
        st.plotly_chart(item["fig"], use_container_width=True)

# -------------------------------
# Export report
# -------------------------------
if export_button:
    if not figs:
        st.error("No charts to export. Provide a prompt and ensure charts are visible first.")
    else:
        out = export_html(figs, "ai_insight_report.html", title="AI Insight Report")
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
st.write("- Try prompts like: 'line chart revenue over time', 'histogram of order_value', 'box profit by region', 'heatmap product x region'.")
st.write("- If your dates are strings, the app will try to parse them automatically.")
