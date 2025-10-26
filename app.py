import os
import io
import streamlit as st
import pandas as pd
import numpy as np
from utils import read_dataset, basic_clean, infer_column_types, safe_fillna
from dashboard_generator import build_figures, export_html

st.set_page_config(page_title="AI Insight Dashboard", page_icon="📊", layout="wide")

st.title("📊 AI Insight Dashboard — local & simple")
st.caption("Upload a dataset, type a short prompt, and auto-generate visuals. Ideal for students, small businesses, and startups.")

with st.sidebar:
    st.header("1) Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    st.markdown("Or try a sample dataset:")
    if st.button("Load startup_sales.csv (sample)"):
        file = io.BytesIO(open('sample_data/startup_sales.csv','rb').read())
        file.name = "startup_sales.csv"

    st.header("2) Options")
    fill_missing = st.checkbox("Fill missing numeric values (median)", value=True)
    export_button = st.button("Export HTML report")

    st.header("About")
    st.write("Runs locally. Your data stays on your computer.")

prompt = st.text_input("Describe what you want to see (e.g., 'show total revenue by month' or 'scatter profit vs marketing_spend by region')")

if not file:
    st.info("Upload a CSV/XLSX or click the sample button in the sidebar.")
    st.stop()

# Read & clean
df_raw = read_dataset(file)
df = basic_clean(df_raw.copy())
if fill_missing:
    df = safe_fillna(df, numeric_only=True)

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

# Build figures from prompt (or default suggestion)
effective_prompt = prompt.strip() or "show total of first numeric column by first category"
figs = build_figures(df, effective_prompt)

if not figs:
    st.warning("I couldn't infer a chart from the prompt. Try mentioning columns or chart types like 'bar', 'line', 'scatter'.")
else:
    st.subheader("Dashboard")
    for item in figs:
        st.plotly_chart(item["fig"], use_container_width=True)

# Export
if export_button:
    if not figs:
        st.error("No charts to export. Provide a prompt and ensure charts are visible first.")
    else:
        out = export_html(figs, "ai_insight_report.html", title="AI Insight Report")
        with open(out, "rb") as f:
            st.download_button("Download report", f, file_name="ai_insight_report.html", mime="text/html")
            st.success("Report ready! Click the download button above.")

st.markdown("---")
st.write("💡 **Tips**:")
st.write("- Try prompts like: 'line chart revenue over time', 'histogram of order_value', 'box profit by region', 'heatmap product x region'.")
st.write("- If your dates are strings, the app will try to parse them automatically.")
