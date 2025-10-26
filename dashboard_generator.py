import os
import json
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Optional
import re
import jinja2
import os

from utils import infer_column_types

# --- Minimal heuristic prompt parser ---
CHART_KEYWORDS = {
    "line": ["trend", "over time", "by month", "line", "timeseries", "weekly", "monthly", "yearly"],
    "bar": ["by", "per", "compare", "bar", "rank", "top", "bottom"],
    "scatter": ["vs", "relationship", "correlation", "scatter"],
    "histogram": ["distribution", "hist", "histogram", "frequency"],
    "box": ["spread", "quartile", "box", "outlier"],
    "heatmap": ["matrix", "pivot", "heatmap"]
}

def guess_chart_type(prompt: str) -> str:
    p = prompt.lower()
    for chart, kws in CHART_KEYWORDS.items():
        if any(k in p for k in kws):
            return chart
    # default
    return "bar"

def extract_columns(prompt: str, df: pd.DataFrame) -> List[str]:
    """Pick columns that are mentioned in the prompt; fallback to top 2 numeric + 1 category"""
    cols = []
    p = prompt.lower()
    for c in df.columns:
        if re.search(rf"\b{re.escape(str(c).lower())}\b", p):
            cols.append(c)

    if cols:
        return cols[:3]

    numerics, cats, dts = infer_column_types(df)
    if len(numerics) >= 2:
        cols = numerics[:2]
        if cats:
            cols.append(cats[0])
    elif numerics and cats:
        cols = [numerics[0], cats[0]]
    elif cats:
        cols = cats[:2]
    else:
        cols = df.columns[:2].tolist()
    return cols

def build_figures(df: pd.DataFrame, prompt: str) -> List[Dict[str, Any]]:
    chart_type = guess_chart_type(prompt)
    cols = extract_columns(prompt, df)

    figures = []
    if chart_type == "line":
        # Expect datetime + numeric; fallback to index
        numerics, cats, dts = infer_column_types(df)
        x = dts[0] if dts else (cols[0] if cols else None)
        y = numerics[0] if numerics else (cols[1] if len(cols) > 1 else None)
        if x is None or y is None:
            return figures
        fig = px.line(df.sort_values(by=x), x=x, y=y, title=f"Trend of {y} over {x}")
        figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "scatter":
        if len(cols) < 2:
            return figures
        x, y = cols[0], cols[1]
        color = cols[2] if len(cols) > 2 else None
        fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {y} vs {x}")
        figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "histogram":
        numerics, _, _ = infer_column_types(df)
        x = numerics[0] if numerics else (cols[0] if cols else None)
        if x is None:
            return figures
        fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
        figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "box":
        numerics, cats, _ = infer_column_types(df)
        y = numerics[0] if numerics else (cols[0] if cols else None)
        x = cats[0] if cats else (cols[1] if len(cols) > 1 else None)
        if y is None:
            return figures
        fig = px.box(df, x=x, y=y, points="suspectedoutliers", title=f"Box plot of {y} by {x}")
        figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "heatmap":
        # Pivot a categorical vs categorical by mean of first numeric
        numerics, cats, _ = infer_column_types(df)
        if len(cats) >= 2 and numerics:
            pivot = df.pivot_table(index=cats[0], columns=cats[1], values=numerics[0], aggfunc="mean")
            fig = px.imshow(pivot, title=f"Heatmap: mean {numerics[0]} by {cats[0]} x {cats[1]}")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    else:  # "bar" default
        numerics, cats, _ = infer_column_types(df)
        if cats and numerics:
            x, y = cats[0], numerics[0]
            fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
            figures.append({"title": fig.layout.title.text, "fig": fig})
        elif numerics:
            fig = px.bar(df.reset_index(), x="index", y=numerics[0], title=f"{numerics[0]} by index")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    # Add a totals bar if the prompt asks "total", "sum by", etc.
    p = prompt.lower()
    if ("total" in p or "sum" in p or "aggregate" in p) and len(df.columns) >= 2:
        numerics, cats, _ = infer_column_types(df)
        if cats and numerics:
            g = df.groupby(cats[0], dropna=False)[numerics[0]].sum().reset_index()
            fig2 = px.bar(g, x=cats[0], y=numerics[0], title=f"Total {numerics[0]} by {cats[0]}")
            figures.append({"title": fig2.layout.title.text, "fig": fig2})

    return figures

def export_html(figs: List[Dict[str, Any]], outfile: str, title: str = "AI Insight Report"):
    template = jinja2.Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>{{ title }}</h1>
  {% for item in figs %}
    <h2>{{ item.title }}</h2>
    {{ item.fig.to_html(full_html=False, include_plotlyjs=False) }}
  {% endfor %}
</body>
</html>
""")
    html = template.render(figs=figs, title=title)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile
