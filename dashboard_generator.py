# dashboard_generator.py — safe ASCII version (pie + mean/avg + clean styling)

from typing import List, Dict, Any
import re
import pandas as pd
import plotly.express as px
import jinja2

from utils import infer_column_types

# ------------------------------
# Helpers: parsing + styling
# ------------------------------

COLORWAY = [
    "#7C83FD", "#5EEAD4", "#F59E0B", "#60A5FA", "#34D399",
    "#F472B6", "#A78BFA", "#F87171", "#22D3EE", "#FBBF24"
]

CHART_KEYWORDS = {
    "line": ["trend", "over time", "by month", "line", "timeseries", "weekly", "monthly", "yearly"],
    "bar": ["by", "per", "compare", "bar", "rank", "top", "bottom", "total", "sum"],
    "scatter": ["vs", "relationship", "correlation", "scatter"],
    "histogram": ["distribution", "hist", "histogram", "frequency"],
    "box": ["spread", "quartile", "box", "outlier"],
    "heatmap": ["matrix", "pivot", "heatmap"],
    "pie": ["pie", "share", "percentage", "proportion", "ratio"]
}

def want_mean(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("average" in p) or ("avg" in p) or ("mean" in p)

def guess_chart_type(prompt: str) -> str:
    p = (prompt or "").lower()
    if ("pie" in p) or ("percentage" in p) or ("proportion" in p) or ("share" in p):
        return "pie"
    for chart, kws in CHART_KEYWORDS.items():
        if any(k in p for k in kws):
            return chart
    return "bar"

def extract_columns(prompt: str, df: pd.DataFrame) -> List[str]:
    """
    Return columns explicitly mentioned in the prompt.
    If none mentioned, fallback to [num,num,cat] or [num,cat] or [cat,cat].
    """
    p = (prompt or "").lower()
    cols: List[str] = []
    for c in df.columns:
        if re.search(r"\b" + re.escape(str(c).lower()) + r"\b", p):
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

def style_fig(fig, title=None, x_title=None, y_title=None):
    fig.update_layout(
        template="plotly_dark",
        colorway=COLORWAY,
        title=dict(text=(title or fig.layout.title.text), x=0.02, xanchor="left"),
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(color="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
    )
    fig.update_xaxes(title=(x_title or fig.layout.xaxis.title.text), gridcolor="#2A2F3A", zeroline=False)
    fig.update_yaxes(title=(y_title or fig.layout.yaxis.title.text), gridcolor="#2A2F3A", zeroline=False)
    return fig

def format_y_ticks(fig, series: pd.Series):
    if pd.api.types.is_integer_dtype(series):
        fig.update_yaxes(tickformat=",d")
    elif pd.api.types.is_float_dtype(series):
        fig.update_yaxes(tickformat=",.2f")

# ------------------------------
# Main: build_figures
# ------------------------------

def build_figures(df: pd.DataFrame, prompt: str) -> List[Dict[str, Any]]:
    figures: List[Dict[str, Any]] = []
    p = (prompt or "").lower()

    chart_type = guess_chart_type(prompt)
    cols = extract_columns(prompt, df)
    numerics, cats, dts = infer_column_types(df)
    use_mean = want_mean(prompt)

    if chart_type == "line":
        # prefer datetime on x
        x = dts[0] if dts else (cols[0] if cols else None)
        y = numerics[0] if numerics else (cols[1] if len(cols) > 1 else None)
        if x is not None and y is not None:
            fig = px.line(df.sort_values(by=x), x=x, y=y, markers=True, title=f"Trend of {y} over {x}")
            format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    elif chart_type == "scatter":
        if len(cols) >= 2:
            x, y = cols[0], cols[1]
            color = cols[2] if len(cols) > 2 and cols[2] in df.columns else None
            fig = px.scatter(df, x=x, y=y, color=color, opacity=0.85, title=f"Scatter: {y} vs {x}")
            if y in df.columns:
                format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    elif chart_type == "histogram":
        x = numerics[0] if numerics else (cols[0] if cols else None)
        if x is not None:
            fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
            format_y_ticks(fig, df[x])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    elif chart_type == "box":
        y = numerics[0] if numerics else (cols[0] if cols else None)
        x = cats[0] if cats else (cols[1] if len(cols) > 1 else None)
        if y is not None:
            fig = px.box(df, x=x, y=y, points="suspectedoutliers", title=f"Box plot of {y} by {x}")
            format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    elif chart_type == "heatmap":
        if len(cats) >= 2 and numerics:
            agg = "mean" if use_mean else "sum"
            pivot = df.pivot_table(index=cats[0], columns=cats[1], values=numerics[0], aggfunc=agg)
            fig = px.imshow(pivot, title=f"Heatmap: {agg} {numerics[0]} by {cats[0]} x {cats[1]}")
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    elif chart_type == "pie":
        cat = next((c for c in cols if c in cats), cats[0] if cats else None)
        if cat:
            num = next((c for c in cols if c in numerics), None)
            if num:
                agg = "mean" if use_mean else "sum"
                g = df.groupby(cat, dropna=False)[num].agg(agg).reset_index()
                fig = px.pie(g, names=cat, values=num, title=f"{agg.title()} {num} share by {cat}")
            else:
                g = df[cat].value_counts(dropna=False).reset_index()
                g.columns = [cat, "count"]
                fig = px.pie(g, names=cat, values="count", title=f"Share by {cat}")
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    else:  # bar default
        if cats and numerics:
            x = next((c for c in cols if c in cats), cats[0])
            y = next((c for c in cols if c in numerics), numerics[0])
            agg = "mean" if use_mean else "sum"
            g = (
                df.groupby(x, dropna=False)[y]
                .agg(agg).reset_index()
                .sort_values(y, ascending=False)
                .head(10)
            )
            fig = px.bar(g, x=x, y=y, color=x, title=f"{y} by {x} (Top 10, {agg})")
            format_y_ticks(fig, g[y])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})
        elif numerics:
            fig = px.bar(df.reset_index(), x="index", y=numerics[0], title=f"{numerics[0]} by index")
            format_y_ticks(fig, df[numerics[0]])
            figures.append({"title": fig.layout.title.text, "fig": style_fig(fig)})

    # Optional extra totals bar when user mentions total/sum/aggregate
    if (("total" in p) or ("sum" in p) or ("aggregate" in p)) and len(df.columns) >= 2:
        if cats and numerics:
            x2, y2 = cats[0], numerics[0]
            g2 = df.groupby(x2, dropna=False)[y2].sum().reset_index().sort_values(y2, ascending=False)
            fig2 = px.bar(g2, x=x2, y=y2, color=x2, title=f"Total {y2} by {x2}")
            format_y_ticks(fig2, g2[y2])
            figures.append({"title": fig2.layout.title.text, "fig": style_fig(fig2)})

    return figures

# ------------------------------
# Export: static HTML report
# ------------------------------

def export_html(figs: List[Dict[str, Any]], outfile: str, title: str = "AI Insight Report"):
    template = jinja2.Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; background: #0E1117; color:#E5E7EB; }
    h1, h2 { font-weight: 600; }
    .card { margin: 16px 0; padding: 8px 0; border-top: 1px solid #2A2F3A; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  {% for item in figs %}
    <div class="card">
      <h2>{{ item.title }}</h2>
      {{ item.fig.to_html(full_html=False, include_plotlyjs=False) }}
    </div>
  {% endfor %}
</body>
</html>
""")
    html = template.render(figs=figs, title=title)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile
