# dashboard_generator.py — final version (AI + heuristics)

from typing import List, Dict, Any
import os, re, json
import pandas as pd
import plotly.express as px
import jinja2

from utils import infer_column_types

# ───────────────────────────────────────────────────────────────────────────────
# Optional LLM client (auto-disabled if OPENAI_API_KEY is not set)
# ───────────────────────────────────────────────────────────────────────────────
_HAS_OPENAI = False
_client = None
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _client = OpenAI()
        _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    _client = None


# ───────────────────────────────────────────────────────────────────────────────
# Minimal heuristic prompt parser (fallback when AI is off/unavailable)
# ───────────────────────────────────────────────────────────────────────────────
CHART_KEYWORDS = {
    "line": ["trend", "over time", "by month", "line", "timeseries", "weekly", "monthly", "yearly"],
    "bar": ["by", "per", "compare", "bar", "rank", "top", "bottom"],
    "scatter": ["vs", "relationship", "correlation", "scatter"],
    "histogram": ["distribution", "hist", "histogram", "frequency"],
    "box": ["spread", "quartile", "box", "outlier"],
    "heatmap": ["matrix", "pivot", "heatmap"],
}

def guess_chart_type(prompt: str) -> str:
    p = (prompt or "").lower()
    for chart, kws in CHART_KEYWORDS.items():
        if any(k in p for k in kws):
            return chart
    return "bar"  # default


def extract_columns(prompt: str, df: pd.DataFrame) -> List[str]:
    """Pick columns mentioned in the prompt; fallback to top 2 numeric + 1 category."""
    cols: List[str] = []
    p = (prompt or "").lower()
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


# ───────────────────────────────────────────────────────────────────────────────
# AI helpers
# ───────────────────────────────────────────────────────────────────────────────
def llm_chart_spec(prompt: str, df: pd.DataFrame):
    """
    Ask an LLM to propose a simple chart spec.
    Returns a dict like: {"chart_type":"bar","x":"region","y":"revenue","color":"product"} or None.
    """
    if not _HAS_OPENAI:
        return None

    cols = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
    system = (
        "You are a data visualization planner. "
        "Given a user request and available columns, return a JSON object with keys: "
        "chart_type (one of: bar, line, scatter, histogram, box, heatmap), "
        "x (column name or null), y (column name or null), color (optional column or null). "
        "Pick only from the provided columns. Prefer datetime on x for line charts. "
        "Return ONLY valid JSON."
    )
    user = f"Columns: {cols}\nUser prompt: {prompt}\nReturn ONLY a JSON object."

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        spec = json.loads(content)
        spec["chart_type"] = str(spec.get("chart_type", "bar")).lower()

        # sanitize columns
        for key in ("x", "y", "color"):
            v = spec.get(key)
            spec[key] = v if (isinstance(v, str) and v in df.columns) else None
        return spec
    except Exception:
        return None


def generate_ai_insights_text(prompt: str, df: pd.DataFrame) -> str:
    """Short narrative bullets about the dataset, if AI is available."""
    if not _HAS_OPENAI:
        return "AI insights are available when OPENAI_API_KEY is set in app secrets."
    numerics, cats, dts = infer_column_types(df)
    schema = {
        "rows": len(df),
        "numerics": numerics[:8],
        "categoricals": cats[:8],
        "datetimes": dts[:8],
    }
    system = "You write short, factual data insights. Provide 3–6 concise bullet points. No code."
    user = f"Schema: {schema}\nUser intent: {prompt}\nWrite short bullets with potential trends or comparisons to check."

    try:
        r = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "AI insights temporarily unavailable."


def answer_question_about_df(question: str, df: pd.DataFrame) -> str:
    """
    Simple Q&A over the dataset using only the schema and a small sample.
    This avoids sending full data; suitable for lightweight guidance.
    """
    if not _HAS_OPENAI:
        return "AI Q&A is available when OPENAI_API_KEY is set."

    sample = df.head(20).to_dict(orient="records")
    numerics, cats, dts = infer_column_types(df)
    schema = {"numerics": numerics, "categoricals": cats, "datetimes": dts}

    system = (
        "You answer questions about a dataset using only the provided schema and sample rows. "
        "If uncertain, explain what calculation the user should run. Keep answers brief."
    )
    user = f"Schema: {schema}\nSample rows (first 20): {sample}\nQuestion: {question}"

    try:
        r = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I couldn't answer that just now."


# ───────────────────────────────────────────────────────────────────────────────
# Figure builder — tries AI first, then falls back to heuristics
# ───────────────────────────────────────────────────────────────────────────────
def build_figures(df: pd.DataFrame, prompt: str) -> List[Dict[str, Any]]:
    figures: List[Dict[str, Any]] = []

    # 1) Try AI planning (if available)
    spec = llm_chart_spec(prompt, df)
    if spec:
        try:
            chart = spec.get("chart_type", "bar")
            x, y, color = spec.get("x"), spec.get("y"), spec.get("color")

            if chart == "line" and x and y:
                fig = px.line(df.sort_values(by=x), x=x, y=y, title=f"Trend of {y} over {x}")
                return [{"title": fig.layout.title.text, "fig": fig}]

            if chart == "scatter" and x and y:
                fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {y} vs {x}")
                return [{"title": fig.layout.title.text, "fig": fig}]

            if chart == "histogram" and x:
                fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
                return [{"title": fig.layout.title.text, "fig": fig}]

            if chart == "box" and y:
                fig = px.box(df, x=x, y=y, points="suspectedoutliers", title=f"Box plot of {y} by {x}")
                return [{"title": fig.layout.title.text, "fig": fig}]

            if chart == "heatmap":
                numerics, cats, _ = infer_column_types(df)
                if len(cats) >= 2 and numerics:
                    pivot = df.pivot_table(index=cats[0], columns=cats[1], values=numerics[0], aggfunc="mean")
                    fig = px.imshow(pivot, title=f"Heatmap: mean {numerics[0]} by {cats[0]} x {cats[1]}")
                    return [{"title": fig.layout.title.text, "fig": fig}]

            # bar or fallback
            if x and y:
                fig = px.bar(df, x=x, y=y, color=color, title=f"{y} by {x}")
                return [{"title": fig.layout.title.text, "fig": fig}]
        except Exception:
            pass  # if AI fails, continue to heuristic path

    # 2) Heuristic fallback (your original logic)
    chart_type = guess_chart_type(prompt)
    cols = extract_columns(prompt, df)

    if chart_type == "line":
        numerics, cats, dts = infer_column_types(df)
        x = dts[0] if dts else (cols[0] if cols else None)
        y = numerics[0] if numerics else (cols[1] if len(cols) > 1 else None)
        if x is not None and y is not None:
            fig = px.line(df.sort_values(by=x), x=x, y=y, title=f"Trend of {y} over {x}")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "scatter":
        if len(cols) >= 2:
            x, y = cols[0], cols[1]
            color = cols[2] if len(cols) > 2 else None
            fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {y} vs {x}")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "histogram":
        numerics, _, _ = infer_column_types(df)
        x = numerics[0] if numerics else (cols[0] if cols else None)
        if x is not None:
            fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "box":
        numerics, cats, _ = infer_column_types(df)
        y = numerics[0] if numerics else (cols[0] if cols else None)
        x = cats[0] if cats else (cols[1] if len(cols) > 1 else None)
        if y is not None:
            fig = px.box(df, x=x, y=y, points="suspectedoutliers", title=f"Box plot of {y} by {x}")
            figures.append({"title": fig.layout.title.text, "fig": fig})

    elif chart_type == "heatmap":
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
    p = (prompt or "").lower()
    if ("total" in p or "sum" in p or "aggregate" in p) and len(df.columns) >= 2:
        numerics, cats, _ = infer_column_types(df)
        if cats and numerics:
            g = df.groupby(cats[0], dropna=False)[numerics[0]].sum().reset_index()
            fig2 = px.bar(g, x=cats[0], y=numerics[0], title=f"Total {numerics[0]} by {cats[0]}")
            figures.append({"title": fig2.layout.title.text, "fig": fig2})

    return figures


# ───────────────────────────────────────────────────────────────────────────────
# Export dashboard HTML
# ───────────────────────────────────────────────────────────────────────────────
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
