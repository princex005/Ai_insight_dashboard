# dashboard_generator.py — Professional visuals + AI planning + heuristics

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
# Plot styling helpers — make every chart look professional
# ───────────────────────────────────────────────────────────────────────────────
_COLORWAY = [
    "#7C83FD", "#5EEAD4", "#F59E0B", "#60A5FA", "#34D399",
    "#F472B6", "#A78BFA", "#F87171", "#22D3EE", "#FBBF24"
]

def _is_int_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_integer_dtype(s)
    except Exception:
        return False

def _is_float_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_float_dtype(s)
    except Exception:
        return False

def _style_fig(fig, x_title=None, y_title=None, title=None):
    """Apply one consistent, modern style to all figures."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title or fig.layout.title.text, x=0.02, xanchor="left",
                   font=dict(size=22, family="Inter, Segoe UI, sans-serif")),
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", family="Inter, Segoe UI, sans-serif"),
        colorway=_COLORWAY,
        hoverlabel=dict(bgcolor="#0E1117", bordercolor="#374151", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
    )
    fig.update_xaxes(title=x_title or fig.layout.xaxis.title.text,
                     gridcolor="#2A2F3A", zeroline=False)
    fig.update_yaxes(title=y_title or fig.layout.yaxis.title.text,
                     gridcolor="#2A2F3A", zeroline=False)
    return fig

def _format_y_ticks(fig, s: pd.Series):
    """Format y axis ticks based on dtype."""
    if _is_int_series(s):
        fig.update_yaxes(tickformat=",d")
    elif _is_float_series(s):
        fig.update_yaxes(tickformat=",.2f")

# ───────────────────────────────────────────────────────────────────────────────
# Minimal heuristic prompt parser (used when AI is off/unavailable)
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
    """Ask an LLM to propose a simple chart spec. Returns dict or None."""
    if not _HAS_OPENAI:
        return None

    cols = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
    system = (
        "You are a data visualization planner. "
        "Given a user request and available columns, return a JSON object with keys: "
        "chart_type (bar, line, scatter, histogram, box, heatmap), "
        "x (column or null), y (column or null), color (optional column or null). "
        "Pick only from the provided columns. Prefer datetime on x for line charts. "
        "Return ONLY valid JSON."
    )
    user = f"Columns: {cols}\nUser prompt: {prompt}\nReturn ONLY a JSON object."

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        spec = json.loads(content)
        spec["chart_type"] = str(spec.get("chart_type", "bar")).lower()
        for key in ("x", "y", "color"):  # sanitize columns
            v = spec.get(key)
            spec[key] = v if (isinstance(v, str) and v in df.columns) else None
        return spec
    except Exception:
        return None

def generate_ai_insights_text(prompt: str, df: pd.DataFrame) -> str:
    """Short narrative bullets (if AI available)."""
    if not _HAS_OPENAI:
        return "AI insights are available when OPENAI_API_KEY is set in app secrets."
    numerics, cats, dts = infer_column_types(df)
    schema = {"rows": len(df), "numerics": numerics[:8], "categoricals": cats[:8], "datetimes": dts[:8]}
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
    """Simple Q&A with schema + small sample only."""
    if not _HAS_OPENAI:
        return "AI Q&A is available when OPENAI_API_KEY is set."
    sample = df.head(20).to_dict(orient="records")
    numerics, cats, dts = infer_column_types(df)
    schema = {"numerics": numerics, "categoricals": cats, "datetimes": dts}
    system = ("You answer questions about a dataset using only the provided schema and sample rows. "
              "If uncertain, explain what calculation the user should run. Keep answers brief.")
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
# Figure builder — tries AI first, then falls back to improved heuristics
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
                fig = px.line(
                    df.sort_values(by=x),
                    x=x, y=y,
                    markers=True,
                    line_shape="spline",
                    title=f"Trend of {y} over {x}",
                )
                _format_y_ticks(fig, df[y])
                figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                return figures

            if chart == "scatter" and x and y:
                fig = px.scatter(
                    df, x=x, y=y, color=color,
                    opacity=0.8, title=f"Scatter: {y} vs {x}",
                )
                _format_y_ticks(fig, df[y])
                figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                return figures

            if chart == "histogram" and x:
                fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
                _format_y_ticks(fig, df[x] if x in df.columns else pd.Series(dtype=float))
                figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                return figures

            if chart == "box" and y:
                fig = px.box(
                    df, x=x, y=y, points="suspectedoutliers",
                    title=f"Box plot of {y} by {x}",
                )
                _format_y_ticks(fig, df[y])
                figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                return figures

            if chart == "heatmap":
                numerics, cats, _ = infer_column_types(df)
                if len(cats) >= 2 and numerics:
                    pivot = df.pivot_table(index=cats[0], columns=cats[1], values=numerics[0], aggfunc="mean")
                    fig = px.imshow(pivot, title=f"Heatmap: mean {numerics[0]} by {cats[0]} x {cats[1]}")
                    figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                    return figures

            # bar or fallback
            if x and y:
                # Smart aggregate & Top-10
                g = df.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(10)
                fig = px.bar(g, x=x, y=y, color=x, title=f"{y} by {x} (Top 10)")
                _format_y_ticks(fig, g[y])
                figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
                return figures

        except Exception:
            pass  # continue to heuristics

    # 2) Heuristic fallback (improved)
    chart_type = guess_chart_type(prompt)
    cols = extract_columns(prompt, df)

    if chart_type == "line":
        numerics, cats, dts = infer_column_types(df)
        x = dts[0] if dts else (cols[0] if cols else None)
        y = numerics[0] if numerics else (cols[1] if len(cols) > 1 else None)
        if x is not None and y is not None:
            fig = px.line(df.sort_values(by=x), x=x, y=y, markers=True, line_shape="spline",
                          title=f"Trend of {y} over {x}")
            _format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    elif chart_type == "scatter":
        if len(cols) >= 2:
            x, y = cols[0], cols[1]
            color = cols[2] if len(cols) > 2 else None
            fig = px.scatter(df, x=x, y=y, color=color, opacity=0.8,
                             title=f"Scatter: {y} vs {x}")
            _format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    elif chart_type == "histogram":
        numerics, _, _ = infer_column_types(df)
        x = numerics[0] if numerics else (cols[0] if cols else None)
        if x is not None:
            fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
            _format_y_ticks(fig, df[x])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    elif chart_type == "box":
        numerics, cats, _ = infer_column_types(df)
        y = numerics[0] if numerics else (cols[0] if cols else None)
        x = cats[0] if cats else (cols[1] if len(cols) > 1 else None)
        if y is not None:
            fig = px.box(df, x=x, y=y, points="suspectedoutliers",
                         title=f"Box plot of {y} by {x}")
            _format_y_ticks(fig, df[y])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    elif chart_type == "heatmap":
        numerics, cats, _ = infer_column_types(df)
        if len(cats) >= 2 and numerics:
            pivot = df.pivot_table(index=cats[0], columns=cats[1], values=numerics[0], aggfunc="mean")
            fig = px.imshow(pivot, title=f"Heatmap: mean {numerics[0]} by {cats[0]} x {cats[1]}")
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    else:  # "bar" default
        numerics, cats, _ = infer_column_types(df)
        if cats and numerics:
            x, y = cats[0], numerics[0]
            g = df.groupby(x, dropna=False)[y].sum().reset_index().sort_values(y, ascending=False).head(10)
            fig = px.bar(g, x=x, y=y, color=x, title=f"{y} by {x} (Top 10)")
            _format_y_ticks(fig, g[y])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})
        elif numerics:
            fig = px.bar(df.reset_index(), x="index", y=numerics[0], title=f"{numerics[0]} by index")
            _format_y_ticks(fig, df[numerics[0]])
            figures.append({"title": fig.layout.title.text, "fig": _style_fig(fig)})

    # Add an extra totals chart if prompt asks "total", "sum by", etc.
    p = (prompt or "").lower()
    if ("total" in p or "sum" in p or "aggregate" in p) and len(df.columns) >= 2:
        numerics, cats, _ = infer_column_types(df)
        if cats and numerics:
            g =
