# ai_agent.py — tiny helpers for planning & insights (no API needed)
from typing import Tuple

def plan_chart(user_msg: str) -> Tuple[str, str]:
    text = (user_msg or "").strip()
    low = text.lower()
    if any(k in low for k in ["forecast", "predict", "projection"]):
        return "forecast", text
    if any(k in low for k in ["pie", "percentage", "share", "proportion"]):
        return "pie", text
    if any(k in low for k in ["line", "trend", "over time", "monthly", "weekly", "yearly", "date"]):
        return "line", text
    if any(k in low for k in ["scatter", "correlation", "relationship", "vs"]):
        return "scatter", text
    if any(k in low for k in ["hist", "histogram", "distribution"]):
        return "histogram", text
    if any(k in low for k in ["box", "outlier", "quartile"]):
        return "box", text
    if any(k in low for k in ["heatmap", "matrix", "pivot"]):
        return "heatmap", text
    return "bar", text

def simple_insight_text(title: str, df_head) -> str:
    return (
        f"• Chart: {title}\n"
        f"• Rows previewed: {len(df_head)}\n"
        f"• Tip: try 'average ... by ...', 'pie ... by ...', or 'forecast ...'."
    )
