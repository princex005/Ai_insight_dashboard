# ai_agent.py
import os
from typing import Tuple
import re

def has_openai() -> bool:
    return bool(os.getenv("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"))

# very tiny heuristic parser: returns ("bar"/"line"/"pie"/..., cleaned_prompt)
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
    # default
    return "bar", text

# very small insight summarizer (no LLM). You can swap with OpenAI later.
def simple_insight_text(title: str, df_head) -> str:
    return (
        f"• Chart: {title}\n"
        f"• Rows previewed: {len(df_head)}\n"
        f"• Tip: try `average ... by ...`, `pie ... by ...`, or `forecast ...`."
    )
