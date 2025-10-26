# AI Insight Dashboard (Local, Streamlit)

A one-file-to-run project that lets students, small businesses, and startups:
- Upload a dataset (CSV/XLSX)
- (Optionally) enter a prompt like “compare monthly revenue by region”
- Auto-clean data (duplicates, missing values)
- Auto-generate visual insights and a simple dashboard
- Export a static HTML report of the generated charts

Runs entirely on your PC with **no servers required**.


## Quick Start

1) **Install Python 3.10+** (https://www.python.org/downloads/).  
2) Open a terminal in this project folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open in your browser (usually at http://localhost:8501).

> **Optional AI features**: If you want the app to read your prompt and auto-design charts using a language model,
set the environment variable `OPENAI_API_KEY` before running the app. If not set, the app will still work with simple, built-in prompt parsing.

```bash
# Windows (PowerShell)
$Env:OPENAI_API_KEY="YOUR_KEY"
# macOS/Linux (bash/zsh)
export OPENAI_API_KEY="YOUR_KEY"
```

## What it does

- **Data cleaning**: drops duplicate rows, trims whitespace, tries to parse dates, and optionally fills missing numeric values.
- **Insight**: classifies columns (numeric, categorical, datetime), then generates recommended charts.
- **Prompt-driven dashboards**: Type a prompt such as “show total sales by month” or “scatter profit vs. marketing_spend by region.”
- **Export**: Click “Export HTML report” to save the current dashboard as a single HTML file.

## Ideal for

- **Students**: Quickly explore datasets for assignments.
- **Small businesses / startups**: Drag in your sales, marketing, or ops CSVs and get instant visuals.

## Files

```
app.py                     # Streamlit web app
utils.py                   # Helpers for cleaning & typing inference
dashboard_generator.py     # Simple prompt->charts logic (heuristics + optional LLM)
requirements.txt           # Python dependencies
sample_data/startup_sales.csv
```

## Notes

- The app keeps everything **local**. Your data stays on your machine.
- If your dataset is large (100MB+), it may take a few seconds to process.
- AI chart design is optional; without an API key, it uses a rule-based parser.
