import pandas as pd
import numpy as np
from dateutil.parser import parse

def read_dataset(file) -> pd.DataFrame:
    name = getattr(file, "name", "")
    if name.lower().endswith(".xlsx") or name.lower().endswith(".xls"):
        return pd.read_excel(file)
    return pd.read_csv(file)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Trim string cells
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Try to coerce potential datetime columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Heuristic: attempt parse if at least 50% look like dates
            sample = df[col].dropna().astype(str).head(50)
            score = sum(_looks_like_date(x) for x in sample) / max(len(sample), 1)
            if score >= 0.5:
                df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

def _looks_like_date(x: str) -> bool:
    try:
        parse(x, fuzzy=True)  # tolerant parse
        return True
    except Exception:
        return False

def infer_column_types(df: pd.DataFrame):
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    datetimes = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categoricals = [c for c in df.columns if c not in numerics and c not in datetimes]
    return numerics, categoricals, datetimes

def safe_fillna(df: pd.DataFrame, numeric_only=True) -> pd.DataFrame:
    out = df.copy()
    if numeric_only:
        for c in out.select_dtypes(include=["number"]).columns:
            out[c] = out[c].fillna(out[c].median())
    else:
        out = out.fillna(method="ffill").fillna(method="bfill")
    return out
