"""CSV handler utilities — read uploaded CSV and serialize results to CSV bytes."""

import io
import pandas as pd
from typing import List, Dict


REQUIRED_COLUMNS = {"company", "designation"}


def parse_csv(file_bytes: bytes) -> List[Dict]:
    """
    Parse uploaded CSV bytes.

    Expected columns: company, designation  (case-insensitive)
    Returns list of dicts with keys lowercase-stripped.
    Raises ValueError on schema mismatch.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows where both columns are empty
    df = df.dropna(subset=["company", "designation"], how="all")
    df["company"]     = df["company"].fillna("").astype(str).str.strip()
    df["designation"] = df["designation"].fillna("").astype(str).str.strip()
    df = df[(df["company"] != "") & (df["designation"] != "")]

    return df[["company", "designation"]].to_dict(orient="records")


def results_to_csv(results: List[Dict]) -> bytes:
    """
    Serialize a list of result dicts to CSV bytes.

    Expected keys per dict: company, designation, name, source, confidence, status
    """
    if not results:
        df = pd.DataFrame(columns=["company", "designation", "name", "source", "confidence", "status"])
    else:
        df = pd.DataFrame(results)
        # Ensure consistent column order
        cols = ["company", "designation", "name", "source", "confidence", "status"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]

    return df.to_csv(index=False).encode("utf-8")
