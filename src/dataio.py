"""Data loading and saving utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


NUMERIC_COLS = [
    "price",
    "goals",
    "assists",
    "mins",
    "pens_scored",
    "pens_missed",
    "yc",
    "rc",
]


def load_csv(path: Path, config: Dict[str, str]) -> pd.DataFrame:
    """Load CSV and normalize column names."""
    df = pd.read_csv(path)
    mapping = config["columns"]
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.rename(columns={v: k for k, v in mapping.items()})
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
