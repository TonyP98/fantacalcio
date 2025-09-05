"""Data loading and saving utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import re

import numpy as np
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


def load_quotes(path: str) -> pd.DataFrame:
    """Load processed quotes CSV and set default price column."""
    df = pd.read_csv(path)
    match = re.search(r"budget(\d+)", Path(path).stem)
    budget = int(match.group(1)) if match else None
    price_col = "price_from_fvm_500" if budget == 500 and "price_from_fvm_500" in df.columns else "fvm"
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    return df


def load_stats(path: str) -> pd.DataFrame:
    """Load processed stats CSV ensuring numeric columns are floats."""
    df = pd.read_csv(path)
    numeric_cols = [
        "season",
        "season_weight",
        "apps",
        "avg",
        "fanta_avg",
        "goals",
        "assists",
        "yc",
        "rc",
        "pens_scored",
        "pens_missed",
        "own_goals",
        "goals_conceded",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def load_goalkeeper_grid(path: str) -> pd.DataFrame:
    """Load goalkeeper grid as square matrix with NaN diagonal."""
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    np.fill_diagonal(df.values, np.nan)
    return df
