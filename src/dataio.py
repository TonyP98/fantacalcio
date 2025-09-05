"""Data loading and saving utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

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


def load_quotes(quotes_path: str, derived_path: str | None = None) -> pd.DataFrame:
    """Load quotes and optional derived prices.

    Parameters
    ----------
    quotes_path:
        Path to ``quotes_2025_26_FVM_budget500.csv``.
    derived_path:
        Optional path to ``derived_prices.csv`` containing ``estimated_price``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``fvm``, ``price_500`` and ``estimated_price``.
    """

    quotes = pd.read_csv(quotes_path)
    cols = [c for c in ["id", "name", "team", "role", "fvm", "price_from_fvm_500"] if c in quotes.columns]
    df = quotes[cols].rename(columns={"price_from_fvm_500": "price_500"})

    df["fvm"] = pd.to_numeric(df["fvm"], errors="coerce").fillna(0)
    df["price_500"] = pd.to_numeric(df["price_500"], errors="coerce").fillna(0)

    if derived_path and Path(derived_path).exists():
        derived = pd.read_csv(derived_path)
        if "estimated_price" in derived.columns:
            derived["estimated_price"] = pd.to_numeric(
                derived["estimated_price"], errors="coerce"
            )
            df = df.merge(derived[["id", "estimated_price"]], on="id", how="left")
        else:
            df["estimated_price"] = np.nan
    else:
        df["estimated_price"] = np.nan

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
