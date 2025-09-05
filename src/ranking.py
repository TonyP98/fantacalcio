"""Ranking utilities."""
from __future__ import annotations

import pandas as pd


def rank_players(
    df: pd.DataFrame,
    by: str,
    role: str,
    top: int,
    budget: float,
) -> pd.DataFrame:
    data = df.copy()
    if role != "ALL":
        data = data[data["role"] == role]
    data["value"] = data["expected_value"] / data["fair_price"]
    sort_col = by if by in data.columns else "value"
    data = data.sort_values(sort_col, ascending=False)
    data["cum_price"] = data["fair_price"].cumsum()
    if budget:
        data = data[data["cum_price"] <= budget]
    return data.head(top)
