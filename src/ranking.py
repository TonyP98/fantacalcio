"""Ranking utilities using official price_500."""
from __future__ import annotations

import pandas as pd


def rank_players(
    df: pd.DataFrame,
    by: str,
    role: str,
    top: int,
    budget: float,
) -> pd.DataFrame:
    """Rank players by value score.

    Parameters
    ----------
    df: pd.DataFrame
        Player data containing ``price_500`` and ``expected_points``.
    by: str
        Column to sort by (default ``value_score``).
    role: str
        Filter by role (``ALL`` for no filter).
    top: int
        Number of rows to return.
    budget: float
        Optional budget constraint applied to cumulative price.
    """

    data = df.copy()
    if role != "ALL" and "role" in data.columns:
        data = data[data["role"] == role]

    price = pd.to_numeric(data.get("price_500"), errors="coerce").fillna(0)
    data["effective_price"] = price.clip(lower=1)
    expected = pd.to_numeric(data.get("expected_points"), errors="coerce").fillna(0)
    data["value_score"] = expected / data["effective_price"]

    sort_col = by if by in data.columns else "value_score"
    data = data.sort_values(sort_col, ascending=False)
    data["cum_price"] = data["effective_price"].cumsum()
    if budget:
        data = data[data["cum_price"] <= budget]
    return data.head(top)
