"""Ranking utilities using official price_500."""
from __future__ import annotations

import pandas as pd

from . import reco


def rank_players(
    df: pd.DataFrame,
    by: str,
    role: str,
    top: int,
    budget: float,
) -> pd.DataFrame:
    """Rank players by composite recommendation score.

    Parameters
    ----------
    df: pd.DataFrame
        Player data containing at least ``price_500`` (and optionally ``fanta_avg``).
    by: str
        Column to sort by (default ``score_z_role``).
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

    # Deduplicate potential multiple entries for the same player so that the
    # ranking list doesn't contain duplicates. Prefer the last occurrence to be
    # consistent with ``services.optimize_roster`` behaviour.
    if "id" in data.columns:
        data = data.drop_duplicates(subset=["id"], keep="last")
    elif {"name", "team"}.issubset(data.columns):
        data = data.drop_duplicates(subset=["name", "team"], keep="last")

    price = pd.to_numeric(data.get("price_500"), errors="coerce").fillna(0)
    data["effective_price"] = price.clip(lower=1)

    data = reco.compute_scores(data)

    sort_col = by if by in data.columns else "score_z_role"
    data = data.sort_values(sort_col, ascending=False)
    data["cum_price"] = data["effective_price"].cumsum()
    if budget:
        data = data[data["cum_price"] <= budget]
    return data.head(top)
