"""Pricing utilities.

The project previously supported multiple pricing strategies and a training
pipeline that produced *derived prices*.  All that logic has been removed and
the module now exposes only simple helpers used in tests: a baseline linear
model and an heuristic score.  Any effective price selection is handled in
``src.services`` which now always falls back to the official ``price_500``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Feature columns used by the lightweight models below
FEATURE_COLS = ["goals_per90", "assists_per90", "availability"]


def _normalize_budget(df: pd.DataFrame, ev: np.ndarray, budget: float) -> pd.DataFrame:
    """Normalise predicted values to match the given budget."""

    total = ev.sum()
    factor = budget / total if total else 0
    out = df.copy()
    out["expected_value"] = ev
    out["fair_price"] = ev * factor
    return out


def baseline_linear(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Simple linear model based on the ``FEATURE_COLS``."""

    X = df[FEATURE_COLS]
    y = df["price"]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    ev = model.predict(X)
    return _normalize_budget(df, ev, budget)


def heuristic_price(df: pd.DataFrame, weights: Dict[str, float], budget: float) -> pd.DataFrame:
    """Heuristic price based on basic counting stats and custom weights."""

    score = (
        df["goals"] * weights["gol"]
        + df["assists"] * weights["assist"]
        + df["yc"] * weights["amm"]
        + df["rc"] * weights["esp"]
        + df["pens_scored"] * weights["rigore_segnato"]
        + df["pens_missed"] * weights["rigore_sbagliato"]
    )
    ev = score.to_numpy()
    return _normalize_budget(df, ev, budget)

