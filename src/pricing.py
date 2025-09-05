"""Player pricing models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV


FEATURE_COLS = ["goals_per90", "assists_per90", "availability"]


def _normalize_budget(df: pd.DataFrame, ev: np.ndarray, budget: float) -> pd.DataFrame:
    total = ev.sum()
    factor = budget / total if total else 0
    out = df.copy()
    out["expected_value"] = ev
    out["fair_price"] = ev * factor
    return out


def baseline_linear(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    y = df["price"]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    ev = model.predict(X)
    return _normalize_budget(df, ev, budget)


def heuristic_price(df: pd.DataFrame, weights: Dict[str, float], budget: float) -> pd.DataFrame:
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


def train_price_model(
    stats: pd.DataFrame, quotes: pd.DataFrame, method: str = "linear"
) -> pd.DataFrame:
    """Train model to estimate fair prices from stats and quotes."""
    numeric_cols = [
        c
        for c in stats.select_dtypes(include="number").columns
        if c not in {"season", "season_weight"}
    ]
    weighted = stats.copy()
    weighted[numeric_cols] = weighted[numeric_cols].multiply(
        weighted["season_weight"], axis=0
    )
    agg = (
        weighted.groupby(["id", "name", "team", "role"])[numeric_cols]
        .sum()
        .reset_index()
    )
    df = agg.merge(quotes, on=["id", "name", "team", "role"], how="inner")
    X = df[numeric_cols]
    y = df["fvm"]

    if method == "linear":
        model = RidgeCV(alphas=np.logspace(-3, 3, 7))
        model.fit(X, y)
        summary = {col: coef for col, coef in zip(numeric_cols, model.coef_)}
    else:
        model = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=0)
        model.fit(X, y)
        summary = {
            col: imp for col, imp in zip(numeric_cols, model.feature_importances_)
        }

    df["estimated_price"] = model.predict(X)

    out_path = Path("data/outputs/price_model_summary.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for col, val in summary.items():
            fh.write(f"{col}: {val}\n")

    return df[
        [
            "id",
            "name",
            "team",
            "role",
            "fvm",
            "price_from_fvm_500",
            "estimated_price",
        ]
    ]
