"""Player pricing models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
import sqlalchemy


def _safe_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index avoiding conflicts when index name matches a column."""
    idx_name = df.index.name or "index"
    if idx_name in df.columns:
        return df.reset_index(drop=True)
    df = df.copy()
    df.index.name = idx_name
    return df.reset_index()


FEATURE_COLS = ["goals_per90", "assists_per90", "availability"]

DERIVED_CSV = Path("data/processed/derived_prices.csv")
DB_PATH = Path("data/processed/fanta.db")


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
        .pipe(_safe_reset_index)
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


def build_derived_prices() -> pd.DataFrame:
    """Construct derived prices DataFrame from processed stats and quotes."""
    from . import dataio, utils

    config = utils.load_config()
    processed = utils.resolve_path(config, "processed")
    stats_path = processed / "stats_master_with_weights.csv"
    quotes_path = processed / "quotes_2025_26_FVM_budget500.csv"
    stats = dataio.load_stats(str(stats_path))
    quotes = dataio.load_quotes(str(quotes_path))
    return train_price_model(stats, quotes, "linear")


def _upsert_derived_prices(
    df: pd.DataFrame, engine: sqlalchemy.engine.Engine, conflict_key: str = "id"
) -> None:
    """Upsert rows of ``df`` into ``derived_prices`` table using SQLite ON CONFLICT."""
    meta = MetaData()
    meta.reflect(bind=engine)
    if "derived_prices" not in meta.tables:
        df.to_sql("derived_prices", engine, if_exists="replace", index=False)
        return

    table = meta.tables["derived_prices"]
    cols = list(df.columns)
    update_cols = {
        c: getattr(sqlite_insert(table).excluded, c) for c in cols if c != conflict_key
    }
    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        for row in records:
            stmt = (
                sqlite_insert(table)
                .values(**row)
                .on_conflict_do_update(index_elements=[conflict_key], set_=update_cols)
            )
            conn.execute(stmt)


def train_derived_prices(overwrite: bool = False) -> pd.DataFrame:
    """Compute derived prices and persist them to SQLite and CSV."""
    df = build_derived_prices()
    if df.empty:
        raise RuntimeError("Derived prices vuoto: controlla le fasi precedenti della pipeline.")

    if "id" not in df.columns:
        key_cols = [c for c in ["player_id", "season"] if c in df.columns]
        if key_cols:
            df["id"] = df[key_cols].astype(str).agg("_".join, axis=1)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        if overwrite:
            conn.execute(text("DELETE FROM derived_prices"))

    if overwrite:
        df.to_sql("derived_prices", engine, if_exists="append", index=False)
    else:
        _upsert_derived_prices(df, engine, conflict_key="id")

    DERIVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DERIVED_CSV, index=False)
    return df
