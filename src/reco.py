"""Recommendation scoring based on fanta_avg and price percentiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dataio


@dataclass
class RecConfig:
    """Thresholds for classification on z-scores."""

    buy_alpha: float = 1.0
    hold_alpha: float = -0.05


STATS_PATH_DEFAULT = Path("data/raw/stats_master_with_weights.csv")


def _zscore_role(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def _winsorize_role(s: pd.Series) -> pd.Series:
    low, high = s.quantile([0.01, 0.99])
    return s.clip(low, high)


def _merge_stats(
    players: pd.DataFrame,
    stats_df: Optional[pd.DataFrame],
    stats_path: Path,
) -> pd.DataFrame:
    if stats_df is None:
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        stats_df = dataio.load_stats(str(stats_path))

    stats = stats_df[["id", "name", "team", "fanta_avg"]].copy()

    out = players.merge(stats[["id", "fanta_avg"]], on="id", how="left")
    missing = out["fanta_avg"].isna()
    if missing.any():
        stats_alt = stats[["name", "team", "fanta_avg"]].copy()
        stats_alt["name"] = stats_alt["name"].str.strip().str.casefold()
        stats_alt["team"] = stats_alt["team"].str.strip().str.casefold()
        tmp = players.loc[missing, ["name", "team"]].copy()
        tmp["name"] = tmp["name"].str.strip().str.casefold()
        tmp["team"] = tmp["team"].str.strip().str.casefold()
        tmp = tmp.merge(stats_alt, on=["name", "team"], how="left")
        out.loc[missing, "fanta_avg"] = tmp["fanta_avg"].values
    return out


def compute_scores(
    players: pd.DataFrame,
    stats_df: pd.DataFrame | None = None,
    stats_path: Path | str = STATS_PATH_DEFAULT,
) -> pd.DataFrame:
    """Return dataframe with composite scores and z-scores per role."""

    df = players.copy()
    if "fanta_avg" not in df.columns:
        df = _merge_stats(df, stats_df, Path(stats_path))

    df["fanta_avg"] = pd.to_numeric(df.get("fanta_avg"), errors="coerce")
    df["price_500"] = pd.to_numeric(df.get("price_500"), errors="coerce")

    df = df.dropna(subset=["fanta_avg", "price_500"])

    df["fanta_avg"] = df.groupby("role")["fanta_avg"].transform(_winsorize_role)

    df["price_pct_role"] = (
        df.groupby("role")["price_500"].rank(pct=True, ascending=True)
    )
    df.loc[df["price_500"] <= 0, "price_pct_role"] = 0.0

    df["score_raw"] = 0.70 * df["fanta_avg"] + 0.30 * df["price_pct_role"]

    df["score_z_role"] = df.groupby("role")["score_raw"].transform(_zscore_role)

    return df


def apply_recommendation(df: pd.DataFrame, cfg: RecConfig) -> pd.DataFrame:
    """Apply BUY/HOLD/AVOID labels based on z-scores and config thresholds."""

    out = df.copy()
    out["Recommendation"] = np.select(
        [
            out["score_z_role"] >= cfg.buy_alpha,
            out["score_z_role"] >= cfg.hold_alpha,
        ],
        ["BUY", "HOLD"],
        default="AVOID",
    )
    out["BUY_alpha"] = cfg.buy_alpha
    out["HOLD_alpha"] = cfg.hold_alpha
    return out

