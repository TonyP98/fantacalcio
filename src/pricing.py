from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

##
##  High-level goal
##  ----------------
##  Provide a robust, non-recursive training pipeline for "derived prices":
##   - strictly separate INPUTS vs OUTPUTS (no self-dependency on derived_prices.csv)
##   - write both CSV and SQLite table consistently
##   - be idempotent and safe (overwrite/append, deduplicate)
##

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

DB_PATH = PROCESSED_DIR / "fanta.db"
DERIVED_CSV = PROCESSED_DIR / "derived_prices.csv"

# True INPUTS (from README)
REQUIRED_INPUTS = [
    PROCESSED_DIR / "quotes_2025_26_FVM_budget500.csv",
    PROCESSED_DIR / "stats_master_with_weights.csv",
    PROCESSED_DIR / "goalkeepers_grid_matrix_square.csv",  # kept as hard dep to stay aligned with README
]


def _engine():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def list_missing_required_inputs() -> List[str]:
    """Return the list of missing *INPUT* files required to train.
    IMPORTANT: this must NOT include the output CSV `derived_prices.csv`."""
    missing = [str(p) for p in REQUIRED_INPUTS if not p.exists()]
    return missing


def reset_derived_prices_table() -> None:
    """Drop table and any related objects if present."""
    eng = _engine()
    with eng.begin() as cx:
        cx.execute(text("DROP TABLE IF EXISTS derived_prices"))
        # If you had indexes/views on this table, drop them here as well.


def debug_derived_prices_schema() -> str:
    """Return CREATE TABLE statement (or a note if table missing)."""
    eng = _engine()
    with eng.begin() as cx:
        res = cx.execute(
            text(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type='table' AND name='derived_prices'
                """
            )
        ).fetchone()
        if not res or not res[0]:
            return "-- table 'derived_prices' does not exist yet"
        return res[0]


# -------------------------------
#  Core training implementation
# -------------------------------

@dataclass
class TrainOutput:
    rows: int
    csv_path: str
    db_path: str
    trained_at: str


def _read_inputs() -> Dict[str, pd.DataFrame]:
    missing = list_missing_required_inputs()
    if missing:
        raise RuntimeError(
            "Input mancanti per il training: " + ", ".join(missing)
        )

    quotes = pd.read_csv(REQUIRED_INPUTS[0])
    stats = pd.read_csv(REQUIRED_INPUTS[1])
    # The GK grid is not used directly here, but we check for existence to keep logic aligned
    # with the README and future-proof the pipeline.
    _ = REQUIRED_INPUTS[2]

    return {"quotes": quotes, "stats": stats}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names predictable (lowercase, underscores)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _compute_expected_points(stats: pd.DataFrame) -> pd.Series:
    """
    Compute a simple, robust proxy for expected fantasy points.
    It is intentionally conservative and tolerant to missing columns.
    """
    s = pd.Series(0.0, index=stats.index)
    def safe(col: str) -> pd.Series:
        return stats[col] if col in stats.columns else 0.0

    # very simple scoring proxy, tweakable later
    s = (
        3.0 * safe("goals")
        + 1.0 * safe("assists")
        + 0.02 * safe("mins")
        - 0.5 * safe("yc")
        - 1.0 * safe("rc")
        + 1.0 * safe("pens_scored")
        - 1.5 * safe("pens_missed")
    )
    # Optional weights: if a 'weight' column exists, apply it
    if "weight" in stats.columns:
        s = s * stats["weight"].clip(lower=0)
    return s


def _blend_prices(quotes: pd.DataFrame) -> pd.Series:
    """
    Build an 'effective' price. If both columns exist we blend,
    otherwise we fallback to whichever is available.
    """
    has_est = "estimated_price" in quotes.columns
    has_p500 = "price_500" in quotes.columns
    if has_est and has_p500:
        # simple 60/40 blend, adjustable later or via config
        return 0.6 * quotes["estimated_price"].astype(float) + 0.4 * quotes["price_500"].astype(float)
    if has_est:
        return quotes["estimated_price"].astype(float)
    if has_p500:
        return quotes["price_500"].astype(float)
    # last resort: create a neutral vector so downstream doesn't explode
    return pd.Series(0.0, index=quotes.index)


def _fit_derived_prices(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    quotes = _normalize_columns(inputs["quotes"])
    stats = _normalize_columns(inputs["stats"])

    # Key for merge: we prioritize 'name', optionally add team/role if present.
    merge_keys: List[str] = ["name"]
    if "team" in quotes.columns and "team" in stats.columns:
        merge_keys.append("team")
    if "role" in quotes.columns and "role" in stats.columns:
        merge_keys.append("role")

    df = quotes.merge(stats, on=merge_keys, how="left", suffixes=("_q", "_s"))
    if df.empty:
        raise RuntimeError("Join vuoto tra quotes e stats: controlla i campi di chiave (name/team/role).")

    df["expected_points"] = _compute_expected_points(df)
    df["effective_price"] = _blend_prices(quotes)

    # Guard rail to avoid division by zero
    df["effective_price"] = df["effective_price"].replace(0, pd.NA).fillna(df["effective_price"].median())
    df["derived_value"] = (df["expected_points"] / df["effective_price"]).fillna(0.0)

    # A price suggestion, proportional to expected_points but anchored around typical budgets
    median_price = quotes["price_500"].median() if "price_500" in quotes.columns else 10.0
    df["derived_price"] = (df["expected_points"].clip(lower=0) / max(df["expected_points"].median(), 1.0)) * median_price
    df["derived_price"] = df["derived_price"].fillna(median_price).round(2)

    # Metadata
    df["trained_at"] = datetime.utcnow().isoformat(timespec="seconds")

    # Minimal, stable output columns (add more if needed)
    keep_cols = [c for c in [
        "name", "team", "role",
        "price_500" if "price_500" in quotes.columns else None,
        "estimated_price" if "estimated_price" in quotes.columns else None,
        "expected_points",
        "effective_price",
        "derived_value",
        "derived_price",
        "trained_at",
    ] if c is not None]

    return df[keep_cols]


def _write_outputs(df: pd.DataFrame, overwrite: bool) -> TrainOutput:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) CSV
    if overwrite or not DERIVED_CSV.exists():
        df.to_csv(DERIVED_CSV, index=False)
    else:
        # append with simple de-dup by (name, team, role)
        old = pd.read_csv(DERIVED_CSV)
        comb = pd.concat([old, df], ignore_index=True)
        subset = [c for c in ["name", "team", "role"] if c in comb.columns]
        if subset:
            comb = comb.drop_duplicates(subset=subset, keep="last")
        comb.to_csv(DERIVED_CSV, index=False)

    # 2) DB
    eng = _engine()
    if_exists = "replace" if overwrite else "append"
    with eng.begin() as cx:
        df.to_sql("derived_prices", con=cx, if_exists=if_exists, index=False)
        # De-dup in DB as well (if we appended)
        if if_exists == "append":
            # keep the latest trained_at per (name, team, role)
            subset_cols = [c for c in ["name","team","role"] if c in df.columns]
            if subset_cols:
                cols = ", ".join(subset_cols)
                order = "trained_at DESC"
                cx.execute(text(f"""
                    CREATE TEMP TABLE _dp AS
                    SELECT * FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY {cols} ORDER BY {order}) AS rn
                        FROM derived_prices
                    ) WHERE rn = 1;
                """))
                cx.execute(text("DELETE FROM derived_prices;"))
                cx.execute(text("INSERT INTO derived_prices SELECT * FROM _dp;"))
                cx.execute(text("DROP TABLE _dp;"))

    return TrainOutput(
        rows=int(len(df)),
        csv_path=str(DERIVED_CSV),
        db_path=str(DB_PATH),
        trained_at=datetime.utcnow().isoformat(timespec="seconds"),
    )


def train_derived_prices(overwrite: bool = True) -> Dict[str, str]:
    """
    Public entry-point used by Streamlit.
    **No recursion here.** It reads INPUTS -> fits -> writes OUTPUTS.
    """
    inputs = _read_inputs()          # may raise clean RuntimeError if inputs missing
    df = _fit_derived_prices(inputs) # compute model/logic
    out = _write_outputs(df, overwrite=overwrite)
    return {
        "rows": out.rows,
        "csv_path": out.csv_path,
        "db_path": out.db_path,
        "trained_at": out.trained_at,
    }


# -------------------------------
# Backward compatibility helpers
# -------------------------------
# If older parts of the app called helpers that *used to* recurse or
# expected derived_prices.csv as input, they can be shimmed here safely.


def ensure_trained() -> None:
    """Train if outputs are missing. This is *not* called from train_derived_prices."""
    if not DERIVED_CSV.exists():
        train_derived_prices(overwrite=True)
