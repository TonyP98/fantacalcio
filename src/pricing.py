from __future__ import annotations
import os
from pathlib import Path
import logging
import inspect
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

FEATURE_COLS = ["goals_per90", "assists_per90", "availability"]

DERIVED_CSV = Path("data/processed/derived_prices.csv")
# Allinea al path visto nello UI: data/processed/fanta.db (override via env se serve)
DB_PATH = Path(os.getenv("FANTA_DB_PATH", "data/processed/fanta.db"))
TABLE_NAME = "derived_prices"
# Aggiorna la version per riflettere la patch di prevenzione ricorsione
MODULE_VERSION = "pricing-2025-09-05-5"
# Flag per evitare ricorsioni (re-entrance) in train_derived_prices
_TRAINING_DERIVED_PRICES = False

# ===== MONKEY-PATCH: reset_index() sicuro a livello GLOBALE =====
# Impedisce il classico ValueError di pandas quando il nome dell'indice
# coincide con una colonna già esistente (es. 'id').
_ORIG_RESET_INDEX = pd.DataFrame.reset_index

def _reset_index_safe_global(
    self: pd.DataFrame,
    level=None,
    drop: bool = False,
    inplace: bool = False,
    col_level: int = 0,
    col_fill: str = "",
):
    try:
        if not drop:
            idx = self.index
            if isinstance(idx, pd.MultiIndex):
                idx_names = list(idx.names)
            else:
                idx_names = [idx.name]
            candidate_names = [n for n in idx_names if n is not None]
            if not candidate_names:
                candidate_names = ["index"]
            if level is not None:
                if isinstance(level, (list, tuple)):
                    level_names = []
                    for lv in level:
                        if isinstance(idx, pd.MultiIndex):
                            name = idx.names[lv] if isinstance(lv, int) else lv
                        else:
                            name = idx.name if lv in (0, None) else lv
                        level_names.append(name if name is not None else "index")
                    candidate_names = [n if n is not None else "index" for n in level_names]
                else:
                    if isinstance(idx, pd.MultiIndex):
                        name = idx.names[level] if isinstance(level, int) else level
                    else:
                        name = idx.name if level in (0, None) else level
                    candidate_names = [name if name is not None else "index"]
            if any(name in self.columns for name in candidate_names):
                drop = True
    except Exception:
        pass
    return _ORIG_RESET_INDEX(
        self,
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )

pd.DataFrame.reset_index = _reset_index_safe_global
# ===== FINE MONKEY-PATCH =====


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


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_builder():
    """
    Seleziona una funzione *pura* che costruisce i derived prices (niente I/O DB e
    soprattutto nessuna chiamata a train_derived_prices, per evitare ricorsioni).
    Ordine di ricerca:
      1) locale: build_derived_prices / compute_derived_prices
      2) src.services: build_derived_prices / compute_derived_prices
    Vengono scartati i candidati che nel body contengono 'train_derived_prices('.
    """

    def valid_builder(fn):
        # Rileva riferimenti diretti o alias a train_derived_prices
        try:
            globs = fn.__globals__
            for name in fn.__code__.co_names:
                if globs.get(name) is train_derived_prices:
                    return False
        except Exception:
            pass
        try:
            src = inspect.getsource(fn)
        except OSError:
            # se non recupero il sorgente, accetto comunque
            return True
        # scarta builder che chiamano train_derived_prices (anche via modulo)
        return "train_derived_prices(" not in src

    # 1) candidati locali
    chosen = None
    g = globals()
    for name in ("build_derived_prices", "compute_derived_prices"):
        if name in g and callable(g[name]) and valid_builder(g[name]):
            chosen = g[name]
            break

    # 2) candidati in services.*
    if chosen is None:
        try:
            from .services import build_derived_prices as b1  # type: ignore
            if callable(b1) and valid_builder(b1):
                chosen = b1
        except Exception:
            pass
    if chosen is None:
        try:
            from .services import compute_derived_prices as b2  # type: ignore
            if callable(b2) and valid_builder(b2):
                chosen = b2
        except Exception:
            pass

    if chosen is None:
        raise RuntimeError(
            "Non trovo una funzione 'pura' per calcolare i derived prices. "
            "Definisci build_derived_prices()/compute_derived_prices() che NON chiami "
            "train_derived_prices(), oppure mettila in src/services.py."
        )

    # log diagnostico (si vede nella caption già presente in Streamlit)
    try:
        origin = inspect.getsourcefile(chosen) or "<?>"
        logging.info("Derived-prices builder scelto: %s @ %s", chosen.__name__, origin)
    except Exception:
        pass
    return chosen


def _drop_table_and_indexes(engine, table_name: str) -> None:
    """Drop tabella e relativi indici (inclusi UNIQUE) così l'overwrite è davvero pulito."""
    with engine.begin() as conn:
        idx = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t"),
            {"t": table_name},
        ).fetchall()
        for (idx_name,) in idx:
            conn.execute(text(f"DROP INDEX IF EXISTS {idx_name}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))


def debug_schema() -> str:
    """Ritorna DDL della tabella e indici per diagnosi veloce in UI."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT type,name,sql FROM sqlite_master "
                "WHERE (name=:t) OR (type='index' AND tbl_name=:t)"
            ),
            {"t": TABLE_NAME},
        ).fetchall()
    if not rows:
        return f"-- {TABLE_NAME} non esiste su {DB_PATH}"
    return "\n\n".join(f"-- {t} {n}\n{sql}" for t, n, sql in rows if sql)


def reset_derived_prices_table() -> None:
    """API esplicita di reset per Streamlit."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    _drop_table_and_indexes(engine, TABLE_NAME)


def _upsert_df(df: pd.DataFrame, engine, table_name: str, conflict_key: str = "id") -> None:
    meta = MetaData()
    meta.reflect(bind=engine)
    if table_name not in meta.tables:
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        return
    table = meta.tables[table_name]
    cols = list(df.columns)
    set_clause = {c: getattr(sqlite_insert(table).excluded, c) for c in cols if c != conflict_key}
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        for row in rows:
            stmt = (
                sqlite_insert(table)
                .values(**row)
                .on_conflict_do_update(index_elements=[conflict_key], set_=set_clause)
            )
            conn.execute(stmt)


def train_derived_prices(overwrite: bool = False) -> pd.DataFrame:
    global _TRAINING_DERIVED_PRICES
    if _TRAINING_DERIVED_PRICES:
        raise RuntimeError("train_derived_prices() chiamata in modo ricorsivo")
    _TRAINING_DERIVED_PRICES = True
    try:
        builder = _load_builder()
        try:
            df = builder()
        except Exception as e:
            # Sovrapponiamo un messaggio chiaro ma lasciamo lo stack originale
            raise RuntimeError(
                f"Errore nel builder dei derived prices: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("La funzione di build ha restituito un DataFrame vuoto o invalido.")
    finally:
        _TRAINING_DERIVED_PRICES = False

    # Assicura una chiave 'id' stabile per l'UPSERT (se non già presente)
    if "id" not in df.columns:
        key_cols = [c for c in ("player_id", "season", "team_id") if c in df.columns]
        if key_cols:
            df["id"] = df[key_cols].astype(str).agg("_".join, axis=1)
        else:
            # fallback poco elegante ma sicuro (non deterministico tra run diversi)
            df = df.copy()
            df.insert(0, "id", range(1, len(df) + 1))

    # Pulizia finale robusta prima del salvataggio
    # 1) Evita nomi di colonna duplicati
    df = df.loc[:, ~df.columns.duplicated()]
    # 2) Deduplica eventuali id duplicati (causa comune dei conflitti logici)
    if "id" in df.columns:
        dups = int(df.duplicated(subset=["id"]).sum())
        if dups:
            logging.warning(
                "derived_prices: trovati %d id duplicati; conservo l'ultima occorrenza.",
                dups,
            )
            # Ordine di "ultima occorrenza": se esiste una colonna timestamp, usala
            order_col = "updated_at" if "updated_at" in df.columns else None
            if order_col:
                df = df.sort_values(order_col)
            df = df.drop_duplicates(subset=["id"], keep="last")

    engine = create_engine(f"sqlite:///{DB_PATH}")
    # se richiesto, resetto la tabella e gli indici per un inserimento "pulito"
    if overwrite:
        _drop_table_and_indexes(engine, TABLE_NAME)

    if overwrite:
        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    else:
        _upsert_df(df, engine, TABLE_NAME, conflict_key="id")

    _atomic_write_csv(df, DERIVED_CSV)
    return df
