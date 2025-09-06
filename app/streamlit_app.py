"""Streamlit UI for roster optimisation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src import dataio, services
from src.db import (
    engine,
    get_session,
    init_db,
    Player,
    upsert_players,
    mark_player_acquired,
    get_my_roster,
)
try:
    from src.db import list_searchable_players
except ImportError:
    list_searchable_players = None
from datetime import datetime


# --- Helper locali DB-bound ---
def _load_player(player_id: int):
    with get_session() as s:
        return s.get(Player, int(player_id))


def _mark_player_sold(player_id: int, price: int | None = None):
    with get_session() as s:
        p = s.get(Player, int(player_id))
        if not p:
            return False, "Player not found"
        if p.is_sold:
            return False, "Player already sold"
        p.is_sold = 1
        p.sold_price = int(price) if price is not None else None
        p.sold_at = datetime.utcnow()
        s.commit()
        return True, None


def _mark_player_unsold(player_id: int):
    with get_session() as s:
        p = s.get(Player, int(player_id))
        if not p:
            return False, "Player not found"
        p.is_sold = 0
        p.sold_price = None
        p.sold_at = None
        s.commit()
        return True, None


ROLE_ORDER = {"P": 1, "D": 2, "C": 3, "A": 4}


def _remove_from_my_roster(player_id: int):
    if not hasattr(Player, "my_acquired"):
        return False, "Roster tracking not supported"
    with get_session() as s:
        p = s.get(Player, int(player_id))
        if not p:
            return False, "Player not found"
        p.my_acquired = 0
        p.my_price = None
        p.my_acquired_at = None
        s.commit()
        return True, None


def _budget_stats(budget_total: int):
    if not hasattr(Player, "my_acquired"):
        return 0, int(budget_total)
    with get_session() as s:
        total = (
            s.query(Player)
            .filter(Player.my_acquired == 1)
            .with_entities(Player.my_price)
            .all()
        )
    spent = sum(int(p[0]) for p in total if p[0] is not None)
    residual = int(budget_total) - spent
    return spent, residual


@st.cache_data
def _list_my_roster():
    return get_my_roster()


@st.cache_data
def _budget_stats_cached(budget_total: int):
    return _budget_stats(budget_total)


# Fallback per list_searchable_players se assente nel DB
if list_searchable_players is not None:
    _list_players = list_searchable_players
else:
    def _list_players(q=None, role=None, team=None, include_sold=False):
        with get_session() as s:
            query = s.query(Player)
            if not include_sold:
                query = query.filter(Player.is_sold == 0)
            if q:
                like = f"%{q}%"
                query = query.filter(Player.name.like(like))
            if role:
                query = query.filter(Player.role == role)
            if team:
                query = query.filter(Player.team == team)
            return query.order_by(Player.name.asc()).all()


# init DB all'avvio
init_db()

DATA_PROCESSED = "data/processed"
OUTPUT_DIR = "data/outputs"
AUCTION_LOG = f"{OUTPUT_DIR}/auction_log.csv"

BASE_PROCESSED = {
    "quotes": f"{DATA_PROCESSED}/quotes_2025_26_FVM_budget500.csv",
    "stats": f"{DATA_PROCESSED}/stats_master_with_weights.csv",
    "gk": f"{DATA_PROCESSED}/goalkeepers_grid_matrix_square.csv",
}
REQUIRED_FILES = [BASE_PROCESSED["quotes"], BASE_PROCESSED["stats"]]
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
DISABLED = bool(missing_files)


def prepare_processed_data() -> None:
    raw_dir = Path("data/raw")
    for out_path in BASE_PROCESSED.values():
        out = Path(out_path)
        if out.exists():
            continue
        stem = out.stem
        src_xlsx = raw_dir / f"{stem}.xlsx"
        src_csv = raw_dir / f"{stem}.csv"
        if src_xlsx.exists():
            df = pd.read_excel(src_xlsx)
        elif src_csv.exists():
            df = pd.read_csv(src_csv)
        else:
            raise FileNotFoundError(f"Missing raw file for {stem}")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
st.sidebar.subheader("Data Setup")
if st.sidebar.button("Prepare processed data"):
    try:
        prepare_processed_data()
        st.sidebar.success("Processed data generated")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Failed to prepare data: {exc}")

if "confirm_reset" not in st.session_state:
    st.session_state["confirm_reset"] = False

if st.sidebar.button("Reset DB"):
    st.session_state["confirm_reset"] = True
if st.session_state["confirm_reset"]:
    st.sidebar.warning("This will delete the database file.")
    if st.sidebar.button("Confirm reset"):
        try:
            init_db(drop=True)
            st.sidebar.success("DB ricreato.")
        except SQLAlchemyError as exc:
            st.sidebar.error(str(exc))
        st.session_state["confirm_reset"] = False
        st.rerun()

if DISABLED:
    message = "Missing required data files:\n" + "\n".join(
        f"- {f}" for f in missing_files
    )
    st.warning(message)


@st.cache_data
def load_players() -> pd.DataFrame:
    quotes = f"{DATA_PROCESSED}/quotes_2025_26_FVM_budget500.csv"
    players = dataio.load_quotes(quotes)
    # expected_points may come from separate processing; default 0
    if "expected_points" not in players.columns:
        players["expected_points"] = 0.0
    return players


def read_log() -> pd.DataFrame:
    try:
        return pd.read_csv(AUCTION_LOG)
    except FileNotFoundError:
        return pd.DataFrame(columns=["id", "name", "team", "role", "price_paid", "acquired"])


def append_log(entry: dict) -> None:
    log = read_log()
    log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
    log.to_csv(AUCTION_LOG, index=False)


def process_log(player_id: int, price_paid: int, acquired: bool):
    """Handle auction logging and roster updates.

    Returns (ok, warning) where warning is a message when roster add failed.
    """
    ok, err = _mark_player_sold(int(player_id), int(price_paid))
    if not ok:
        return False, err
    warn = None
    if acquired:
        mark_player_acquired(int(player_id), int(price_paid))
    p = _load_player(int(player_id))
    append_log(
        {
            "id": p.id,
            "name": p.name,
            "team": p.team,
            "role": p.role,
            "price_paid": int(price_paid),
            "acquired": 1 if acquired else 0,
        }
    )
    return True, warn


players = load_players() if not DISABLED else pd.DataFrame()
if not players.empty:
    df = players.copy()
    df["fvm"] = pd.to_numeric(df.get("fvm"), errors="coerce")
    df["price_500"] = pd.to_numeric(df.get("price_500"), errors="coerce")
    df["expected_points"] = pd.to_numeric(df.get("expected_points"), errors="coerce")

    mask_missing = df["expected_points"].isna() | (df["expected_points"] <= 0)
    df.loc[mask_missing, "expected_points"] = df.loc[mask_missing, "fvm"].fillna(0)

    df = df.dropna(subset=["price_500"])
    df["price_500"] = df["price_500"].astype(int)
    df["fvm"] = df["fvm"].fillna(0).astype(int)
    df["expected_points"] = df["expected_points"].fillna(0.0).astype(float)

    rows = df[["id", "name", "team", "role", "fvm", "price_500", "expected_points"]].to_dict("records")
    try:
        upsert_players(rows)
    except SQLAlchemyError as exc:
        st.sidebar.error(f"Failed to upsert players: {exc}")
    players = df
st.title("Fantacalcio Roster Optimizer")

if DISABLED:
    st.selectbox("Search player", [], disabled=True)
    st.subheader("Auction log")
    with st.form("auction_form"):
        st.number_input("Price paid", min_value=0, step=1, disabled=True)
        st.checkbox("Acquired", value=True, disabled=True)
        st.form_submit_button("Log", disabled=True)
    st.subheader("Roster Optimizer")
    st.number_input("Total budget", value=500, disabled=True)
    st.number_input("Team cap", value=3, disabled=True)
    st.button("Optimize", disabled=True)
    st.sidebar.subheader("Il mio roster")
    st.sidebar.write({})
    st.sidebar.button("Esporta il mio roster", disabled=True)
    st.stop()

players["effective_price"] = players["price_500"].clip(lower=1)
players["value_score"] = players["expected_points"] / players["effective_price"]

log = read_log()

include_sold = st.checkbox("Mostra venduti", value=False)
search_players = _list_players(include_sold=include_sold)
name_to_id = {pl.name: pl.id for pl in search_players}
if name_to_id:
    name = st.selectbox("Search player", sorted(name_to_id.keys()))
    selected_id = name_to_id.get(name)
else:
    name = st.selectbox("Search player", [], disabled=True)
    selected_id = None

p = _load_player(int(selected_id)) if selected_id is not None else None

st.subheader("Player details")
if p is not None:
    value_score = float(p.expected_points) / max(int(p.price_500), 1)
    st.json(
        {
            "name": p.name,
            "team": p.team,
            "role": p.role,
            "price_500": int(p.price_500),
            "expected_points": float(p.expected_points),
            "value_score": round(value_score, 3),
            "status": "SOLD" if p.is_sold else "AVAILABLE",
            "sold_price": int(p.sold_price) if p.sold_price is not None else None,
        }
    )

    state = services.RosterState(
        budget_residual=500 - pd.to_numeric(log.get("price_paid", 0)).sum(),
        team_cap=3,
        team_counts=log["team"].value_counts().to_dict(),
        slots_needed={
            r: services.QUOTAS[r] - log["role"].value_counts().to_dict().get(r, 0)
            for r in services.QUOTAS
        },
        value_threshold=players["value_score"].quantile(0.6),
    )

    sel_row = pd.Series(
        {
            "id": p.id,
            "name": p.name,
            "team": p.team,
            "role": p.role,
            "expected_points": float(p.expected_points),
            "effective_price": max(int(p.price_500), 1),
            "value_score": value_score,
        }
    )
    rec = services.recommend_player(sel_row, state)
    st.markdown(f"**Recommendation:** {rec['label']} - {rec['reason']}")
else:
    st.warning("Player not found in DB.")

st.subheader("Auction log")
acquired = st.checkbox("Acquired", value=False)
price_paid = st.number_input("Price paid", min_value=0, step=1)
log_btn = st.button("Log", disabled=bool(p and p.is_sold))

if log_btn and selected_id is not None:
    ok, warn = process_log(int(selected_id), int(price_paid), acquired)
    if not ok:
        st.error(warn or "Unable to mark SOLD")
    else:
        if acquired:
            st.cache_data.clear()
            st.toast("Giocatore aggiunto al roster")
        if warn:
            st.warning(warn)
        st.rerun()

if p and p.is_sold and st.button("Undo (rimetti disponibile)"):
    ok, err = _mark_player_unsold(p.id)
    if ok:
        st.success("Player reso disponibile.")
        st.rerun()
    else:
        st.error(err or "Undo failed")


st.subheader("Roster Optimizer")
budget_total = st.number_input("Total budget", value=500)
team_cap = st.number_input("Team cap", value=3)
if st.button("Optimize"):
    roster = services.optimize_roster(players, log, budget_total, team_cap)
    roster.to_csv(f"{OUTPUT_DIR}/recommended_roster.csv", index=False)
    st.dataframe(
        roster[
            [
                "role",
                "name",
                "team",
                "expected_points",
                "effective_price",
                "value_score",
                "cum_budget",
            ]
        ]
    )

st.subheader("Il mio roster")

budget_total = st.session_state.get("budget_total")
if budget_total is None:
    budget_total = st.number_input("Budget totale", min_value=0, value=500, step=1)
    st.session_state["budget_total"] = budget_total
else:
    budget_total = st.number_input(
        "Budget totale", min_value=0, value=int(budget_total), step=1
    )
    st.session_state["budget_total"] = budget_total

my_players = _list_my_roster()

if not my_players:
    st.info("Nessun giocatore nel tuo roster.")
else:
    rows = []
    for p in my_players:
        value_score = float(p.expected_points) / max(int(p.price_500), 1)
        rows.append(
            {
                "Ruolo": p.role,
                "Giocatore": p.name,
                "Team": p.team,
                "Prezzo acquisto": int(p.my_price) if p.my_price is not None else 0,
                "price_500": int(p.price_500),
                "Expected Points": round(float(p.expected_points), 2),
                "Value Score": round(value_score, 3),
            }
        )
    df_roster = pd.DataFrame(rows)
    df_roster["__k__"] = df_roster["Ruolo"].map(lambda r: ROLE_ORDER.get(r, 99))
    df_roster = df_roster.sort_values(["__k__", "Giocatore"]).drop(columns="__k__")
    st.dataframe(df_roster, use_container_width=True)

    options = [
        (p.id, f"{p.name} ({p.role} - {p.team})") for p in my_players
    ]
    to_remove = st.multiselect(
        "Rimuovi dal mio roster (correzione typo)",
        options=options,
        format_func=lambda x: x[1] if isinstance(x, tuple) else x,
    )
    if st.button("Rimuovi selezionati", disabled=len(to_remove) == 0):
        errs: list[str] = []
        for item in to_remove:
            pid = item[0] if isinstance(item, tuple) else int(item)
            ok, err = _remove_from_my_roster(pid)
            if not ok:
                errs.append(f"{pid}: {err}")
        if errs:
            st.error("Alcuni elementi non sono stati rimossi:\n" + "\n".join(errs))
        else:
            st.success("Rimozione completata.")
            st.rerun()

spent, residual = _budget_stats_cached(int(st.session_state["budget_total"]))
st.metric("Speso", f"{spent}")
st.metric("Budget residuo", f"{residual}")

