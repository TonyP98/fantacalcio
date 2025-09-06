"""Streamlit UI for roster optimisation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src import dataio, services, db
from src.reco import RecConfig, compute_scores, apply_recommendation
from src.gk_grid import GKGrid, FULLNAME_BY_CODE
from src.db import (
    engine,
    get_session,
    init_db,
    Player,
    upsert_players,
    mark_player_acquired,
    get_my_roster,
    remove_from_roster,
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


@st.cache_data(show_spinner=False)
def _list_my_roster(version: int):
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

if "roster_version" not in st.session_state:
    st.session_state["roster_version"] = 0

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

# -- Recommendation tuning --
st.sidebar.subheader("Tuning")
cfg_buy_alpha = st.sidebar.number_input("BUY alpha", value=1.0, step=0.05)
cfg_hold_alpha = st.sidebar.number_input("HOLD alpha", value=-0.05, step=0.05)
rec_cfg = RecConfig(buy_alpha=cfg_buy_alpha, hold_alpha=cfg_hold_alpha)

if DISABLED:
    message = "Missing required data files:\n" + "\n".join(
        f"- {f}" for f in missing_files
    )
    st.warning(message)


@st.cache_data
def load_players() -> pd.DataFrame:
    quotes = f"{DATA_PROCESSED}/quotes_2025_26_FVM_budget500.csv"
    players = dataio.load_quotes(quotes)
    stats = dataio.load_stats(BASE_PROCESSED["stats"])
    players = players.merge(stats[["id", "fanta_avg"]], on="id", how="left")
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
    df["fanta_avg"] = pd.to_numeric(df.get("fanta_avg"), errors="coerce")

    df = df.dropna(subset=["price_500", "fanta_avg"])
    df["price_500"] = df["price_500"].astype(int)
    df["fvm"] = df["fvm"].fillna(0).astype(int)
    df["fanta_avg"] = df["fanta_avg"].astype(float)
    df["expected_points"] = 0.0

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

players = compute_scores(players)
players = apply_recommendation(players, rec_cfg)

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
    role = str(getattr(p, "role", "")).upper()
    team = getattr(p, "team", None)
    price_500 = getattr(p, "price_500", None)

    rec_label = None
    score_z = None
    row_match = players[players["id"] == p.id]
    if not row_match.empty:
        rec_label = row_match.iloc[0]["Recommendation"]
        score_z = row_match.iloc[0]["score_z_role"]

    details = {
        "name": p.name,
        "team": team,
        "role": role,
        "price_500": int(price_500) if price_500 is not None else None,
        "status": "SOLD" if p.is_sold else "AVAILABLE",
        "sold_price": int(p.sold_price) if p.sold_price is not None else None,
    }
    # Aggiungi Recommendation solo se non è un portiere
    if role != "P" and rec_label is not None:
        details["Recommendation"] = f"{rec_label}"
        details["score_z_role"] = round(float(score_z), 3) if score_z is not None else None

    if role == "P":
        grid = GKGrid()  # default: data/raw/goalkeepers_grid_matrix_square.csv
        if not grid.available:
            st.info(
                "⚠️ GK grid non trovato (atteso in data/raw/goalkeepers_grid_matrix_square.csv). "
                "Puoi impostare un path custom con la variabile d'ambiente GK_GRID_PATH."
            )
        else:
            best3 = grid.best_couples_pretty(str(team), top_n=3)
            details["best_couple"] = [f"{d['team']}: {d['score']:.3f}" for d in best3]

            roster = get_my_roster()
            owned_gk = next((pl for pl in roster if str(pl.role).upper() == "P"), None)
            if owned_gk:
                owned_gk_team = str(getattr(owned_gk, "team", "") or "")
                pair_score = grid.score_pair(owned_gk_team, str(team))
                owned_name = FULLNAME_BY_CODE.get(owned_gk_team.strip().upper(), owned_gk_team)
                cand_name = FULLNAME_BY_CODE.get(str(team).strip().upper(), str(team))
                details["owned_pair_score"] = f"{owned_name} ↔ {cand_name}: {pair_score:.3f}"

    st.json(details)
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
            st.session_state["roster_version"] += 1
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
    # 1) innesta gli acquisti dentro players (CSV o sessione)
    players_enriched = services.attach_my_roster(players, st)
    # 2) safety: tipi coerenti
    if "my_acquired" in players_enriched.columns:
        players_enriched["my_acquired"] = services._truthy(players_enriched["my_acquired"]).astype(int)
    if "my_price" in players_enriched.columns:
        players_enriched["my_price"] = pd.to_numeric(players_enriched["my_price"], errors="coerce")
    # 3) optimize sul DF arricchito
    roster = services.optimize_roster(players_enriched, st, budget_total, team_cap)
    roster.to_csv(f"{OUTPUT_DIR}/recommended_roster.csv", index=False)
    cols = ["role", "name", "team", "price_500", "score_raw", "score_z_role"]
    if "cum_budget" in roster.columns:
        cols.append("cum_budget")
    st.dataframe(roster[cols])

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

my_players = _list_my_roster(st.session_state["roster_version"])

if not my_players:
    st.info("Nessun giocatore nel tuo roster.")
else:
    rows = []
    for p in my_players:
        row_match = players[players["id"] == p.id]
        if row_match.empty:
            continue
        r = row_match.iloc[0]
        rows.append(
            {
                "Ruolo": p.role,
                "Giocatore": p.name,
                "Team": p.team,
                "Prezzo acquisto": int(p.my_price) if p.my_price is not None else 0,
                "price_500": int(p.price_500),
                "fanta_avg": round(float(r["fanta_avg"]), 2),
                "score_raw": round(float(r["score_raw"]), 3),
                "score_z_role": round(float(r["score_z_role"]), 3),
                "Recommendation": r["Recommendation"],
            }
        )
    df_roster = pd.DataFrame(rows)
    df_roster["__k__"] = df_roster["Ruolo"].map(lambda r: ROLE_ORDER.get(r, 99))
    df_roster = df_roster.sort_values(["__k__", "Giocatore"]).drop(columns="__k__")
    st.dataframe(df_roster, use_container_width=True)

    if st.button("Esporta il mio roster"):
        output_path = Path(OUTPUT_DIR) / "my_roster.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_roster.to_csv(output_path, index=False)
        st.success("Roster esportato")

    options = [
        (p.id, f"{p.name} ({p.role} - {p.team})") for p in my_players
    ]
    selected = st.multiselect(
        "Rimuovi dal mio roster (correzione typo)",
        options=options,
        format_func=lambda x: x[1] if isinstance(x, tuple) else x,
    )
    selected_ids = [item[0] if isinstance(item, tuple) else int(item) for item in selected]
    if st.button("Rimuovi selezionati", disabled=len(selected_ids) == 0):
        removed = remove_from_roster(selected_ids)
        if removed != len(selected_ids):
            st.error("Alcuni elementi non sono stati rimossi.")
        else:
            st.success("Rimozione completata.")
        st.session_state["roster_version"] += 1
        st.cache_data.clear()
        st.rerun()

spent, residual = _budget_stats_cached(int(st.session_state["budget_total"]))
st.metric("Speso", f"{spent}")
st.metric("Budget residuo", f"{residual}")

