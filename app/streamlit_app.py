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

from src import dataio, pricing, services


DATA_PROCESSED = "data/processed"
OUTPUT_DIR = "data/outputs"
AUCTION_LOG = f"{OUTPUT_DIR}/auction_log.csv"

BASE_PROCESSED = {
    "quotes": f"{DATA_PROCESSED}/quotes_2025_26_FVM_budget500.csv",
    "stats": f"{DATA_PROCESSED}/stats_master_with_weights.csv",
    "gk": f"{DATA_PROCESSED}/goalkeepers_grid_matrix_square.csv",
}
REQUIRED_FILES = [
    BASE_PROCESSED["quotes"],
    BASE_PROCESSED["stats"],
    f"{DATA_PROCESSED}/derived_prices.csv",
]
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


def train_derived_prices() -> None:
    stats_path = Path(BASE_PROCESSED["stats"])
    quotes_path = Path(BASE_PROCESSED["quotes"])
    stats = dataio.load_stats(stats_path)
    quotes = dataio.load_quotes(quotes_path)
    derived = pricing.train_price_model(stats, quotes, "linear")
    out_path = Path(f"{DATA_PROCESSED}/derived_prices.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(out_path, index=False)


st.sidebar.subheader("Data Setup")
if st.sidebar.button("Prepare processed data"):
    try:
        prepare_processed_data()
        st.sidebar.success("Processed data generated")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Failed to prepare data: {exc}")
if st.sidebar.button("Train derived prices"):
    try:
        train_derived_prices()
        st.sidebar.success("Derived prices trained")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Failed to train prices: {exc}")

if DISABLED:
    message = "Missing required data files:\n" + "\n".join(
        f"- {f}" for f in missing_files
    )
    st.warning(message)


@st.cache_data
def load_players() -> pd.DataFrame:
    quotes = f"{DATA_PROCESSED}/quotes_2025_26_FVM_budget500.csv"
    derived = f"{DATA_PROCESSED}/derived_prices.csv"
    players = dataio.load_quotes(quotes, derived)
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


players = load_players() if not DISABLED else pd.DataFrame()
st.title("Fantacalcio Roster Optimizer")

# price strategy controls
strategy = st.selectbox(
    "Price strategy", ["estimated", "fvm500", "blend"], index=0, disabled=DISABLED
)
alpha = st.slider("Blend alpha", 0.0, 1.0, 0.6, disabled=DISABLED)

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

players["effective_price"] = services.choose_price(players, strategy, alpha)
players["value_score"] = players["expected_points"] / players["effective_price"]

# search bar
name = st.selectbox("Search player", players["name"].sort_values())
sel = players[players["name"] == name].iloc[0]

st.subheader("Player details")
st.write(
    {
        "fvm": sel.get("fvm"),
        "price_500": sel.get("price_500"),
        "estimated_price": sel.get("estimated_price"),
        "expected_points": sel.get("expected_points"),
        "value_score": sel.get("value_score"),
    }
)

# recommendation badge
log = read_log()
state = services.RosterState(
    budget_residual=500 - pd.to_numeric(log.get("price_paid", 0)).sum(),
    team_cap=3,
    team_counts=log[log.get("acquired", 0) == 1]["team"].value_counts().to_dict(),
    slots_needed={r: services.QUOTAS[r] - log[log.get("acquired", 0) == 1]["role"].value_counts().to_dict().get(r, 0) for r in services.QUOTAS},
    value_threshold=players["value_score"].quantile(0.6),
)
rec = services.recommend_player(sel, state)
st.markdown(f"**Recommendation:** {rec['label']} - {rec['reason']}")

# auction log form
st.subheader("Auction log")
with st.form("auction_form"):
    price_paid = st.number_input("Price paid", min_value=0, step=1)
    acquired = st.checkbox("Acquired", value=True)
    submitted = st.form_submit_button("Log")
    if submitted:
        append_log(
            {
                "id": sel["id"],
                "name": sel["name"],
                "team": sel["team"],
                "role": sel["role"],
                "price_paid": int(price_paid),
                "acquired": int(acquired),
            }
        )
        st.success("Entry added to log")


st.subheader("Roster Optimizer")
budget_total = st.number_input("Total budget", value=500)
team_cap = st.number_input("Team cap", value=3)
if st.button("Optimize"):
    roster = services.optimize_roster(
        players, log, budget_total, team_cap, strategy, alpha
    )
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

st.sidebar.subheader("Il mio roster")
acquired = log[log.get("acquired", 0) == 1]
spent = pd.to_numeric(acquired.get("price_paid", 0), errors="coerce").sum()
counts = acquired["role"].value_counts().to_dict()
st.sidebar.write({"spent": spent, "budget_residual": 500 - spent, "counts": counts})
if st.sidebar.button("Esporta il mio roster"):
    acquired.to_csv(f"{OUTPUT_DIR}/my_roster.csv", index=False)
    st.sidebar.success("Roster esportato")

