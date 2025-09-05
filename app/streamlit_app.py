"""Streamlit UI for roster optimisation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import dataio, services


DATA_PROCESSED = "data/processed"
OUTPUT_DIR = "data/outputs"
AUCTION_LOG = f"{OUTPUT_DIR}/auction_log.csv"


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


players = load_players()
st.title("Fantacalcio Roster Optimizer")

# price strategy controls
strategy = st.selectbox("Price strategy", ["estimated", "fvm500", "blend"], index=0)
alpha = st.slider("Blend alpha", 0.0, 1.0, 0.6)
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

