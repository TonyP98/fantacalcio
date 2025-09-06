"""Service layer helpers for pricing and roster optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Roster optimiser

QUOTAS: Dict[str, int] = {"P": 3, "D": 8, "C": 8, "A": 6}


def _initial_state(auction_log: pd.DataFrame | None) -> Dict[str, Dict[str, int]]:
    roles = {r: 0 for r in QUOTAS}
    teams: Dict[str, int] = {}
    spent = 0.0
    if auction_log is not None and not auction_log.empty:
        acquired = auction_log[auction_log.get("acquired", 0) == 1]
        roles.update(acquired["role"].value_counts().to_dict())
        teams = acquired["team"].value_counts().to_dict()
        spent = pd.to_numeric(acquired.get("price_paid", 0), errors="coerce").sum()
    return {"roles": roles, "teams": teams, "spent": spent}


def optimize_roster(
    players: pd.DataFrame,
    auction_log: pd.DataFrame | None = None,
    budget_total: float = 500,
    team_cap: int = 3,
    allow_low_mins: bool = False,
    grid_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Greedy roster optimisation.

    Returns a DataFrame with selected players and cumulative budget used.
    """

    state = _initial_state(auction_log)
    budget_residual = budget_total - state["spent"]

    pool = players.copy()
    if not allow_low_mins and "apps" in pool.columns:
        pool = pool[pool["apps"] >= 8]

    # exclude already acquired players
    if auction_log is not None and not auction_log.empty:
        acquired_ids = auction_log[auction_log.get("acquired", 0) == 1]["id"].tolist()
        pool = pool[~pool["id"].isin(acquired_ids)]

    price_500 = pd.to_numeric(pool["price_500"], errors="coerce").fillna(0)
    pool["effective_price"] = price_500.clip(lower=1)
    pool["value_score"] = pool["expected_points"] / pool["effective_price"]

    selected = []
    spent = 0.0

    for role, quota in QUOTAS.items():
        current = state["roles"].get(role, 0)
        needed = quota - current
        if needed <= 0:
            continue

        candidates = pool[pool["role"] == role].copy()
        candidates = candidates.sort_values(
            ["value_score", "expected_points", "effective_price"],
            ascending=[False, False, True],
        )

        for _, row in candidates.iterrows():
            if needed <= 0:
                break
            if spent + row["effective_price"] > budget_residual:
                continue
            if state["teams"].get(row["team"], 0) >= team_cap:
                continue

            selected.append(row)
            spent += row["effective_price"]
            state["teams"][row["team"]] = state["teams"].get(row["team"], 0) + 1
            needed -= 1

        if needed > 0:
            raise RuntimeError(f"Insufficient data or budget for role {role}")

    roster = pd.DataFrame(selected)
    if roster.empty:
        return roster

    roster["cum_budget"] = roster["effective_price"].cumsum()

    if grid_matrix is not None:
        gks = roster[roster["role"] == "P"]
        if len(gks) == 3:
            teams = gks["team"].tolist()
            vals = []
            for i in range(3):
                for j in range(i + 1, 3):
                    vals.append(grid_matrix.loc[teams[i], teams[j]])
            avg = np.nanmean(vals)
            roster.attrs["goalkeeper_grid_avg"] = avg
            if avg < 8:
                roster.attrs["goalkeeper_penalty"] = 0.5
            else:
                roster.attrs["goalkeeper_penalty"] = 0.0

    return roster


# ---------------------------------------------------------------------------
# Player recommendation


@dataclass
class RosterState:
    """Minimal representation of current roster status."""

    budget_residual: float
    team_cap: int
    team_counts: Dict[str, int]
    slots_needed: Dict[str, int]
    value_threshold: float


def recommend_player(row: pd.Series, roster_state: RosterState) -> Dict[str, str]:
    """Return BUY/AVOID recommendation for a player."""

    team_count = roster_state.team_counts.get(row["team"], 0)
    if team_count >= roster_state.team_cap:
        return {"label": "AVOID", "reason": "team cap reached"}

    slots_left = sum(roster_state.slots_needed.values()) or 1
    budget_per_slot = roster_state.budget_residual / slots_left
    if row["effective_price"] > budget_per_slot:
        return {"label": "AVOID", "reason": "price too high"}

    threshold = roster_state.value_threshold
    if roster_state.slots_needed.get(row["role"], 0) > 0:
        threshold *= 0.9  # slightly easier if role is needed

    if row["value_score"] >= threshold:
        return {"label": "BUY", "reason": "value above threshold"}
    return {"label": "AVOID", "reason": "low value"}

