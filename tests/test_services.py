import logging
import pandas as pd

from src import services

ROLES_NEED = {"P": 3, "D": 8, "C": 8, "A": 6}


def _sample_players() -> pd.DataFrame:
    rows = []
    for role, quota in ROLES_NEED.items():
        for i in range(quota + 2):  # extra candidates
            rows.append(
                {
                    "id": f"{role}{i}",
                    "team": f"T{i%5}",
                    "role": role,
                    "price_500": 10 + i,
                    "score_z_role": 1 + i / 10.0,
                    "status": "AVAILABLE",
                }
            )
    return pd.DataFrame(rows)


def test_optimizer_handles_locked_players():
    players = _sample_players()
    # mark one player per role as already acquired
    for role in ROLES_NEED.keys():
        idx = players[players["role"] == role].index[0]
        players.loc[idx, "my_acquired"] = 1
        players.loc[idx, "my_price"] = 5 + idx

    logger = logging.getLogger("test")
    roster = services.optimize_roster(players, logger, budget_total=500, team_cap=10)

    assert len(roster) == sum(ROLES_NEED.values())
    locked = roster[roster["locked"] == True]
    assert len(locked) == len(ROLES_NEED)

    expected_locked = players.loc[players["my_acquired"] == 1, "my_price"].sum()
    assert roster["budget_locked"].iloc[0] == expected_locked


def test_optimizer_best_effort_on_low_budget():
    players = _sample_players()
    logger = logging.getLogger("test")
    roster = services.optimize_roster(players, logger, budget_total=50, team_cap=10)

    assert len(roster) < sum(ROLES_NEED.values())
    assert roster["budget_left"].iloc[0] >= 0

