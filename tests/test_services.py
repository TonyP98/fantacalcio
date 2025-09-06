import logging
import os
from importlib import reload
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


def test_optimizer_dedupes_duplicate_locked_players():
    players = _sample_players()
    idx = players[players["role"] == "P"].index[0]
    players.loc[idx, "my_acquired"] = 1
    players.loc[idx, "my_price"] = 5
    # introduce a duplicate entry for the same locked player
    duplicate = players.loc[[idx]].copy()
    players_dup = pd.concat([players, duplicate], ignore_index=True)

    roster = services.optimize_roster(
        players_dup, logging.getLogger("test"), budget_total=500, team_cap=10
    )
    locked = roster[roster["locked"] == True]
    # the locked player should appear only once in the final roster
    assert locked["id"].tolist().count(players.loc[idx, "id"]) == 1


def test_optimizer_best_effort_on_low_budget():
    players = _sample_players()
    logger = logging.getLogger("test")
    roster = services.optimize_roster(players, logger, budget_total=50, team_cap=10)

    assert len(roster) < sum(ROLES_NEED.values())
    assert roster["budget_left"].iloc[0] >= 0


def test_attach_my_roster_from_csv(tmp_path):
    players = pd.DataFrame({
        "id": [1, 2],
        "name": ["A", "B"],
        "team": ["T1", "T2"],
    })
    roster = pd.DataFrame({"id": [2], "my_price": [10]})
    roster.to_csv(tmp_path / "my_roster.csv", index=False)
    orig = services.OUTPUT_DIR
    services.OUTPUT_DIR = str(tmp_path)
    try:
        enriched = services.attach_my_roster(players)
    finally:
        services.OUTPUT_DIR = orig

    assert list(enriched["my_acquired"]) == [0, 1]
    assert enriched.loc[1, "my_price"] == 10


def test_attach_my_roster_from_session_state():
    players = pd.DataFrame({
        "id": [1, 2],
        "name": ["A", "B"],
        "team": ["T1", "T2"],
    })

    class FakeSt:
        def __init__(self):
            self.session_state = {
                "my_roster": [{"id": 1, "my_price": 20}]
            }

    st = FakeSt()
    enriched = services.attach_my_roster(players, st)
    assert list(enriched["my_acquired"]) == [1, 0]
    assert enriched.loc[0, "my_price"] == 20


def test_attach_my_roster_from_db(monkeypatch):
    os.environ["FANTA_DB_URL"] = "sqlite:///:memory:"
    from src import db as _db

    reload(_db)
    _db.init_db()
    _db.upsert_players(
        [
            {
                "id": 1,
                "name": "A",
                "team": "T1",
                "role": "P",
                "fvm": 0,
                "price_500": 5,
                "expected_points": 0.0,
            },
            {
                "id": 2,
                "name": "B",
                "team": "T2",
                "role": "D",
                "fvm": 0,
                "price_500": 6,
                "expected_points": 0.0,
            },
        ]
    )
    _db.mark_player_acquired(2, 11)

    players = pd.DataFrame(
        {"id": [1, 2], "name": ["A", "B"], "team": ["T1", "T2"]}
    )
    enriched = services.attach_my_roster(players)
    assert list(enriched["my_acquired"]) == [0, 1]
    assert enriched.loc[1, "my_price"] == 11

