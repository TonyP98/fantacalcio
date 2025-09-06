import os
from datetime import datetime

os.environ["FANTA_DB_URL"] = "sqlite:///:memory:"

from src.db import (
    init_db,
    upsert_players,
    mark_player_acquired,
    get_my_roster,
    Player,
    get_session,
)


def test_mark_player_acquired_and_roster_fetch():
    init_db()
    upsert_players(
        [
            {
                "id": 1,
                "name": "Foo",
                "team": "AAA",
                "role": "P",
                "fvm": 1,
                "price_500": 10,
                "expected_points": 5.0,
            }
        ]
    )
    mark_player_acquired(1, 7, when="2024-01-01T00:00:00")
    roster = get_my_roster()
    assert len(roster) == 1
    p = roster[0]
    assert p.my_acquired == 1
    assert p.my_price == 7
    assert isinstance(p.my_acquired_at, datetime)
    with get_session() as s:
        db_p = s.get(Player, 1)
        assert db_p.my_acquired == 1
        assert db_p.my_price == 7
