from sqlalchemy import text

from src.db import ENGINE, init_db, upsert_players


def test_init_and_upsert_creates_players_table(tmp_path):
    init_db(drop=True)
    rows = [
        {
            "id": 1,
            "name": "Foo",
            "team": "T1",
            "role": "A",
            "fvm": 10,
            "price_500": 5,
            "expected_points": 3.2,
        }
    ]
    upsert_players(rows)
    with ENGINE.begin() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(players)"))]
        expected = {
            "id",
            "name",
            "team",
            "role",
            "fvm",
            "price_500",
            "expected_points",
        }
        assert set(cols) == expected
        res = conn.execute(text("SELECT price_500 FROM players WHERE id=1")).fetchone()
        assert res[0] == 5
