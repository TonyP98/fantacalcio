import os
import types
from pathlib import Path
from sqlalchemy.orm import Session

os.environ["FANTA_DB_URL"] = "sqlite:///:memory:"
from src.db import Player, upsert_players, engine

def load_module(tmp_path):
    code = Path("app/streamlit_app.py").read_text()
    prefix = code.split("players = load_players()", 1)[0]
    mod = types.ModuleType("sa")
    mod.__file__ = "app/streamlit_app.py"
    exec(prefix, mod.__dict__)
    mod.AUCTION_LOG = tmp_path / "auction_log.csv"
    return mod

def test_process_log_updates_roster_and_log(tmp_path):
    sa = load_module(tmp_path)
    upsert_players([
        {"id": 1, "name": "Foo", "team": "AAA", "role": "P", "fvm": 1, "price_500": 10, "expected_points": 5.0},
        {"id": 2, "name": "Bar", "team": "BBB", "role": "D", "fvm": 1, "price_500": 12, "expected_points": 6.0},
    ])

    ok, warn = sa.process_log(1, 7, True)
    assert ok and warn is None
    with Session(engine()) as s:
        p1 = s.get(Player, 1)
        assert p1.my_acquired == 1
        assert p1.my_price == 7
    log = sa.read_log()
    assert len(log) == 1 and log.iloc[0]["acquired"] == 1

    ok, warn = sa.process_log(2, 5, False)
    assert ok and warn is None
    with Session(engine()) as s:
        p2 = s.get(Player, 2)
        assert p2.my_acquired == 0
    log = sa.read_log()
    assert list(log["acquired"]) == [1, 0]
