import numpy as np
import pandas as pd

from src import services


def _sample_players() -> pd.DataFrame:
    rows = []
    for role, quota in services.QUOTAS.items():
        for i in range(quota + 2):  # extra candidates
            rows.append(
                {
                    "id": f"{role}{i}",
                    "name": f"{role}{i}",
                    "team": f"T{i%5}",
                    "role": role,
                    "expected_points": 10 + i,
                    "price_500": 10,
                    "apps": 10,
                }
            )
    return pd.DataFrame(rows)


def test_choose_price_fallback(caplog):
    df = pd.DataFrame({"price_500": [10, 20]})
    with caplog.at_level("WARNING"):
        prices = services.choose_price(df, "estimated")
    assert prices.tolist() == [10, 20]
    assert "Derived prices rimossi" in caplog.text


def test_optimizer_respects_quotas_and_budget():
    players = _sample_players()
    roster = services.optimize_roster(
        players, budget_total=500, team_cap=10, price_strategy="fvm500"
    )
    assert len(roster) == sum(services.QUOTAS.values())
    counts = roster["role"].value_counts().to_dict()
    for role, qty in services.QUOTAS.items():
        assert counts.get(role, 0) == qty
    assert roster["effective_price"].sum() <= 500

