import pandas as pd

from src import ranking


def test_rank_players_value_score():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A", "B"],
            "team": ["T1", "T2"],
            "role": ["P", "D"],
            "expected_points": [20, 20],
            "price_500": [10, 20],
        }
    )
    ranked = ranking.rank_players(df, "value_score", "ALL", 2, 0)
    assert list(ranked["name"]) == ["A", "B"]
    assert ranked.iloc[0]["value_score"] == 2
