import pandas as pd

from src import ranking


def test_rank_players_score():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A", "B"],
            "team": ["T1", "T2"],
            "role": ["P", "P"],
            "fanta_avg": [6.0, 5.0],
            "price_500": [10, 20],
        }
    )
    ranked = ranking.rank_players(df, "score_z_role", "ALL", 2, 0)
    assert list(ranked["name"]) == ["A", "B"]
    assert "score_z_role" in ranked.columns


def test_rank_players_deduplicates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "name": ["A", "A", "B"],
            "team": ["T1", "T1", "T2"],
            "role": ["P", "P", "P"],
            "fanta_avg": [6.0, 6.0, 5.0],
            "price_500": [10, 10, 20],
        }
    )
    ranked = ranking.rank_players(df, "score_z_role", "ALL", 10, 0)
    assert ranked["id"].tolist().count(1) == 1
    assert len(ranked) == 2
