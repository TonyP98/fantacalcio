import pandas as pd
import pytest

from src import features


def test_build_features_per90():
    df = pd.read_csv("examples/sample_players.csv")
    feats = features.build_features(df)
    first = feats.iloc[0]
    expected = 15 / (2800 / 90)
    assert pytest.approx(first["goals_per90"], 0.01) == expected
    assert 0 <= first["availability"] <= 1
    assert first["expected_points"] == df.iloc[0]["goals"] + df.iloc[0]["assists"]
