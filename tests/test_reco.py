import pandas as pd

from src.reco import compute_scores, apply_recommendation, RecConfig


def _sample_df():
    return pd.DataFrame(
        [
            {"id": 1, "name": "A", "team": "T", "role": "C", "fanta_avg": 6.0, "price_500": 10},
            {"id": 2, "name": "B", "team": "T", "role": "C", "fanta_avg": 6.0, "price_500": 20},
            {"id": 3, "name": "C", "team": "T", "role": "C", "fanta_avg": 7.0, "price_500": 20},
            {"id": 4, "name": "D", "team": "T", "role": "C", "fanta_avg": 4.0, "price_500": 50},
            {"id": 5, "name": "E", "team": "T", "role": "C", "fanta_avg": 5.0, "price_500": 5},
        ]
    )


def test_monotonicity_and_labels():
    df = compute_scores(_sample_df())

    # same fanta_avg, higher price -> higher score_raw
    a = df.loc[df["name"] == "A", "score_raw"].iloc[0]
    b = df.loc[df["name"] == "B", "score_raw"].iloc[0]
    assert b > a

    # same price, higher fanta_avg -> higher score_raw
    c = df.loc[df["name"] == "C", "score_raw"].iloc[0]
    assert c > b

    cfg = RecConfig(buy_alpha=0.5, hold_alpha=0.0)
    labeled = apply_recommendation(df, cfg)
    assert {"BUY", "HOLD", "AVOID"}.issubset(set(labeled["Recommendation"]))


def test_thresholds_change_labels():
    df = compute_scores(_sample_df())
    cfg1 = RecConfig(buy_alpha=0.5, hold_alpha=0.0)
    cfg2 = RecConfig(buy_alpha=1.5, hold_alpha=0.5)
    labeled1 = apply_recommendation(df, cfg1)
    labeled2 = apply_recommendation(df, cfg2)
    assert (labeled1["score_raw"] == labeled2["score_raw"]).all()
    assert labeled1["Recommendation"].nunique() >= labeled2["Recommendation"].nunique()
