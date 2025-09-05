import pandas as pd
import pytest

from src import features, pricing, utils


def prepare():
    config = utils.load_config()
    df = pd.read_csv("examples/sample_players.csv")
    feats = features.build_features(df)
    return feats, config


def test_heuristic_price_budget():
    feats, config = prepare()
    priced = pricing.heuristic_price(feats, config["scoring_weights"], config["budget"])
    assert pytest.approx(priced["fair_price"].sum(), abs=1e-6) == config["budget"]


def test_baseline_price_budget():
    feats, config = prepare()
    priced = pricing.baseline_linear(feats, config["budget"])
    assert pytest.approx(priced["fair_price"].sum(), abs=1e-6) == config["budget"]
