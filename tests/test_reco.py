import pandas as pd

from src.reco import recommend_player
from src.gk_grid import GKGrid, grid_signal_from_value


def make_grid(tmp_path):
    df = pd.DataFrame([[0, 0], [2, 0]], index=["A", "B"], columns=["A", "B"])
    path = tmp_path / "grid.csv"
    df.to_csv(path)
    return GKGrid(path)


def test_player_details_fields_hidden():
    row = {
        "name": "Test",
        "team": "Inter",
        "role": "C",
        "price_500": 15,
        "fvm": 20,
        "status": "AVAILABLE",
        "sold_price": None,
    }
    fair_value = row.get("fvm") or row.get("estimated_price")
    price_500 = row.get("price_500")
    role = row["role"]
    team = row["team"]
    rec_label, _ = recommend_player(row, fair_value=fair_value, price_500=price_500, role=role, team=team)
    details = {
        "name": row["name"],
        "team": team,
        "role": role,
        "price_500": price_500,
        "status": row.get("status", "AVAILABLE"),
        "sold_price": row.get("sold_price"),
        "Recommendation": rec_label,
    }
    assert "expected_points" not in details
    assert "value_score" not in details


def test_reco_outfield():
    label, score = recommend_player({}, fair_value=20, price_500=15, role="C", team="Inter")
    assert label == "BUY"
    assert round(score, 3) == round((20 - 15) / 15, 3)


def test_reco_gk_unbound(tmp_path):
    grid = make_grid(tmp_path)
    grid_val = grid.single_score("B")
    gk_signal = grid_signal_from_value(grid_val)
    label, _ = recommend_player({}, fair_value=10, price_500=10, role="P", team="B", gk_signal=gk_signal)
    assert label == "AVOID"


def test_reco_gk_bound(tmp_path):
    grid = make_grid(tmp_path)
    gk_signal_unbound = grid_signal_from_value(grid.single_score("B"))
    label_unbound, _ = recommend_player({}, fair_value=10, price_500=10, role="P", team="B", gk_signal=gk_signal_unbound)
    gk_signal_bound = grid_signal_from_value(grid.score_pair("A", "B"))
    label_bound, _ = recommend_player({}, fair_value=10, price_500=10, role="P", team="B", gk_signal=gk_signal_bound)
    assert label_unbound != label_bound
    assert label_bound == "HOLD"
