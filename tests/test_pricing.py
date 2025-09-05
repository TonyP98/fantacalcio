import pandas as pd
from pathlib import Path

from src import pricing


def test_train_derived_prices(tmp_path):
    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    quotes = pd.DataFrame(
        {
            "name": ["Player A"],
            "team": ["AAA"],
            "role": ["F"],
            "price_500": [10],
        }
    )
    stats = pd.DataFrame(
        {
            "name": ["Player A"],
            "team": ["AAA"],
            "role": ["F"],
            "goals": [1],
            "assists": [0],
            "mins": [90],
        }
    )

    quotes.to_csv(processed / "quotes_2025_26_FVM_budget500.csv", index=False)
    stats.to_csv(processed / "stats_master_with_weights.csv", index=False)
    (processed / "goalkeepers_grid_matrix_square.csv").write_text("")

    assert pricing.list_missing_required_inputs() == []

    out = pricing.train_derived_prices(overwrite=True)
    assert out["rows"] == 1
    assert Path(out["csv_path"]).exists()
    assert Path(out["db_path"]).exists()
