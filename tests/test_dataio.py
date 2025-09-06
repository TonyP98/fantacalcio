from pathlib import Path

from src import dataio, utils


def test_load_csv_price500():
    config = utils.load_config()
    df = dataio.load_csv(Path("examples/sample_players.csv"), config)
    assert "price_500" in df.columns
    assert "price" not in df.columns
