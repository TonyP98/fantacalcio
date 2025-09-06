from src.gk_grid import GKGrid


def test_missing_grid_file(tmp_path):
    missing = tmp_path / "non_existing.csv"
    grid = GKGrid(missing)
    assert not grid.available
    assert grid.single_score("A") == 0.0
    assert grid.score_pair("A", "B") == 0.0


def test_grid_default_and_aliases():
    """Ensure the grid loads default path and normalizes team names."""

    grid = GKGrid()  # use default file resolution
    assert grid.available
    # full names map to 3-letter codes
    assert grid.score_pair("Atalanta", "Bologna") == 12.0
    # alias handling (Internazionale -> INT)
    assert grid.score_pair("Internazionale", "Juventus") == 12.0
    # single score uses normalized headers
    assert grid.single_score("Atalanta") == 9.0

