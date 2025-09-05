"""Feature engineering functions."""
from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple features per 90 and availability."""
    out = df.copy()
    minutes_factor = out["mins"].replace(0, 1) / 90
    out["goals_per90"] = out["goals"] / minutes_factor
    out["assists_per90"] = out["assists"] / minutes_factor
    out["form"] = (out["goals"] + out["assists"]) / minutes_factor
    out["availability"] = out["mins"] / (38 * 90)
    return out
