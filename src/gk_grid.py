from __future__ import annotations

from pathlib import Path

import pandas as pd

GRID_PATH = Path("data/goalkeepers_grid_matrix_square.csv")

TEAM_ALIAS = {
    "ATALANTA": "ATA",
    "BOLOGNA": "BOL",
    "CAGLIARI": "CAG",
    "COMO": "COM",
    "CREMONESE": "CRE",
    "FIORENTINA": "FIO",
    "GENOA": "GEN",
    "INTER": "INT",
    "JUVENTUS": "JUV",
    "LAZIO": "LAZ",
    "LECCE": "LEC",
    "MILAN": "MIL",
    "NAPOLI": "NAP",
    "PARMA": "PAR",
    "PISA": "PIS",
    "ROMA": "ROM",
    "SASSUOLO": "SAS",
    "TORINO": "TOR",
    "UDINESE": "UDI",
    "VERONA": "VER",
}


class GKGrid:
    def __init__(self, path: str | Path = GRID_PATH):
        self.df = pd.read_csv(path, index_col=0)
        self.df.index = self.df.index.str.strip().str.upper()
        self.df.columns = self.df.columns.str.strip().str.upper()

    def _norm(self, team: str) -> str:
        t = team.strip().upper()
        return TEAM_ALIAS.get(t, t)

    def score_pair(self, team_a: str, team_b: str) -> float:
        a = self._norm(team_a)
        b = self._norm(team_b)
        if a in self.df.index and b in self.df.columns:
            return float(self.df.loc[a, b])
        return 0.0

    def single_score(self, team: str) -> float:
        t = self._norm(team)
        if t in self.df.index:
            row = self.df.loc[t].astype(float).abs()
            return float(row.mean())
        return 0.0


def grid_signal_from_value(v: float) -> float:
    return -abs(v)
