"""Goalkeeper grid handling with flexible file resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# nome file atteso
GRID_FILENAME = "goalkeepers_grid_matrix_square.csv"


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


def _candidate_paths() -> list[Path]:
    """Possibili posizioni per il CSV (più robuste)."""

    here = Path(__file__).resolve()
    src_root = here.parent  # .../src
    repo_root = src_root.parent  # repo root
    cwd = Path.cwd()
    env = os.getenv("GK_GRID_PATH", "")

    cands: list[Path] = []
    if env:
        p = Path(env)
        cands.append(p if p.name.endswith(".csv") else p / GRID_FILENAME)

    # Ordine: preferito = data/raw/, poi fallback legacy
    cands += [
        # percorso corretto richiesto
        repo_root / "data" / "raw" / GRID_FILENAME,
        cwd / "data" / "raw" / GRID_FILENAME,
        Path("data") / "raw" / GRID_FILENAME,
        # fallback legacy/supporto
        repo_root / "data" / GRID_FILENAME,
        cwd / "data" / GRID_FILENAME,
        Path("data") / GRID_FILENAME,
        repo_root / "app" / "data" / GRID_FILENAME,
    ]

    # dedup preservando ordine
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in cands:
        if p and str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


class GKGrid:
    def __init__(self, path: Optional[str | Path] = None):
        """Se il file non esiste: available=False, nessuna eccezione."""

        self.available = False
        self.df: Optional[pd.DataFrame] = None
        self.resolved_path: Optional[Path] = None

        if path:
            p = Path(path)
            paths = [p if p.name.endswith(".csv") else p / GRID_FILENAME]
        else:
            paths = _candidate_paths()

        for candidate in paths:
            if candidate.exists():
                df = pd.read_csv(candidate, index_col=0)
                df.index = df.index.str.strip().str.upper()
                df.columns = df.columns.str.strip().str.upper()
                self.df = df
                self.available = True
                self.resolved_path = candidate
                break

    def _norm(self, team: str) -> str:
        t = team.strip().upper()
        return TEAM_ALIAS.get(t, t)

    def score_pair(self, team_a: str, team_b: str) -> float:
        if not self.available or self.df is None:
            return 0.0
        a = self._norm(team_a)
        b = self._norm(team_b)
        if a in self.df.index and b in self.df.columns:
            return float(self.df.loc[a, b])
        return 0.0

    def single_score(self, team: str) -> float:
        """Valuta un team senza vincolo: media assoluta della riga (più bassa è meglio)."""

        if not self.available or self.df is None:
            return 0.0
        t = self._norm(team)
        if t in self.df.index:
            row = self.df.loc[t].astype(float).abs()
            return float(row.mean())
        return 0.0


def grid_signal_from_value(v: float) -> float:
    return -abs(v)

