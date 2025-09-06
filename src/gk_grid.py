"""Goalkeeper grid handling with flexible file resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


GRID_FILENAME = "goalkeepers_grid_matrix_square.csv"
# default richiesto (file reale indicato dall'utente)
GRID_DEFAULT = Path("data/raw") / GRID_FILENAME


TEAM_ALIAS: Dict[str, str] = {
    # Mappa full name e 3-letter → codici della matrice (maiuscoli):
    "ATALANTA": "ATA", "ATA": "ATA",
    "BOLOGNA": "BOL", "BOL": "BOL",
    "CAGLIARI": "CAG", "CAG": "CAG",
    "COMO": "COM", "COM": "COM",
    "CREMONESE": "CRE", "CRE": "CRE",
    "FIORENTINA": "FIO", "FIO": "FIO",
    "GENOA": "GEN", "GEN": "GEN",
    "INTER": "INT", "INT": "INT", "INTERNAZIONALE": "INT",
    "JUVENTUS": "JUV", "JUV": "JUV",
    "LAZIO": "LAZ", "LAZ": "LAZ",
    "LECCE": "LEC", "LEC": "LEC",
    "MILAN": "MIL", "MIL": "MIL",
    "NAPOLI": "NAP", "NAP": "NAP",
    "PARMA": "PAR", "PAR": "PAR",
    "PISA": "PIS", "PIS": "PIS",
    "ROMA": "ROM", "ROM": "ROM",
    "SASSUOLO": "SAS", "SAS": "SAS",
    "TORINO": "TOR", "TOR": "TOR",
    "UDINESE": "UDI", "UDI": "UDI",
    "VERONA": "VER", "VER": "VER",
}


def _candidate_paths() -> list[Path]:
    """Ordine di ricerca (preferito: data/raw/...)."""

    here = Path(__file__).resolve()
    src_root = here.parent  # .../src
    repo_root = src_root.parent  # repo root
    cwd = Path.cwd()
    env = os.getenv("GK_GRID_PATH", "")

    cands: list[Path] = []
    if env:
        p = Path(env)
        cands.append(p if p.suffix.lower() == ".csv" else p / GRID_FILENAME)

    cands += [
        repo_root / "data" / "raw" / GRID_FILENAME,  # preferito
        cwd / "data" / "raw" / GRID_FILENAME,
        Path("data") / "raw" / GRID_FILENAME,
        # fallback legacy
        repo_root / "data" / GRID_FILENAME,
        cwd / "data" / GRID_FILENAME,
        Path("data") / GRID_FILENAME,
    ]

    # dedup preservando ordine
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in cands:
        if p and str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


def _norm(s: str) -> str:
    return str(s).strip().upper()


def _norm_team(t: str, headers: list[str]) -> str:
    """Restituisce un token presente negli header della matrice."""

    t_up = _norm(t)
    if t_up in headers:
        return t_up
    alias = TEAM_ALIAS.get(t_up, t_up)
    if alias in headers:
        return alias
    if len(alias) == 3:
        for h in headers:
            if alias == h[:3].upper() or alias in h.upper():
                return h.upper()
    else:
        for h in headers:
            if h[:3].upper() == alias[:3].upper():
                return h.upper()
    return t_up


class GKGrid:
    def __init__(self, path: Optional[str | Path] = None):
        """Se il file non esiste: available=False, nessuna eccezione."""

        self.available = False
        self.df: Optional[pd.DataFrame] = None
        self.resolved_path: Optional[Path] = None

        if path:
            p = Path(path)
            paths = [p if p.suffix.lower() == ".csv" else p / GRID_FILENAME]
        else:
            # priorità al path default corretto
            paths = [GRID_DEFAULT] + _candidate_paths()

        for candidate in paths:
            if candidate.exists():
                df = pd.read_csv(candidate, index_col=0)
                # normalizza header a MAIUSCOLO: 'Ata' → 'ATA'
                df.index = df.index.map(lambda x: str(x).strip().upper())
                df.columns = df.columns.map(lambda x: str(x).strip().upper())
                self.df = df
                self.available = True
                self.resolved_path = candidate
                break

    def score_pair(self, team_a: str, team_b: str) -> float:
        """Valore grid tra due team (più vicino a 0 è migliore)."""

        if not self.available or self.df is None:
            return 0.0
        headers_r = list(self.df.index)
        headers_c = list(self.df.columns)
        a = _norm_team(team_a, headers_r)
        b = _norm_team(team_b, headers_c)
        if a in self.df.index and b in self.df.columns:
            return float(self.df.loc[a, b])
        return 0.0

    def single_score(self, team: str) -> float:
        """Valuta un team senza vincolo: media assoluta della riga (più bassa è meglio)."""

        if not self.available or self.df is None:
            return 0.0
        headers_r = list(self.df.index)
        t = _norm_team(team, headers_r)
        if t in self.df.index:
            row = self.df.loc[t].astype(float).abs()
            return float(row.mean())
        return 0.0


def grid_signal_from_value(v: float) -> float:
    """0 è ottimo → segnale = -|v| (più alto = meglio)."""

    return -abs(v)

