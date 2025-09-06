from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from sqlalchemy import create_engine, Integer, Float, String, DateTime, select, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    team: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    fvm: Mapped[int] = mapped_column(Integer, nullable=True)
    price_500: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_points: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Stato vendita (0/1 in SQLite), prezzo e timestamp
    is_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sold_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sold_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Tracking acquisti personali
    my_acquired: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    my_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    my_acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

_engine: Optional["Engine"] = None


def get_db_path() -> Path:
    """Return absolute path to the SQLite DB inside the repo."""
    return Path(__file__).resolve().parents[1] / "data" / "fanta.db"


def get_engine():
    """Return a singleton SQLAlchemy engine with SQLite pragmas."""
    global _engine
    if _engine is None:
        db_url = os.environ.get("FANTA_DB_URL")
        if not db_url:
            db_url = f"sqlite:///{get_db_path()}"
        _engine = create_engine(db_url, future=True)
        with _engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    return _engine


def get_session() -> Session:
    """Return a Session bound to the shared engine."""
    return Session(get_engine())


# Backwards compatibility
def engine():
    return get_engine()


def init_db(drop: bool = False):
    eng = get_engine()
    if drop:
        Base.metadata.drop_all(bind=eng)
    Base.metadata.create_all(bind=eng)
    # Migrazione add-only per DB esistente
    with eng.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(players)").fetchall()}
        if "is_sold" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN is_sold INTEGER NOT NULL DEFAULT 0")
        if "sold_price" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN sold_price INTEGER")
        if "sold_at" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN sold_at TEXT")
        if "my_acquired" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN my_acquired INTEGER NOT NULL DEFAULT 0")
        if "my_price" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN my_price INTEGER")
        if "my_acquired_at" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN my_acquired_at TEXT")


def upsert_players(rows: list[dict]):
    """
    rows: [{id,name,team,role,fvm,price_500,expected_points}, ...]
    """
    with get_session() as s:
        for r in rows:
            s.merge(Player(**r))
        s.commit()


def list_searchable_players(
    q: str | None = None,
    role: str | None = None,
    team: str | None = None,
    include_sold: bool = False,
) -> list[Player]:
    with get_session() as s:
        stmt = select(Player)
        if q:
            stmt = stmt.where(Player.name.like(f"%{q}%"))
        if role:
            stmt = stmt.where(Player.role == role)
        if team:
            stmt = stmt.where(Player.team == team)
        if not include_sold:
            stmt = stmt.where(Player.is_sold == 0)
        return list(s.scalars(stmt))


def mark_player_acquired(player_id: int, price: int, when: Optional[str] = None) -> None:
    """Mark player as acquired with price and timestamp."""
    when_dt = datetime.fromisoformat(when) if when else datetime.utcnow()
    with get_session() as s:
        p = s.get(Player, int(player_id))
        if not p:
            return
        p.my_acquired = 1
        p.my_price = int(price)
        p.my_acquired_at = when_dt
        s.commit()


def remove_from_roster(ids: list[int]) -> int:
    """Unset acquisition fields for the given player IDs.

    Returns the number of updated rows.
    """
    if not ids:
        return 0
    with get_session() as s:
        count = (
            s.query(Player)
            .filter(Player.id.in_(ids))
            .update(
                {
                    Player.my_acquired: 0,
                    Player.my_price: None,
                    Player.my_acquired_at: None,
                },
                synchronize_session=False,
            )
        )
        s.commit()
        return count


def get_my_roster() -> list[Player]:
    """Return list of players marked as acquired."""
    with get_session() as s:
        stmt = (
            select(Player)
            .where(Player.my_acquired == 1)
            .order_by(Player.role, Player.team, Player.name)
        )
        return list(s.scalars(stmt))


def read_players_df() -> pd.DataFrame:
    """Return the players table as a pandas DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql_table("players", conn)


__all__ = [
    "engine",
    "get_engine",
    "get_session",
    "get_db_path",
    "init_db",
    "Player",
    "upsert_players",
    "list_searchable_players",
    "mark_player_acquired",
    "remove_from_roster",
    "get_my_roster",
    "read_players_df",
]
