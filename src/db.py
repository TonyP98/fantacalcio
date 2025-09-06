from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Integer, Float, String, DateTime, select
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
    # Stato vendita
    is_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sold_price: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sold_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

DB_URL = os.environ.get("FANTA_DB_URL", "sqlite:///data/fanta.db")
_engine = create_engine(DB_URL, future=True)


def engine():
    return _engine


def init_db(drop: bool = False):
    if drop:
        Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)
    # aggiunte "safe" su DB già esistente
    with _engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(players)").fetchall()}
        if "is_sold" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN is_sold INTEGER NOT NULL DEFAULT 0")
        if "sold_price" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN sold_price INTEGER")
        if "sold_at" not in cols:
            conn.exec_driver_sql("ALTER TABLE players ADD COLUMN sold_at TEXT")


def upsert_players(rows: list[dict]):
    """
    rows: [{id,name,team,role,fvm,price_500,expected_points}, ...]
    """
    with Session(_engine) as s:
        for r in rows:
            s.merge(Player(**r))
        s.commit()


def get_player(player_id: int) -> Optional[Player]:
    with Session(_engine) as s:
        return s.get(Player, player_id)


def mark_player_sold(player_id: int, price: Optional[int] = None) -> tuple[bool, Optional[str]]:
    with Session(_engine) as s:
        p = s.get(Player, player_id)
        if p is None:
            return False, "Player not found"
        if p.is_sold:
            return False, "Player already sold"
        p.is_sold = 1
        p.sold_price = int(price) if price is not None else None
        p.sold_at = datetime.utcnow()
        s.commit()
        return True, None


def mark_player_unsold(player_id: int) -> tuple[bool, Optional[str]]:
    with Session(_engine) as s:
        p = s.get(Player, player_id)
        if p is None:
            return False, "Player not found"
        p.is_sold = 0
        p.sold_price = None
        p.sold_at = None
        s.commit()
        return True, None


def list_searchable_players(
    q: Optional[str] = None,
    role: Optional[str] = None,
    team: Optional[str] = None,
    include_sold: bool = False,
) -> list[Player]:
    with Session(_engine) as s:
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


__all__ = [
    "engine",
    "init_db",
    "Player",
    "upsert_players",
    "get_player",
    "mark_player_sold",
    "mark_player_unsold",
    "list_searchable_players",
]
