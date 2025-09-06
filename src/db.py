from __future__ import annotations

import os
from datetime import datetime

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
    # Stato vendita (0/1 in SQLite), prezzo e timestamp
    is_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sold_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sold_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Tracking acquisti personali
    my_acquired: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    my_price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    my_acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

DB_URL = os.environ.get("FANTA_DB_URL", "sqlite:///data/fanta.db")
_engine = create_engine(DB_URL, future=True)


def engine():
    return _engine


def init_db(drop: bool = False):
    if drop:
        Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)
    # Migrazione add-only per DB esistente
    with _engine.begin() as conn:
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
    with Session(_engine) as s:
        for r in rows:
            s.merge(Player(**r))
        s.commit()


def list_searchable_players(
    q: str | None = None,
    role: str | None = None,
    team: str | None = None,
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
    "list_searchable_players",
]
