import os
from datetime import datetime
from sqlalchemy import create_engine, String, Integer, Float, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from src.core.paths import db_uri


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
    is_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sold_price: Mapped[int] = mapped_column(Integer, nullable=True)
    sold_at: Mapped[str] = mapped_column(String, nullable=True)


_engine = create_engine(os.environ.get("FANTA_DB_URL", db_uri()), future=True)


def engine():
    return _engine


def init_db(drop: bool = False):
    if drop:
        Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)


def upsert_players(rows: list[dict]):
    """
    rows: [{id,name,team,role,fvm,price_500,expected_points}, ...]
    """
    with Session(_engine) as s:
        for r in rows:
            s.merge(Player(**r))
        s.commit()


def get_player(player_id: int) -> Player | None:
    with Session(_engine) as s:
        return s.get(Player, player_id)


def mark_player_sold(player_id: int, price: int | None) -> tuple[bool, str | None]:
    with Session(_engine) as s:
        p = s.get(Player, player_id)
        if p is None:
            return False, "Player not found"
        if p.is_sold:
            return False, "Player already sold"
        p.is_sold = 1
        p.sold_price = price
        p.sold_at = datetime.utcnow().isoformat()
        s.commit()
        return True, None


def mark_player_unsold(player_id: int) -> tuple[bool, str | None]:
    with Session(_engine) as s:
        p = s.get(Player, player_id)
        if p is None:
            return False, "Player not found"
        if not p.is_sold:
            return False, "Player not sold"
        p.is_sold = 0
        p.sold_price = None
        p.sold_at = None
        s.commit()
        return True, None


def list_searchable_players(
    q: str | None = None,
    role: str | None = None,
    team: str | None = None,
    include_sold: bool = False,
) -> list[Player]:
    with Session(_engine) as s:
        stmt = select(Player)
        if q:
            stmt = stmt.where(Player.name.ilike(f"%{q}%"))
        if role:
            stmt = stmt.where(Player.role == role)
        if team:
            stmt = stmt.where(Player.team == team)
        if not include_sold:
            stmt = stmt.where(Player.is_sold == 0)
        return list(s.scalars(stmt))
