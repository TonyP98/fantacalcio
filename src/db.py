import os
from sqlalchemy import create_engine, String, Integer, Float
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
