from sqlalchemy import create_engine, text

ENGINE = create_engine("sqlite:///data/fanta.db", future=True)


def init_db(drop: bool = False):
    with ENGINE.begin() as conn:
        if drop:
            conn.execute(text("DROP TABLE IF EXISTS players"))
        conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT NOT NULL,
            role TEXT NOT NULL,
            fvm INTEGER,
            price_500 INTEGER NOT NULL,
            expected_points REAL DEFAULT 0.0
        )
        """
        ))


def upsert_players(rows):
    # rows: iterable di dict con chiavi: id,name,team,role,fvm,price_500,expected_points
    sql = text(
    """
    INSERT INTO players (id, name, team, role, fvm, price_500, expected_points)
    VALUES (:id, :name, :team, :role, :fvm, :price_500, :expected_points)
    ON CONFLICT(id) DO UPDATE SET
        name=excluded.name,
        team=excluded.team,
        role=excluded.role,
        fvm=excluded.fvm,
        price_500=excluded.price_500,
        expected_points=excluded.expected_points
    """
    )
    with ENGINE.begin() as conn:
        conn.execute(sql, rows)
