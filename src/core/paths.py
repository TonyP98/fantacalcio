from pathlib import Path

# repository root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "fanta.db"

def db_uri() -> str:
    return f"sqlite:///{DB_PATH.as_posix()}"
