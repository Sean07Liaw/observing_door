from pathlib import Path

from app.config import settings
from app.db import Base, engine
from app import models  # noqa: F401


def ensure_directories() -> None:
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    settings.clip_dir.mkdir(parents=True, exist_ok=True)

    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.replace("sqlite:///", "", 1)
        db_parent = Path(db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()