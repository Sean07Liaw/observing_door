import argparse

from app.config import settings
from app.db import SessionLocal
from app.logger import get_logger
from app.services.event_service import delete_all_events

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all events from the database (development only)."
    )
    parser.add_argument(
        "--confirm",
        type=str,
        required=True,
        help='Must be exactly "YES" to proceed',
    )
    args = parser.parse_args()

    if settings.app_env.lower() != "dev":
        raise RuntimeError("delete_all_events is only allowed when APP_ENV=dev")

    if args.confirm != "YES":
        raise ValueError('Deletion aborted. Use --confirm YES to proceed.')

    db = SessionLocal()
    try:
        deleted_count = delete_all_events(db)
        print(f"Deleted {deleted_count} event(s)")
    except Exception:
        logger.error("Failed to delete all events", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()