import argparse

from app.db import SessionLocal
from app.logger import get_logger
from app.services.event_service import delete_event_by_id

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete one event by id.")
    parser.add_argument("--id", type=int, required=True, help="Event id to delete")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        deleted = delete_event_by_id(db, args.id)

        if deleted:
            print(f"Deleted event id={args.id}")
        else:
            print(f"Event id={args.id} not found")
    except Exception:
        logger.error("Failed to delete event", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()