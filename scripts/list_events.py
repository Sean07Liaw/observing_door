import argparse

from app.db import SessionLocal
from app.logger import get_logger
from app.services.event_service import list_events

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="List events from the database.")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Maximum number of events")
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="List events in ascending order by id",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        events = list_events(
            db,
            limit=args.limit,
            newest_first=not args.oldest_first,
        )

        if not events:
            print("No events found.")
            return

        print(f"Found {len(events)} event(s):")
        for event in events:
            print(
                f"[id={event.id}] "
                f"{event.timestamp} | "
                f"camera={event.camera_id} | "
                f"zone={event.zone} | "
                f"type={event.event_type} | "
                f"occupancy={event.occupancy_state} | "
                f"count={event.person_count_estimate}"
            )
    except Exception:
        logger.error("Failed to list events", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()