from datetime import datetime

from app.db import SessionLocal
from app.logger import get_logger
from app.schemas import EventCreate
from app.services.event_service import create_event

logger = get_logger(__name__)


def main() -> None:
    payload = EventCreate(
        timestamp=datetime.now(),
        camera_id="door_cam_01",
        zone="entrance",
        event_type="occupancy_change",
        occupancy_state="occupied",
        person_count_estimate=2,
        confidence=0.92,
        image_uri="./data/images/sample_001.jpg",
        clip_uri=None,
        privacy_flags='["face_blur", "screen_mask"]',
        raw_metadata='{"source":"manual_test","note":"step_c_insert"}',
    )

    db = SessionLocal()
    try:
        event = create_event(db, payload)

        print("Event created successfully:")
        print(f"  id={event.id}")
        print(f"  timestamp={event.timestamp}")
        print(f"  camera_id={event.camera_id}")
        print(f"  zone={event.zone}")
        print(f"  event_type={event.event_type}")
        print(f"  occupancy_state={event.occupancy_state}")
        print(f"  person_count_estimate={event.person_count_estimate}")
        print(f"  confidence={event.confidence}")
    except Exception:
        logger.error("Failed to create event", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()