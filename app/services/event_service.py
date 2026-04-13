from typing import Optional
from pathlib import Path

from sqlalchemy.orm import Session

from app.logger import get_logger
from app.models import Event
from app.schemas import EventCreate

logger = get_logger(__name__)


def create_event(db: Session, payload: EventCreate) -> Event:
    logger.info(
        "Creating event: camera_id=%s event_type=%s zone=%s",
        payload.camera_id,
        payload.event_type,
        payload.zone,
    )

    event = Event(**payload.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)

    logger.info("Event created successfully: id=%s", event.id)
    return event


def list_events(
    db: Session,
    limit: Optional[int] = None,
    newest_first: bool = True,
) -> list[Event]:
    query = db.query(Event)

    if newest_first:
        query = query.order_by(Event.id.desc())
    else:
        query = query.order_by(Event.id.asc())

    if limit is not None:
        query = query.limit(limit)

    events = query.all()
    logger.info("Listed %s event(s)", len(events))
    return events


def get_event_by_id(db: Session, event_id: int) -> Optional[Event]:
    event = db.query(Event).filter(Event.id == event_id).first()

    if event is None:
        logger.info("Event not found: id=%s", event_id)
    else:
        logger.info("Fetched event: id=%s", event_id)

    return event

def _delete_image_file(image_uri: str | None) -> None:
    if not image_uri:
        return

    path = Path(image_uri)

    if not path.exists():
        logger.warning("Image file not found, skip deleting: path=%s", path)
        return

    if not path.is_file():
        logger.warning("Image path is not a file, skip deleting: path=%s", path)
        return

    path.unlink()
    logger.info("Deleted image file: path=%s", path)

def delete_event_by_id(db: Session, event_id: int) -> bool:
    event = get_event_by_id(db, event_id)
    if event is None:
        logger.warning("Event not found: id=%s", event_id)
        return False

    image_uri = event.image_uri

    db.delete(event)
    db.commit()

    try:
        _delete_image_file(image_uri)
    except Exception:
        logger.exception(
            "Failed to delete image file after deleting event: id=%s image_uri=%s",
            event_id,
            image_uri,
        )

    logger.info("Event deleted successfully: id=%s", event_id)
    return True


def delete_all_events(db: Session) -> int:
    events = list_events(db)
    deleted_count = 0
    image_uris: list[str] = []

    for event in events:
        if event.image_uri:
            image_uris.append(event.image_uri)
        db.delete(event)
        deleted_count += 1

    db.commit()

    for image_uri in image_uris:
        try:
            _delete_image_file(image_uri)
        except Exception:
            logger.exception(
                "Failed to delete image file while deleting all events: image_uri=%s",
                image_uri,
            )

    logger.info("All events deleted successfully: count=%s", deleted_count)
    return deleted_count