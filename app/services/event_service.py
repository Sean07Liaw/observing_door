from typing import Optional

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


def delete_event_by_id(db: Session, event_id: int) -> bool:
    event = db.query(Event).filter(Event.id == event_id).first()

    if event is None:
        logger.info("Delete skipped; event not found: id=%s", event_id)
        return False

    db.delete(event)
    db.commit()

    logger.info("Deleted event: id=%s", event_id)
    return True


def delete_all_events(db: Session) -> int:
    events = db.query(Event).all()
    count = len(events)

    if count == 0:
        logger.info("No events to delete")
        return 0

    for event in events:
        db.delete(event)

    db.commit()

    logger.info("Deleted all events: count=%s", count)
    return count