from typing import Optional
from pathlib import Path
import json

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


def _delete_file(file_path: str | None) -> None:
    if not file_path:
        return

    path = Path(file_path)

    if not path.exists():
        logger.warning("File not found, skip deleting: path=%s", path)
        return

    if not path.is_file():
        logger.warning("Path is not a file, skip deleting: path=%s", path)
        return

    path.unlink()
    logger.info("Deleted file: path=%s", path)


def _extract_debug_image_path(raw_metadata: str | None) -> str | None:
    if not raw_metadata:
        return None

    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        logger.warning("Failed to parse raw_metadata JSON")
        return None

    detector_signals = metadata.get("detector_signals", {})
    if not isinstance(detector_signals, dict):
        return None

    debug_image_path = detector_signals.get("debug_image_path")
    if isinstance(debug_image_path, str) and debug_image_path.strip():
        return debug_image_path

    return None


def _collect_event_file_paths(event: Event) -> list[str]:
    file_paths: list[str] = []

    if event.image_uri:
        file_paths.append(event.image_uri)

    debug_image_path = _extract_debug_image_path(event.raw_metadata)
    if debug_image_path:
        file_paths.append(debug_image_path)

    deduped_paths: list[str] = []
    seen: set[str] = set()

    for file_path in file_paths:
        normalized = str(Path(file_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_paths.append(file_path)

    return deduped_paths


def delete_event_by_id(db: Session, event_id: int) -> bool:
    event = get_event_by_id(db, event_id)
    if event is None:
        logger.warning("Event not found: id=%s", event_id)
        return False

    file_paths = _collect_event_file_paths(event)

    db.delete(event)
    db.commit()

    for file_path in file_paths:
        try:
            _delete_file(file_path)
        except Exception:
            logger.exception(
                "Failed to delete file after deleting event: id=%s file_path=%s",
                event_id,
                file_path,
            )

    logger.info(
        "Event deleted successfully: id=%s deleted_file_count=%s",
        event_id,
        len(file_paths),
    )
    return True


def delete_all_events(db: Session) -> int:
    events = list_events(db)
    deleted_count = 0
    all_file_paths: list[str] = []

    for event in events:
        all_file_paths.extend(_collect_event_file_paths(event))
        db.delete(event)
        deleted_count += 1

    db.commit()

    deduped_paths: list[str] = []
    seen: set[str] = set()

    for file_path in all_file_paths:
        normalized = str(Path(file_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_paths.append(file_path)

    for file_path in deduped_paths:
        try:
            _delete_file(file_path)
        except Exception:
            logger.exception(
                "Failed to delete file while deleting all events: file_path=%s",
                file_path,
            )

    logger.info(
        "All events deleted successfully: count=%s deleted_file_count=%s",
        deleted_count,
        len(deduped_paths),
    )
    return deleted_count