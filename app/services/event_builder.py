from __future__ import annotations

import json
from datetime import datetime

from app.logger import get_logger
from app.schemas import EventCreate

logger = get_logger(__name__)


def normalize_person_count_estimate(value) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        if value < 0:
            return None
        return min(value, 2)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"2+", "2 +", "2plus", "many", "multiple"}:
            return 2
        if normalized.isdigit():
            return min(int(normalized), 2)

    return None


def build_observation_event(
    *,
    camera_id: str,
    zone: str | None,
    image_result: dict,
    detection_result: dict,
    event_type: str = "space_observation",
    timestamp: datetime | None = None,
) -> EventCreate:
    event_timestamp = timestamp or datetime.now()

    occupancy_state = detection_result.get("scene_state")
    person_count_estimate = normalize_person_count_estimate(
        detection_result.get("person_count_estimate")
    )
    confidence = detection_result.get("confidence")

    privacy_flags_json = json.dumps(
        image_result.get("privacy_flags", {}),
        ensure_ascii=False,
    )

    raw_metadata = {
        "source_type": image_result.get("source_type"),
        "source_ref": image_result.get("source_ref"),
        "image_metadata": image_result.get("metadata", {}),
        "detector_signals": detection_result.get("signals", {}),
    }
    raw_metadata_json = json.dumps(raw_metadata, ensure_ascii=False)

    event = EventCreate(
        timestamp=event_timestamp,
        camera_id=camera_id,
        zone=zone,
        event_type=event_type,
        occupancy_state=occupancy_state,
        person_count_estimate=person_count_estimate,
        confidence=confidence,
        image_uri=image_result.get("output_path"),
        clip_uri=None,
        privacy_flags=privacy_flags_json,
        raw_metadata=raw_metadata_json,
    )

    logger.info(
        "Built observation event: camera_id=%s zone=%s event_type=%s occupancy_state=%s person_count_estimate=%s",
        camera_id,
        zone,
        event_type,
        occupancy_state,
        person_count_estimate,
    )
    return event