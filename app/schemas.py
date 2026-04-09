from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EventCreate(BaseModel):
    timestamp: datetime
    camera_id: str = Field(..., min_length=1, max_length=100)
    zone: Optional[str] = Field(default=None, max_length=100)

    event_type: str = Field(..., min_length=1, max_length=100)
    occupancy_state: Optional[str] = Field(default=None, max_length=50)

    person_count_estimate: Optional[int] = Field(default=None, ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    image_uri: Optional[str] = None
    clip_uri: Optional[str] = None

    privacy_flags: Optional[str] = None
    raw_metadata: Optional[str] = None


class EventRead(BaseModel):
    id: int
    timestamp: datetime
    camera_id: str
    zone: Optional[str]
    event_type: str
    occupancy_state: Optional[str]
    person_count_estimate: Optional[int]
    confidence: Optional[float]
    image_uri: Optional[str]
    clip_uri: Optional[str]
    privacy_flags: Optional[str]
    raw_metadata: Optional[str]
    created_at: datetime

    model_config = {
        "from_attributes": True
    }