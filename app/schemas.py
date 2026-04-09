from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


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
        "from_attributes": True,
    }


class PrivacyRegion(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class PrivacyConfig(BaseModel):
    roi: Optional[PrivacyRegion] = None
    mask_regions: list[PrivacyRegion] = Field(default_factory=list)
    blur_regions: list[PrivacyRegion] = Field(default_factory=list)
    mask_color: tuple[int, int, int] = (0, 0, 0)
    blur_kernel_size: int = 31

    @model_validator(mode="after")
    def validate_blur_kernel_size(self) -> "PrivacyConfig":
        if self.blur_kernel_size <= 0:
            raise ValueError("blur_kernel_size must be > 0")
        return self


class CapturedImage(BaseModel):
    source_type: str
    source_value: str
    frame: Any
    width: int
    height: int
    captured_at: datetime

    model_config = {
        "arbitrary_types_allowed": True,
    }


class ProcessedImageResult(BaseModel):
    source_type: str
    source_value: str
    output_path: str
    width: int
    height: int
    privacy_flags: list[str]
    captured_at: datetime
    processed_at: datetime