from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    zone: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    event_type: Mapped[str] = mapped_column(String(100), index=True)
    occupancy_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    person_count_estimate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    image_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    clip_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    privacy_flags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    analyses: Mapped[list["EventAnalysis"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )


class EventAnalysis(Base):
    __tablename__ = "event_analysis"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)

    scene_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    activity_labels: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    anomaly_flags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    free_text_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    event: Mapped["Event"] = relationship(back_populates="analyses")


class TimeWindowSummary(Base):
    __tablename__ = "time_window_summary"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, index=True)

    camera_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    zone: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    total_events: Mapped[int] = mapped_column(Integer, default=0)
    avg_person_count: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_person_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    anomaly_flags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Rule(Base):
    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rule_type: Mapped[str] = mapped_column(String(100), index=True)
    rule_key: Mapped[str] = mapped_column(String(100), index=True)
    rule_value: Mapped[str] = mapped_column(Text)

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)