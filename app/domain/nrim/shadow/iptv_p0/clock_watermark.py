from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import utc
from .source_inventory import ClockQuality


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class TimeDisposition(str, Enum):
    ACCEPT = "ACCEPT"
    LATE = "LATE"
    QUARANTINE = "QUARANTINE"


class ClockWatermarkPolicy(StrictModel):
    policy_id: str = Field(min_length=3, max_length=128)
    allowed_lateness_seconds: float = Field(ge=0.0)
    maximum_future_skew_seconds: float = Field(ge=0.0)
    maximum_observation_delay_seconds: float = Field(gt=0.0)
    unknown_clock_blocks: Literal[True] = True
    watermark_rule: str = "minimum_source_max_event_time_minus_lateness"


class TimeAssessment(StrictModel):
    disposition: TimeDisposition
    reason_code: str
    event_time_utc: datetime
    observation_time_utc: datetime
    ingestion_time_utc: datetime

    @field_validator(
        "event_time_utc",
        "observation_time_utc",
        "ingestion_time_utc",
    )
    @classmethod
    def normalize(cls, value: datetime) -> datetime:
        return utc(value)


def assess_record_time(
    *,
    event_time_utc: datetime,
    observation_time_utc: datetime,
    ingestion_time_utc: datetime,
    clock_quality: ClockQuality,
    policy: ClockWatermarkPolicy,
    watermark_utc: datetime | None = None,
) -> TimeAssessment:
    event_time = utc(event_time_utc)
    observation_time = utc(observation_time_utc)
    ingestion_time = utc(ingestion_time_utc)
    if clock_quality in {ClockQuality.UNKNOWN, ClockQuality.UNSYNCHRONIZED}:
        return TimeAssessment(
            disposition=TimeDisposition.QUARANTINE,
            reason_code="UNKNOWN_OR_UNSYNCHRONIZED_CLOCK",
            event_time_utc=event_time,
            observation_time_utc=observation_time,
            ingestion_time_utc=ingestion_time,
        )
    if event_time > ingestion_time + timedelta(
        seconds=policy.maximum_future_skew_seconds
    ):
        disposition = TimeDisposition.QUARANTINE
        reason = "FUTURE_EVENT_TIME"
    elif observation_time < event_time:
        disposition = TimeDisposition.QUARANTINE
        reason = "OBSERVATION_PRECEDES_EVENT"
    elif ingestion_time < observation_time:
        disposition = TimeDisposition.QUARANTINE
        reason = "INGESTION_PRECEDES_OBSERVATION"
    elif (
        observation_time - event_time
    ).total_seconds() > policy.maximum_observation_delay_seconds:
        disposition = TimeDisposition.QUARANTINE
        reason = "OBSERVATION_DELAY_EXCEEDED"
    elif watermark_utc is not None and event_time <= utc(watermark_utc):
        disposition = TimeDisposition.LATE
        reason = "AT_OR_BEHIND_WATERMARK"
    else:
        disposition = TimeDisposition.ACCEPT
        reason = "TIME_SEMANTICS_ACCEPTED"
    return TimeAssessment(
        disposition=disposition,
        reason_code=reason,
        event_time_utc=event_time,
        observation_time_utc=observation_time,
        ingestion_time_utc=ingestion_time,
    )


def deterministic_watermark(
    source_event_times: dict[str, Iterable[datetime]],
    *,
    policy: ClockWatermarkPolicy,
) -> datetime:
    if not source_event_times:
        raise ValueError("At least one source is required for a watermark")
    maxima = []
    for source_id, values in sorted(source_event_times.items()):
        normalized = tuple(utc(value) for value in values)
        if not normalized:
            raise ValueError(f"Source has no event times: {source_id}")
        maxima.append(max(normalized))
    return min(maxima) - timedelta(
        seconds=policy.allowed_lateness_seconds
    )
