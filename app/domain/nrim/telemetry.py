from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class TelemetrySignalType(str, Enum):
    METRIC = "metric"
    LOG = "log"
    ALARM = "alarm"
    SERVICE_EVENT = "service_event"
    CONFIGURATION_CHANGE = "configuration_change"
    SOFTWARE_UPDATE = "software_update"
    FIRMWARE_UPDATE = "firmware_update"
    MAINTENANCE_ACTION = "maintenance_action"
    EXTERNAL_PROBE = "external_probe"
    USER_REPORTED_SYMPTOM = "user_reported_symptom"


class GoldenSignal(str, Enum):
    LATENCY = "latency"
    TRAFFIC = "traffic"
    ERRORS = "errors"
    SATURATION = "saturation"


class TelemetryQuality(str, Enum):
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    QUARANTINED = "quarantined"


class BaselineStatus(str, Enum):
    UNKNOWN = "unknown"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NORMAL = "normal"
    WARNING = "warning"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"


class TelemetryAttribute(BaseModel):
    key: str = Field(min_length=1)
    value: str | int | float | bool
    quality: TelemetryQuality = TelemetryQuality.HIGH


class CanonicalTelemetryEvent(BaseModel):
    schema_version: str = "1.0.0"
    event_id: str = Field(default_factory=lambda: generate_id("telemetry"))

    deployment_id: str = Field(min_length=1)
    component_node_id: str = Field(min_length=1)
    service_node_ids: list[str] = Field(default_factory=list)

    observed_at: datetime
    ingested_at: datetime = Field(default_factory=utc_now)
    timezone_source: str | None = None
    clock_skew_ms: int | None = None

    signal_type: TelemetrySignalType
    signal_name: str = Field(
        min_length=3,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$",
    )

    value: Any
    unit: str | None = None
    attributes: list[TelemetryAttribute] = Field(default_factory=list)

    collection_source: str = Field(min_length=1)
    source_record_id: str | None = None

    quality: TelemetryQuality
    golden_signal: GoldenSignal | None = None
    baseline_status: BaselineStatus = BaselineStatus.UNKNOWN

    baseline_value: float | None = None
    deviation_score: float | None = None

    case_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_event_semantics(self) -> "CanonicalTelemetryEvent":
        future_tolerance = timedelta(minutes=5)

        if self.observed_at > self.ingested_at + future_tolerance:
            raise ValueError(
                "Observation occured more than 5 minutes after ingestion time")

        if self.signal_type == TelemetrySignalType.METRIC and not self.unit and isinstance(self.value, (int, float)):
            raise ValueError(
                "Unit must be provided when signal_type is METRIC and value is numeric")

        if self.quality == TelemetryQuality.QUARANTINED and self.baseline_status not in [BaselineStatus.UNKNOWN, BaselineStatus.INSUFFICIENT_HISTORY]:
            raise ValueError(
                "Quarantined event must have baseline_status UNKNOWN or INSUFFICIENT_HISTORY")

        if self.baseline_value is None and self.deviation_score is not None:
            raise ValueError(
                "deviation_score must be None when baseline_value is missing")

        if self.signal_type == TelemetrySignalType.USER_REPORTED_SYMPTOM and not self.evidence_ids:
            raise ValueError(
                "USER_REPORTED_SYMPTOM events must contain at least one evidence ID")

        return self


class TelemetryBatch(BaseModel):
    schema_version: str = "1.0.0"
    batch_id: str = Field(default_factory=lambda: generate_id("batch"))
    deployment_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    collected_at: datetime = Field(default_factory=utc_now)
    events: list[CanonicalTelemetryEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch(self) -> "TelemetryBatch":
        event_ids = {event.event_id for event in self.events}
        if len(self.events) != len(event_ids):
            raise ValueError("Duplicate event IDs found")

        if any(event.deployment_id != self.deployment_id for event in self.events):
            raise ValueError("Deployment ID mismatch in events")

        return self
