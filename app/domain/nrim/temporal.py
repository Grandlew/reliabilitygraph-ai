from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .telemetry import TelemetryQuality


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class TemporalStatus(str, Enum):
    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID = "invalid"
    ABSTAINED = "abstained"


class ForecastStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    AVAILABLE = "available"
    THRESHOLD_ALREADY_CROSSED = "threshold_already_crossed"
    STABLE_OR_DECREASING = "stable_or_decreasing"
    ABSTAINED = "abstained"


class WarningTier(str, Enum):
    NONE = "none"
    WATCH = "watch"
    EARLY_WARNING = "early_warning"
    HIGH_RISK = "high_risk"
    ACTIVE_INCIDENT = "active_incident"


class TemporalPoint(BaseModel):
    timestamp: datetime
    value: float
    quality: TelemetryQuality
    event_id: str = Field(min_length=1)


class TemporalSeries(BaseModel):
    deployment_id: str = Field(min_length=1)
    component_node_id: str = Field(min_length=1)
    signal_name: str = Field(min_length=3)
    unit: str = Field(min_length=1)
    points: list[TemporalPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_series(self) -> "TemporalSeries":
        timestamps = [p.timestamp for p in self.points]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(
                "Duplicate timestamps are not allowed in a series")
        for i in range(len(self.points) - 1):
            if self.points[i].timestamp >= self.points[i+1].timestamp:
                raise ValueError("Points must be strictly ordered by time")
        if any(p.quality == TelemetryQuality.QUARANTINED for p in self.points):
            raise ValueError(
                "Temporal series cannot contain quarantined points")
        return self


class DataSufficiency(BaseModel):
    status: TemporalStatus
    sample_count: int = Field(ge=0)
    covered_hours: float = Field(ge=0.0)
    high_quality_fraction: float = Field(ge=0.0, le=1.0)
    maximum_gap_minutes: float | None = Field(default=None, ge=0.0)
    reasons: list[str] = Field(default_factory=list)


class TrendEstimate(BaseModel):
    slope_per_hour: float
    intercept: float
    fit_mae: float = Field(ge=0.0)
    start_time: datetime
    end_time: datetime
    sample_count: int = Field(ge=2)
    quality_score: float = Field(ge=0.0, le=1.0)
    stable: bool
    assumptions: list[str] = Field(default_factory=list)


class ThresholdForecast(BaseModel):
    forecast_id: str = Field(
        default_factory=lambda: generate_id("forecast")
    )

    deployment_id: str
    component_node_id: str
    signal_name: str
    unit: str

    generated_at: datetime = Field(default_factory=utc_now)
    latest_observation_at: datetime
    latest_value: float

    threshold: float
    status: ForecastStatus

    slope_per_hour: float | None = None
    central_hours_to_threshold: float | None = None
    earliest_hours_to_threshold: float | None = None
    latest_hours_to_threshold: float | None = None

    trend_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    supporting_event_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    abstention_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_forecast(self) -> "ThresholdForecast":
        if self.status == ForecastStatus.AVAILABLE:
            if self.central_hours_to_threshold is None or \
               self.earliest_hours_to_threshold is None or \
               self.latest_hours_to_threshold is None:
                raise ValueError(
                    "AVAILABLE forecasts must include all threshold-crossing estimates")
        if self.status == ForecastStatus.ABSTAINED and not self.abstention_reasons:
            raise ValueError(
                "ABSTAINED forecasts must include at least one reason")

        if self.status == ForecastStatus.AVAILABLE:
            if not (0 <= self.earliest_hours_to_threshold <= self.central_hours_to_threshold <= self.latest_hours_to_threshold):
                raise ValueError(
                    "Crossing estimates must be non-negative and ordered (earliest <= central <= latest)")
        return self


class PrecursorEvidence(BaseModel):
    evidence_id: str
    signal_name: str
    event_ids: list[str] = Field(min_length=1)
    statement: str = Field(min_length=1)
    strength: float = Field(ge=0.0, le=1.0)
    quality: TelemetryQuality


class EarlyWarning(BaseModel):
    warning_id: str = Field(
        default_factory=lambda: generate_id("warning")
    )

    case_id: str
    deployment_id: str
    component_node_id: str
    failure_signature_id: str

    generated_at: datetime = Field(default_factory=utc_now)
    tier: WarningTier

    title: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)

    evidence: list[PrecursorEvidence] = Field(default_factory=list)
    forecast_id: str | None = None

    recommended_next_test: str | None = None
    requires_engineer_review: bool = True

    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_warning(self) -> "EarlyWarning":
        if self.tier in {WarningTier.EARLY_WARNING, WarningTier.HIGH_RISK} and not self.evidence:
            raise ValueError(
                f"{self.tier} requires at least one piece of evidence")

        if self.tier in {WarningTier.HIGH_RISK, WarningTier.ACTIVE_INCIDENT} and not self.requires_engineer_review:
            raise ValueError(f"{self.tier} tier must require engineer review")
        return self
