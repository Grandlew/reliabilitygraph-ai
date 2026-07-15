from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from .telemetry import BaselineStatus, CanonicalTelemetryEvent


class BaselineWindow(BaseModel):
    deployment_id: str = Field(min_length=1)
    component_node_id: str = Field(min_length=1)
    signal_name: str = Field(min_length=1)

    hour_of_day: int | None = Field(default=None, ge=0, le=23)
    day_of_week: int | None = Field(default=None, ge=0, le=6)

    sample_count: int = Field(ge=0)

    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0.0)
    median: float | None = None
    p95: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    window_start: datetime
    window_end: datetime

    @model_validator(mode="after")
    def validate_statistics(self) -> "BaselineWindow":

        if self.window_end < self.window_start:
            raise ValueError("window_end cannot be before window_start")

        if self.sample_count == 0:
            stats = [
                self.mean, self.standard_deviation, self.median,
                self.p95, self.minimum, self.maximum
            ]
            if any(s is not None for s in stats):
                raise ValueError(
                    "Statistical values must be None if sample_count is zero")
        return self


class BaselineAssessment(BaseModel):
    event_id: str
    status: BaselineStatus
    baseline_value: float | None = None
    deviation_score: float | None = None
    reason: str


def assess_against_baseline(
    event: CanonicalTelemetryEvent,
    baseline: BaselineWindow | None,
    *,
    minimum_samples: int = 30,
    warning_z: float = 2.0,
    anomaly_z: float = 3.0,
    critical_z: float = 5.0,
) -> BaselineAssessment:
    if baseline is None or baseline.sample_count < minimum_samples:
        return BaselineAssessment(
            event_id=event.event_id,
            status=BaselineStatus.INSUFFICIENT_HISTORY,
            reason="Not enough historical samples for a learned baseline.",
        )

    if not isinstance(event.value, (int, float)):
        return BaselineAssessment(
            event_id=event.event_id,
            status=BaselineStatus.UNKNOWN,
            reason="Baseline assessment requires a numeric value.",
        )

    if baseline.mean is None or baseline.standard_deviation is None:
        return BaselineAssessment(
            event_id=event.event_id,
            status=BaselineStatus.UNKNOWN,
            reason="Baseline statistics are incomplete.",
        )

    if baseline.standard_deviation == 0:
        deviation_score = (
            0.0
            if float(event.value) == baseline.mean
            else float("inf")
        )
    else:
        deviation_score = abs(
            (float(event.value) - baseline.mean)
            / baseline.standard_deviation
        )

    if deviation_score >= critical_z:
        status = BaselineStatus.CRITICAL
    elif deviation_score >= anomaly_z:
        status = BaselineStatus.ANOMALOUS
    elif deviation_score >= warning_z:
        status = BaselineStatus.WARNING
    else:
        status = BaselineStatus.NORMAL

    return BaselineAssessment(
        event_id=event.event_id,
        status=status,
        baseline_value=baseline.mean,
        deviation_score=deviation_score,
        reason=f"Z-score {deviation_score:.2f} compared to thresholds ({warning_z}, {anomaly_z}, {critical_z})"
    )
