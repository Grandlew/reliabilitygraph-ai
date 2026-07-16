from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.domain.nrim.telemetry import TelemetryQuality
from app.domain.nrim.temporal import (
    ForecastStatus,
    TemporalPoint,
    TemporalSeries,
    ThresholdForecast,
)


def make_point(
    *,
    hour: int,
    value: float,
    event_id: str,
) -> TemporalPoint:
    return TemporalPoint(
        timestamp=datetime(
            2026,
            7,
            14,
            hour,
            tzinfo=timezone.utc,
        ),
        value=value,
        quality=TelemetryQuality.HIGH,
        event_id=event_id,
    )


def test_series_rejects_unordered_points() -> None:
    with pytest.raises(ValidationError):
        TemporalSeries(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            unit="percent",
            points=[
                make_point(
                    hour=10,
                    value=80.0,
                    event_id="event_1",
                ),
                make_point(
                    hour=9,
                    value=79.0,
                    event_id="event_2",
                ),
            ],
        )


def test_series_rejects_quarantined_point() -> None:
    with pytest.raises(ValidationError):
        TemporalSeries(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            unit="percent",
            points=[
                TemporalPoint(
                    timestamp=datetime.now(timezone.utc),
                    value=80.0,
                    quality=TelemetryQuality.QUARANTINED,
                    event_id="event_1",
                )
            ],
        )


def test_available_forecast_requires_crossing_times() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        ThresholdForecast(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            unit="percent",
            latest_observation_at=now,
            latest_value=80.0,
            threshold=95.0,
            status=ForecastStatus.AVAILABLE,
        )


def test_abstained_forecast_requires_reason() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        ThresholdForecast(
            deployment_id="deployment_1",
            component_node_id="storage_1",
            signal_name="system.disk.utilization",
            unit="percent",
            latest_observation_at=now,
            latest_value=80.0,
            threshold=95.0,
            status=ForecastStatus.ABSTAINED,
            abstention_reasons=[],
        )
