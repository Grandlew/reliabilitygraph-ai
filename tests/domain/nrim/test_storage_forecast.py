from datetime import datetime, timedelta, timezone

from app.domain.nrim.storage_forecast import (
    forecast_threshold_crossing,
)
from app.domain.nrim.telemetry import TelemetryQuality
from app.domain.nrim.temporal import (
    ForecastStatus,
    TemporalPoint,
    TemporalSeries,
)


def make_series(
    *,
    initial: float = 70.0,
    slope: float = 1.0,
    count: int = 16,
) -> TemporalSeries:
    start = datetime(
        2026,
        7,
        14,
        8,
        tzinfo=timezone.utc,
    )

    return TemporalSeries(
        deployment_id="deployment_1",
        component_node_id="storage_1",
        signal_name="system.disk.utilization",
        unit="percent",
        points=[
            TemporalPoint(
                timestamp=start + timedelta(hours=index),
                value=initial + slope * index,
                quality=TelemetryQuality.HIGH,
                event_id=f"event_{index}",
            )
            for index in range(count)
        ],
    )


def test_increasing_series_produces_forecast() -> None:
    forecast = forecast_threshold_crossing(
        make_series(),
        threshold=95.0,
    )

    assert forecast.status == ForecastStatus.AVAILABLE
    assert forecast.central_hours_to_threshold is not None


def test_decreasing_series_does_not_forecast_exhaustion() -> None:
    forecast = forecast_threshold_crossing(
        make_series(initial=90.0, slope=-0.5),
        threshold=95.0,
    )

    assert (
        forecast.status
        == ForecastStatus.STABLE_OR_DECREASING
    )


def test_crossed_threshold_is_current_state() -> None:
    forecast = forecast_threshold_crossing(
        make_series(initial=96.0, slope=0.2),
        threshold=95.0,
    )

    assert (
        forecast.status
        == ForecastStatus.THRESHOLD_ALREADY_CROSSED
    )


def test_short_series_abstains() -> None:
    forecast = forecast_threshold_crossing(
        make_series(count=5),
        threshold=95.0,
    )

    assert forecast.status == ForecastStatus.ABSTAINED
