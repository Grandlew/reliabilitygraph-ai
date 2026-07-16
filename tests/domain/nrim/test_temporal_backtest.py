from datetime import datetime, timedelta, timezone

from app.domain.nrim.telemetry import TelemetryQuality
from app.domain.nrim.temporal import (
    TemporalPoint,
    TemporalSeries,
)
from app.domain.nrim.temporal_backtest import (
    rolling_origin_backtest,
)


def make_crossing_series() -> TemporalSeries:
    start = datetime(
        2026,
        7,
        14,
        0,
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
                value=70.0 + index,
                quality=TelemetryQuality.HIGH,
                event_id=f"event_{index}",
            )
            for index in range(30)
        ],
    )


def test_backtest_uses_multiple_cutoffs() -> None:
    result = rolling_origin_backtest(
        make_crossing_series(),
        threshold=95.0,
        minimum_training_points=12,
    )

    assert result.evaluation_count > 1


def test_backtest_observes_threshold_crossing() -> None:
    result = rolling_origin_backtest(
        make_crossing_series(),
        threshold=95.0,
        minimum_training_points=12,
    )

    assert result.threshold_crossing_observed is True


def test_backtest_reports_available_forecasts() -> None:
    result = rolling_origin_backtest(
        make_crossing_series(),
        threshold=95.0,
        minimum_training_points=12,
    )

    assert result.available_forecast_count > 0
