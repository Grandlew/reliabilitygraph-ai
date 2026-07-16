from datetime import datetime, timedelta, timezone

from app.domain.nrim.telemetry import TelemetryQuality
from app.domain.nrim.temporal import (
    TemporalPoint,
    TemporalSeries,
    TemporalStatus,
)
from app.domain.nrim.temporal_features import (
    assess_data_sufficiency,
    fit_linear_trend,
    fit_median_pairwise_slope,
)


def make_series(
    *,
    sample_count: int = 12,
    slope_per_hour: float = 1.0,
) -> TemporalSeries:
    start = datetime(
        2026,
        7,
        14,
        8,
        tzinfo=timezone.utc,
    )

    points = [
        TemporalPoint(
            timestamp=start + timedelta(hours=index),
            value=70.0 + slope_per_hour * index,
            quality=TelemetryQuality.HIGH,
            event_id=f"event_{index}",
        )
        for index in range(sample_count)
    ]

    return TemporalSeries(
        deployment_id="deployment_1",
        component_node_id="storage_1",
        signal_name="system.disk.utilization",
        unit="percent",
        points=points,
    )


def test_sufficient_series_is_valid() -> None:
    assessment = assess_data_sufficiency(make_series())

    assert assessment.status == TemporalStatus.VALID


def test_short_series_is_insufficient() -> None:
    assessment = assess_data_sufficiency(
        make_series(sample_count=5)
    )

    assert (
        assessment.status
        == TemporalStatus.INSUFFICIENT_DATA
    )


def test_linear_trend_recovers_slope() -> None:
    trend = fit_linear_trend(
        make_series(slope_per_hour=1.5)
    )

    assert abs(trend.slope_per_hour - 1.5) < 1e-6


def test_robust_trend_handles_outlier() -> None:
    series = make_series(slope_per_hour=1.0)

    modified_points = list(series.points)
    modified_points[5] = modified_points[5].model_copy(
        update={"value": 200.0}
    )

    modified_series = series.model_copy(
        update={"points": modified_points}
    )

    trend = fit_median_pairwise_slope(modified_series)

    assert abs(trend.slope_per_hour - 1.0) < 0.2
