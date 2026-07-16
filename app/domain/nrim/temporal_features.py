from __future__ import annotations

from datetime import timedelta
from statistics import mean

from .telemetry import TelemetryQuality
from .temporal import (
    DataSufficiency,
    TemporalSeries,
    TemporalStatus,
    TrendEstimate,
)


QUALITY_RANK = {
    TelemetryQuality.VERIFIED: 4,
    TelemetryQuality.HIGH: 3,
    TelemetryQuality.MEDIUM: 2,
    TelemetryQuality.LOW: 1,
    TelemetryQuality.QUARANTINED: 0,
}


def assess_data_sufficiency(
    series: TemporalSeries,
    *,
    minimum_samples: int = 12,
    minimum_covered_hours: float = 3.0,
    minimum_high_quality_fraction: float = 0.70,
    maximum_allowed_gap_minutes: float = 90.0,
) -> DataSufficiency:
    points = series.points
    reasons: list[str] = []

    sample_count = len(points)

    if sample_count == 1:
        covered_hours = 0.0
        maximum_gap_minutes = None
    else:
        covered_hours = (
            points[-1].timestamp - points[0].timestamp
        ).total_seconds() / 3600.0

        gaps = [
            (
                points[index].timestamp
                - points[index - 1].timestamp
            ).total_seconds()
            / 60.0
            for index in range(1, len(points))
        ]

        maximum_gap_minutes = max(gaps)

    high_quality_count = sum(
        1
        for point in points
        if point.quality
        in {
            TelemetryQuality.VERIFIED,
            TelemetryQuality.HIGH,
        }
    )

    high_quality_fraction = (
        high_quality_count / sample_count
        if sample_count
        else 0.0
    )

    if sample_count < minimum_samples:
        reasons.append(
            f"Need at least {minimum_samples} samples; "
            f"received {sample_count}."
        )

    if covered_hours < minimum_covered_hours:
        reasons.append(
            f"Need at least {minimum_covered_hours:.1f} covered hours; "
            f"received {covered_hours:.2f}."
        )

    if high_quality_fraction < minimum_high_quality_fraction:
        reasons.append(
            "High-quality sample fraction is below the required "
            f"{minimum_high_quality_fraction:.0%}."
        )

    if (
        maximum_gap_minutes is not None
        and maximum_gap_minutes > maximum_allowed_gap_minutes
    ):
        reasons.append(
            f"Maximum gap {maximum_gap_minutes:.1f} minutes exceeds "
            f"the allowed {maximum_allowed_gap_minutes:.1f} minutes."
        )

    status = (
        TemporalStatus.VALID
        if not reasons
        else TemporalStatus.INSUFFICIENT_DATA
    )

    return DataSufficiency(
        status=status,
        sample_count=sample_count,
        covered_hours=round(covered_hours, 3),
        high_quality_fraction=round(high_quality_fraction, 3),
        maximum_gap_minutes=(
            round(maximum_gap_minutes, 3)
            if maximum_gap_minutes is not None
            else None
        ),
        reasons=reasons,
    )


def _hours_from_start(series: TemporalSeries) -> list[float]:
    start = series.points[0].timestamp

    return [
        (point.timestamp - start).total_seconds() / 3600.0
        for point in series.points
    ]


def fit_linear_trend(
    series: TemporalSeries,
) -> TrendEstimate:
    if len(series.points) < 2:
        raise ValueError("At least two points are required.")

    x = _hours_from_start(series)
    y = [point.value for point in series.points]

    x_mean = mean(x)
    y_mean = mean(y)

    denominator = sum(
        (value - x_mean) ** 2
        for value in x
    )

    if denominator == 0:
        raise ValueError("Temporal points have no usable time span.")

    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator

    intercept = y_mean - slope * x_mean

    predictions = [
        intercept + slope * x_value
        for x_value in x
    ]

    absolute_errors = [
        abs(actual - predicted)
        for actual, predicted in zip(y, predictions, strict=True)
    ]

    fit_mae = mean(absolute_errors)

    value_range = max(y) - min(y)

    normalized_error = (
        fit_mae / value_range
        if value_range > 0
        else fit_mae
    )

    quality_score = max(
        0.0,
        min(1.0, 1.0 - normalized_error),
    )

    stable = (
        quality_score >= 0.60
        and len(series.points) >= 6
    )

    return TrendEstimate(
        slope_per_hour=round(slope, 6),
        intercept=round(intercept, 6),
        fit_mae=round(fit_mae, 6),
        start_time=series.points[0].timestamp,
        end_time=series.points[-1].timestamp,
        sample_count=len(series.points),
        quality_score=round(quality_score, 4),
        stable=stable,
        assumptions=[
            "Recent linear growth approximates near-term behaviour.",
            "No major intervention changes the operating regime.",
            "Telemetry values and timestamps are valid.",
        ],
    )


def calculate_recent_rate_of_change(
    series: TemporalSeries,
    *,
    lookback_points: int = 6,
) -> float:
    if lookback_points < 2:
        raise ValueError("lookback_points must be at least 2.")

    selected_points = series.points[-lookback_points:]

    if len(selected_points) < 2:
        raise ValueError("Insufficient points for rate calculation.")

    selected_series = TemporalSeries(
        deployment_id=series.deployment_id,
        component_node_id=series.component_node_id,
        signal_name=series.signal_name,
        unit=series.unit,
        points=selected_points,
    )

    return fit_linear_trend(selected_series).slope_per_hour
