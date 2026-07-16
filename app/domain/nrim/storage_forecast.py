from __future__ import annotations

from .temporal import (
    ForecastStatus,
    TemporalSeries,
    ThresholdForecast,
)
from .temporal_features import (
    assess_data_sufficiency,
    fit_median_pairwise_slope,
)


def forecast_threshold_crossing(
    series: TemporalSeries,
    *,
    threshold: float,
    maximum_forecast_hours: float = 168.0,
    minimum_trend_quality: float = 0.60,
) -> ThresholdForecast:
    latest_point = series.points[-1]

    if latest_point.value >= threshold:
        return ThresholdForecast(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            latest_observation_at=latest_point.timestamp,
            latest_value=latest_point.value,
            threshold=threshold,
            status=ForecastStatus.THRESHOLD_ALREADY_CROSSED,
            supporting_event_ids=[
                point.event_id
                for point in series.points
            ],
            limitations=[
                "The configured threshold is already crossed.",
                "This is a current-state condition, not a forecast.",
            ],
        )

    sufficiency = assess_data_sufficiency(series)

    if sufficiency.status.value != "valid":
        return ThresholdForecast(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            latest_observation_at=latest_point.timestamp,
            latest_value=latest_point.value,
            threshold=threshold,
            status=ForecastStatus.ABSTAINED,
            supporting_event_ids=[
                point.event_id
                for point in series.points
            ],
            abstention_reasons=sufficiency.reasons,
            limitations=[
                "Forecast suppressed because temporal data "
                "requirements were not satisfied."
            ],
        )

    trend = fit_median_pairwise_slope(series)

    if trend.slope_per_hour <= 0:
        return ThresholdForecast(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            latest_observation_at=latest_point.timestamp,
            latest_value=latest_point.value,
            threshold=threshold,
            status=ForecastStatus.STABLE_OR_DECREASING,
            slope_per_hour=trend.slope_per_hour,
            trend_quality=trend.quality_score,
            supporting_event_ids=[
                point.event_id
                for point in series.points
            ],
            assumptions=trend.assumptions,
            limitations=[
                "No positive exhaustion trend was detected.",
                "A future workload change may invalidate this state.",
            ],
        )

    if (
        not trend.stable
        or trend.quality_score < minimum_trend_quality
    ):
        return ThresholdForecast(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            latest_observation_at=latest_point.timestamp,
            latest_value=latest_point.value,
            threshold=threshold,
            status=ForecastStatus.ABSTAINED,
            slope_per_hour=trend.slope_per_hour,
            trend_quality=trend.quality_score,
            supporting_event_ids=[
                point.event_id
                for point in series.points
            ],
            assumptions=trend.assumptions,
            abstention_reasons=[
                "Recent trend is not sufficiently stable."
            ],
            limitations=[
                "An unstable trend would produce an unreliable "
                "threshold-crossing estimate."
            ],
        )

    central_hours = (
        threshold - latest_point.value
    ) / trend.slope_per_hour

    if central_hours > maximum_forecast_hours:
        return ThresholdForecast(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            latest_observation_at=latest_point.timestamp,
            latest_value=latest_point.value,
            threshold=threshold,
            status=ForecastStatus.ABSTAINED,
            slope_per_hour=trend.slope_per_hour,
            trend_quality=trend.quality_score,
            supporting_event_ids=[
                point.event_id
                for point in series.points
            ],
            assumptions=trend.assumptions,
            abstention_reasons=[
                "Estimated crossing lies beyond the supported "
                "forecast horizon."
            ],
            limitations=[
                f"Maximum supported horizon is "
                f"{maximum_forecast_hours:.1f} hours."
            ],
        )

    error_margin = max(trend.fit_mae, 0.25)

    pessimistic_current = min(
        threshold,
        latest_point.value + error_margin,
    )
    optimistic_current = max(
        0.0,
        latest_point.value - error_margin,
    )

    earliest_hours = (
        threshold - pessimistic_current
    ) / trend.slope_per_hour

    conservative_slope = max(
        trend.slope_per_hour * 0.65,
        1e-9,
    )

    latest_hours = (
        threshold - optimistic_current
    ) / conservative_slope

    earliest_hours = max(0.0, earliest_hours)
    central_hours = max(0.0, central_hours)
    latest_hours = max(central_hours, latest_hours)

    return ThresholdForecast(
        deployment_id=series.deployment_id,
        component_node_id=series.component_node_id,
        signal_name=series.signal_name,
        unit=series.unit,
        latest_observation_at=latest_point.timestamp,
        latest_value=latest_point.value,
        threshold=threshold,
        status=ForecastStatus.AVAILABLE,
        slope_per_hour=trend.slope_per_hour,
        central_hours_to_threshold=round(central_hours, 3),
        earliest_hours_to_threshold=round(
            earliest_hours,
            3,
        ),
        latest_hours_to_threshold=round(
            latest_hours,
            3,
        ),
        trend_quality=trend.quality_score,
        supporting_event_ids=[
            point.event_id
            for point in series.points
        ],
        assumptions=trend.assumptions,
        limitations=[
            "The interval is a conservative engineering estimate, "
            "not a calibrated probability interval.",
            "Changes in occupancy, retention, cleanup, or storage "
            "capacity may invalidate the forecast.",
        ],
    )
