from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from pydantic import BaseModel, Field

from .storage_forecast import forecast_threshold_crossing
from .temporal import (
    ForecastStatus,
    TemporalSeries,
)


class ForecastEvaluation(BaseModel):
    cutoff_index: int
    forecast_status: ForecastStatus

    forecast_hours: float | None = None
    actual_hours: float | None = None
    absolute_error_hours: float | None = None

    warned_before_crossing: bool = False
    useful_lead_time_hours: float | None = None


class BacktestSummary(BaseModel):
    evaluation_count: int = Field(ge=0)
    available_forecast_count: int = Field(ge=0)
    abstention_count: int = Field(ge=0)

    threshold_crossing_observed: bool

    mean_absolute_error_hours: float | None = None
    early_warning_recall: float | None = None
    false_warning_count: int = Field(ge=0)

    evaluations: list[ForecastEvaluation]


def _find_actual_crossing_hours(
    series: TemporalSeries,
    *,
    cutoff_index: int,
    threshold: float,
) -> float | None:
    cutoff_time = series.points[cutoff_index].timestamp

    for point in series.points[cutoff_index + 1:]:
        if point.value >= threshold:
            return (
                point.timestamp - cutoff_time
            ).total_seconds() / 3600.0

    return None


def rolling_origin_backtest(
    series: TemporalSeries,
    *,
    threshold: float,
    minimum_training_points: int = 12,
    early_warning_horizon_hours: float = 72.0,
) -> BacktestSummary:
    if minimum_training_points < 3:
        raise ValueError(
            "minimum_training_points must be at least 3."
        )

    evaluations: list[ForecastEvaluation] = []

    for cutoff_index in range(
        minimum_training_points - 1,
        len(series.points) - 1,
    ):
        training_series = TemporalSeries(
            deployment_id=series.deployment_id,
            component_node_id=series.component_node_id,
            signal_name=series.signal_name,
            unit=series.unit,
            points=series.points[: cutoff_index + 1],
        )

        forecast = forecast_threshold_crossing(
            training_series,
            threshold=threshold,
        )

        actual_hours = _find_actual_crossing_hours(
            series,
            cutoff_index=cutoff_index,
            threshold=threshold,
        )

        forecast_hours = (
            forecast.central_hours_to_threshold
            if forecast.status == ForecastStatus.AVAILABLE
            else None
        )

        absolute_error = (
            abs(forecast_hours - actual_hours)
            if forecast_hours is not None
            and actual_hours is not None
            else None
        )

        warned_before_crossing = (
            forecast_hours is not None
            and forecast_hours <= early_warning_horizon_hours
            and actual_hours is not None
        )

        useful_lead_time = (
            actual_hours
            if warned_before_crossing
            else None
        )

        evaluations.append(
            ForecastEvaluation(
                cutoff_index=cutoff_index,
                forecast_status=forecast.status,
                forecast_hours=forecast_hours,
                actual_hours=actual_hours,
                absolute_error_hours=absolute_error,
                warned_before_crossing=warned_before_crossing,
                useful_lead_time_hours=useful_lead_time,
            )
        )

    available = [
        item
        for item in evaluations
        if item.forecast_status == ForecastStatus.AVAILABLE
    ]

    errors = [
        item.absolute_error_hours
        for item in evaluations
        if item.absolute_error_hours is not None
    ]

    crossing_evaluations = [
        item
        for item in evaluations
        if item.actual_hours is not None
    ]

    successful_warnings = sum(
        1
        for item in crossing_evaluations
        if item.warned_before_crossing
    )

    early_warning_recall = (
        successful_warnings / len(crossing_evaluations)
        if crossing_evaluations
        else None
    )

    false_warning_count = sum(
        1
        for item in evaluations
        if item.forecast_hours is not None
        and item.forecast_hours <= early_warning_horizon_hours
        and item.actual_hours is None
    )

    return BacktestSummary(
        evaluation_count=len(evaluations),
        available_forecast_count=len(available),
        abstention_count=sum(
            1
            for item in evaluations
            if item.forecast_status == ForecastStatus.ABSTAINED
        ),
        threshold_crossing_observed=bool(crossing_evaluations),
        mean_absolute_error_hours=(
            round(mean(errors), 3)
            if errors
            else None
        ),
        early_warning_recall=(
            round(early_warning_recall, 3)
            if early_warning_recall is not None
            else None
        ),
        false_warning_count=false_warning_count,
        evaluations=evaluations,
    )
