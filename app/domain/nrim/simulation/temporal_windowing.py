from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TemporalWindowSpecification(BaseModel):
    observation_hours: int = Field(gt=0)
    prediction_horizon_hours: int = Field(gt=0)
    stride_hours: int = Field(gt=0)
    minimum_event_count: int = Field(default=1, ge=1)


class TemporalWindowBoundary(BaseModel):
    window_id: str
    observation_start: datetime
    observation_cutoff: datetime
    prediction_end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "TemporalWindowBoundary":
        if not (
            self.observation_start
            < self.observation_cutoff
            < self.prediction_end
        ):
            raise ValueError(
                "Window timestamps are not strictly ordered."
            )

        return self


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def opaque_window_id(
    *,
    scenario_id: str,
    observation_start: datetime,
    observation_cutoff: datetime,
    prediction_end: datetime,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "observation_start": (
            observation_start.isoformat()
        ),
        "observation_cutoff": (
            observation_cutoff.isoformat()
        ),
        "prediction_end": prediction_end.isoformat(),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:16]

    return f"window_{digest}"


def scenario_time_range(
    scenario: dict[str, Any],
) -> tuple[datetime, datetime]:
    timestamps = [
        parse_timestamp(str(event["observed_at"]))
        for event in scenario.get("telemetry", [])
        if isinstance(event, dict)
        and event.get("observed_at")
    ]

    if not timestamps:
        raise ValueError(
            "Scenario contains no timestamped telemetry."
        )

    return min(timestamps), max(timestamps)


def generate_window_boundaries(
    *,
    scenario_id: str,
    scenario_start: datetime,
    scenario_end: datetime,
    specification: TemporalWindowSpecification,
) -> list[TemporalWindowBoundary]:
    observation_length = timedelta(
        hours=specification.observation_hours
    )
    prediction_length = timedelta(
        hours=specification.prediction_horizon_hours
    )
    stride = timedelta(
        hours=specification.stride_hours
    )

    cutoff = scenario_start + observation_length
    boundaries: list[TemporalWindowBoundary] = []

    while cutoff + prediction_length <= scenario_end:
        observation_start = cutoff - observation_length
        prediction_end = cutoff + prediction_length

        boundaries.append(
            TemporalWindowBoundary(
                window_id=opaque_window_id(
                    scenario_id=scenario_id,
                    observation_start=observation_start,
                    observation_cutoff=cutoff,
                    prediction_end=prediction_end,
                ),
                observation_start=observation_start,
                observation_cutoff=cutoff,
                prediction_end=prediction_end,
            )
        )

        cutoff += stride

    return boundaries


def events_in_observation_window(
    *,
    events: list[dict[str, Any]],
    boundary: TemporalWindowBoundary,
) -> list[dict[str, Any]]:
    selected = []

    for event in events:
        raw_timestamp = event.get("observed_at")

        if not raw_timestamp:
            continue

        timestamp = parse_timestamp(
            str(raw_timestamp)
        )

        if (
            boundary.observation_start
            <= timestamp
            <= boundary.observation_cutoff
        ):
            selected.append(event)

    return selected


def assert_no_future_events(
    *,
    events: list[dict[str, Any]],
    cutoff: datetime,
) -> None:
    future_events = [
        event
        for event in events
        if event.get("observed_at")
        and parse_timestamp(
            str(event["observed_at"])
        )
        > cutoff
    ]

    if future_events:
        raise ValueError(
            "Feature events exist after observation cutoff."
        )
