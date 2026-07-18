from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.simulation.temporal_windowing import (
    TemporalWindowSpecification,
    assert_no_future_events,
    events_in_observation_window,
    generate_window_boundaries,
    opaque_window_id,
)


def test_window_identifier_is_opaque() -> None:
    start = datetime(
        2026,
        7,
        18,
        tzinfo=timezone.utc,
    )

    identifier = opaque_window_id(
        scenario_id="scenario_abcdef",
        observation_start=start,
        observation_cutoff=(
            start + timedelta(hours=6)
        ),
        prediction_end=(
            start + timedelta(hours=12)
        ),
    )

    assert identifier.startswith("window_")
    assert "scenario" not in identifier
    assert "storage" not in identifier
    assert "healthy" not in identifier


def test_window_generation_is_deterministic() -> None:
    start = datetime(
        2026,
        7,
        18,
        tzinfo=timezone.utc,
    )

    specification = TemporalWindowSpecification(
        observation_hours=6,
        prediction_horizon_hours=6,
        stride_hours=2,
    )

    first = generate_window_boundaries(
        scenario_id="scenario_abc",
        scenario_start=start,
        scenario_end=start + timedelta(hours=24),
        specification=specification,
    )

    second = generate_window_boundaries(
        scenario_id="scenario_abc",
        scenario_start=start,
        scenario_end=start + timedelta(hours=24),
        specification=specification,
    )

    assert first == second


def test_event_after_cutoff_is_rejected() -> None:
    cutoff = datetime(
        2026,
        7,
        18,
        12,
        tzinfo=timezone.utc,
    )

    events = [
        {
            "observed_at": (
                cutoff
                + timedelta(minutes=1)
            ).isoformat()
        }
    ]

    with pytest.raises(ValueError):
        assert_no_future_events(
            events=events,
            cutoff=cutoff,
        )
