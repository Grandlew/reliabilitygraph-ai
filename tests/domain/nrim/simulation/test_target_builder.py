from datetime import datetime, timedelta, timezone

from app.domain.nrim.simulation.target_builder import (
    build_window_targets,
)
from app.domain.nrim.simulation.temporal_windowing import (
    TemporalWindowBoundary,
)


def make_boundary() -> TemporalWindowBoundary:
    cutoff = datetime(
        2026,
        7,
        18,
        12,
        tzinfo=timezone.utc,
    )

    return TemporalWindowBoundary(
        window_id="window_abc",
        observation_start=(
            cutoff - timedelta(hours=6)
        ),
        observation_cutoff=cutoff,
        prediction_end=(
            cutoff + timedelta(hours=6)
        ),
    )


def test_future_incident_target() -> None:
    cutoff = make_boundary().observation_cutoff

    hidden = {
        "ground_truth": {
            "failure_type": (
                "storage_io_degradation"
            ),
            "root_cause_node_id": "storage_1",
            "incident_onset_time": (
                cutoff
                + timedelta(hours=3)
            ).isoformat(),
            "affected_service_node_ids": [
                "catchup_1"
            ],
        }
    }

    targets = build_window_targets(
        node_ids=[
            "storage_1",
            "catchup_1",
        ],
        hidden_scenario=hidden,
        boundary=make_boundary(),
    )

    assert targets["future_incident"] == 1
    assert targets["current_incident"] == 0
    assert targets["root_cause_node"] == [1, 0]


def test_healthy_window_has_no_root_cause() -> None:
    hidden = {
        "ground_truth": {
            "failure_type": "healthy",
            "root_cause_node_id": None,
            "incident_onset_time": None,
            "affected_service_node_ids": [],
        }
    }

    targets = build_window_targets(
        node_ids=["storage_1", "catchup_1"],
        hidden_scenario=hidden,
        boundary=make_boundary(),
    )

    assert sum(
        targets["root_cause_node"]
    ) == 0
    assert targets["future_incident"] == 0
