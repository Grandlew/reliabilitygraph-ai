import random
from datetime import datetime, timezone

from app.domain.nrim.simulation.confounders import (
    apply_environment_confounders,
    apply_observable_confounders,
    confounder_event_records,
)
from app.domain.nrim.simulation.scenario_design import (
    ConfounderType,
)


def test_high_workload_confounder_increases_occupancy() -> None:
    values = {
        "base_occupancy_fraction": 0.40,
        "evening_peak_multiplier": 1.10,
        "retention_days": 7,
    }

    updated = apply_environment_confounders(
        environment_values=values,
        confounders=[
            ConfounderType.HIGH_HEALTHY_WORKLOAD
        ],
        rng=random.Random(42),
    )

    assert (
        updated["base_occupancy_fraction"]
        > values["base_occupancy_fraction"]
    )


def test_retention_confounder_increases_retention() -> None:
    values = {
        "base_occupancy_fraction": 0.60,
        "evening_peak_multiplier": 1.20,
        "retention_days": 7,
    }

    updated = apply_environment_confounders(
        environment_values=values,
        confounders=[
            ConfounderType.RETENTION_INCREASE
        ],
        rng=random.Random(42),
    )

    assert (
        updated["retention_days"]
        > values["retention_days"]
    )


def test_latency_confounder_changes_only_latency() -> None:
    telemetry = [
        {
            "signal_name": (
                "system.disk.io_latency"
            ),
            "value": 10.0,
            "attributes": [],
        },
        {
            "signal_name": (
                "system.disk.utilization"
            ),
            "value": 70.0,
            "attributes": [],
        },
    ]

    updated = apply_observable_confounders(
        telemetry=telemetry,
        confounders=[
            ConfounderType.TEMPORARY_LATENCY_SPIKE
        ],
        rng=random.Random(42),
    )

    latency = next(
        item
        for item in updated
        if item["signal_name"]
        == "system.disk.io_latency"
    )

    utilization = next(
        item
        for item in updated
        if item["signal_name"]
        == "system.disk.utilization"
    )

    assert latency["value"] > 10.0
    assert utilization["value"] == 70.0


def test_confounder_events_do_not_name_root_cause() -> None:
    records = confounder_event_records(
        confounders=[
            ConfounderType.RETENTION_INCREASE,
            ConfounderType.TEMPORARY_LATENCY_SPIKE,
        ],
        start_time=datetime.now(timezone.utc),
        duration_hours=24,
        rng=random.Random(42),
    )

    serialized = str(records).lower()

    assert "root cause" not in serialized
