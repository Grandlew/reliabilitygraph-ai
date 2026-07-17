import random
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.nrim.simulation.dataset_builder import (
    build_scenario,
    recorded_storage_growth_gb,
    sample_injection_time,
    sanitize_observable_message,
    utilization_increment_percent,
)
from app.domain.nrim.simulation.models import (
    FailureType,
    FaultSpecification,
    OperatingRegime,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)


def test_storage_growth_uses_capacity_dimensions() -> None:
    growth_gb = recorded_storage_growth_gb(
        channel_count=40,
        average_bitrate_mbps=5.0,
        interval_seconds=900,
    )

    assert growth_gb == pytest.approx(22.5)
    assert utilization_increment_percent(
        growth_gb=growth_gb,
        usable_capacity_tb=16.0,
    ) == pytest.approx(0.140625)


def test_injection_time_is_reproducible_and_in_range() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)

    first = sample_injection_time(
        start_time=start,
        end_time=end,
        sampling_interval_minutes=15,
        rng=random.Random(42),
    )
    second = sample_injection_time(
        start_time=start,
        end_time=end,
        sampling_interval_minutes=15,
        rng=random.Random(42),
    )

    assert first == second
    assert start + timedelta(hours=6) <= first
    assert first <= end - timedelta(hours=6)
    assert (first - start) % timedelta(minutes=15) == timedelta(0)


def test_observable_message_does_not_expose_failure_class() -> None:
    message = (
        "Storage I/O degradation detected; "
        "CATCHUP WORKER FAILURE suspected"
    )

    sanitized = sanitize_observable_message(message)

    assert "storage i/o degradation" not in sanitized.lower()
    assert "catchup worker failure" not in sanitized.lower()
    assert sanitized.count("component condition") == 2


def test_missingness_mask_is_independent_of_fault_label() -> None:
    seed = 17
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    topology = generate_hotel_topology(
        room_count=80,
        floor_count=2,
        redundant_middleware=False,
        shared_storage=True,
        seed=seed,
    )
    regime = OperatingRegime(
        duration_hours=4,
        sampling_interval_minutes=15,
        base_occupancy_fraction=0.6,
        evening_peak_multiplier=1.2,
        weekend_multiplier=1.0,
        catchup_recording_channels=24,
        average_bitrate_mbps=4.0,
        retention_days=7,
        telemetry_missing_probability=0.4,
        random_seed=seed,
    )
    fault = FaultSpecification(
        failure_type=FailureType.STORAGE_IO_DEGRADATION,
        target_node_id="catchup_storage_1",
        injection_time=start + timedelta(hours=2),
        severity=0.9,
    )

    healthy = build_scenario(
        scenario_id="scenario_control",
        topology=topology,
        regime=regime,
        start_time=start,
        fault=None,
    )
    faulty = build_scenario(
        scenario_id="scenario_fault",
        topology=topology,
        regime=regime,
        start_time=start,
        fault=fault,
    )

    def observation_mask(scenario: dict) -> set[tuple[str, str, str]]:
        return {
            (
                event["component_node_id"],
                event["signal_name"],
                event["observed_at"],
            )
            for event in scenario["telemetry"]
        }

    assert observation_mask(healthy.observable) == observation_mask(
        faulty.observable
    )
