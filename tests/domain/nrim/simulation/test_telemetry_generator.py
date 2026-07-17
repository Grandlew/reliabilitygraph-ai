import random
from datetime import datetime, timezone

import pytest

from app.domain.nrim.simulation.dataset_builder import (
    initialize_states,
)
from app.domain.nrim.simulation.telemetry_generator import (
    generate_catchup_service_telemetry,
    generate_storage_telemetry,
    sample_poisson,
)
from app.domain.nrim.simulation.topology_generator import (
    generate_hotel_topology,
)
from app.domain.nrim.telemetry import (
    GoldenSignal,
    TelemetryQuality,
    TelemetrySignalType,
)


def make_topology():
    return generate_hotel_topology(
        room_count=100,
        floor_count=4,
        redundant_middleware=False,
        shared_storage=True,
        seed=42,
    )


def make_states():
    topology = make_topology()
    timestamp = datetime.now(timezone.utc)

    states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    return topology, states, timestamp


def test_storage_generator_creates_expected_signals() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=76.5,
        rng=random.Random(42),
    )

    signal_names = {
        event.signal_name
        for event in events
    }

    assert signal_names == {
        "system.disk.utilization",
        "system.disk.io_latency",
        "system.disk.io_errors",
    }


def test_storage_events_reference_storage_component() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=80.0,
        rng=random.Random(42),
    )

    assert events

    assert all(
        event.component_node_id == "catchup_storage_1"
        for event in events
    )

    assert all(
        event.deployment_id == topology.topology_id
        for event in events
    )


def test_storage_events_have_metric_type() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=70.0,
        rng=random.Random(42),
    )

    assert all(
        event.signal_type == TelemetrySignalType.METRIC
        for event in events
    )


def test_storage_utilization_remains_bounded() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=99.9,
        rng=random.Random(42),
    )

    utilization_event = next(
        event
        for event in events
        if event.signal_name
        == "system.disk.utilization"
    )

    assert 0.0 <= float(utilization_event.value) <= 100.0
    assert utilization_event.unit == "percent"
    assert (
        utilization_event.golden_signal
        == GoldenSignal.SATURATION
    )


def test_negative_storage_utilization_is_clamped() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=-10.0,
        rng=random.Random(42),
    )

    utilization_event = next(
        event
        for event in events
        if event.signal_name
        == "system.disk.utilization"
    )

    assert 0.0 <= float(utilization_event.value) <= 100.0


def test_storage_latency_is_positive() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=70.0,
        rng=random.Random(42),
    )

    latency_event = next(
        event
        for event in events
        if event.signal_name
        == "system.disk.io_latency"
    )

    assert float(latency_event.value) > 0.0
    assert latency_event.unit == "milliseconds"
    assert (
        latency_event.golden_signal
        == GoldenSignal.LATENCY
    )


def test_storage_io_error_count_is_non_negative_integer() -> None:
    topology, states, timestamp = make_states()

    events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=70.0,
        rng=random.Random(42),
    )

    error_event = next(
        event
        for event in events
        if event.signal_name
        == "system.disk.io_errors"
    )

    assert isinstance(error_event.value, int)
    assert error_event.value >= 0
    assert error_event.unit == "errors_per_interval"
    assert (
        error_event.golden_signal
        == GoldenSignal.ERRORS
    )


def test_degraded_storage_increases_generated_latency() -> None:
    topology = make_topology()
    timestamp = datetime.now(timezone.utc)

    healthy_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states[
        "catchup_storage_1"
    ].latent_latency_factor = 5.0

    healthy_events = generate_storage_telemetry(
        topology=topology,
        node_state=healthy_states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=70.0,
        rng=random.Random(42),
    )

    degraded_events = generate_storage_telemetry(
        topology=topology,
        node_state=degraded_states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=70.0,
        rng=random.Random(42),
    )

    healthy_latency = next(
        float(event.value)
        for event in healthy_events
        if event.signal_name
        == "system.disk.io_latency"
    )

    degraded_latency = next(
        float(event.value)
        for event in degraded_events
        if event.signal_name
        == "system.disk.io_latency"
    )

    assert degraded_latency > healthy_latency


def test_degraded_storage_can_increase_io_errors() -> None:
    topology = make_topology()
    timestamp = datetime.now(timezone.utc)

    healthy_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states[
        "catchup_storage_1"
    ].latent_error_factor = 100.0

    healthy_total = 0
    degraded_total = 0

    for seed in range(100):
        healthy_events = generate_storage_telemetry(
            topology=topology,
            node_state=healthy_states["catchup_storage_1"],
            timestamp=timestamp,
            storage_utilization=70.0,
            rng=random.Random(seed),
        )

        degraded_events = generate_storage_telemetry(
            topology=topology,
            node_state=degraded_states["catchup_storage_1"],
            timestamp=timestamp,
            storage_utilization=70.0,
            rng=random.Random(seed),
        )

        healthy_total += next(
            int(event.value)
            for event in healthy_events
            if event.signal_name
            == "system.disk.io_errors"
        )

        degraded_total += next(
            int(event.value)
            for event in degraded_events
            if event.signal_name
            == "system.disk.io_errors"
        )

    assert degraded_total > healthy_total


def test_catchup_generator_creates_recording_failure_metric() -> None:
    topology, states, timestamp = make_states()

    events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=20,
        rng=random.Random(42),
    )

    assert len(events) == 1

    event = events[0]

    assert (
        event.signal_name
        == "iptv.catchup.recording_failures"
    )
    assert event.signal_type == TelemetrySignalType.METRIC
    assert event.component_node_id == "catchup_service_1"
    assert event.service_node_ids == ["catchup_service_1"]
    assert event.unit == "failures_per_interval"
    assert event.golden_signal == GoldenSignal.ERRORS
    assert isinstance(event.value, int)
    assert event.value >= 0


def test_catchup_failures_do_not_exceed_attempts() -> None:
    topology, states, timestamp = make_states()

    recording_attempts = 50

    events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=recording_attempts,
        rng=random.Random(42),
    )

    failures = int(events[0].value)

    assert 0 <= failures <= recording_attempts


def test_degraded_catchup_service_produces_more_failures() -> None:
    topology = make_topology()
    timestamp = datetime.now(timezone.utc)

    healthy_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states = initialize_states(
        topology=topology,
        timestamp=timestamp,
    )

    degraded_states[
        "catchup_service_1"
    ].latent_error_factor = 100.0

    healthy_events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=healthy_states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=1000,
        rng=random.Random(42),
    )

    degraded_events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=degraded_states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=1000,
        rng=random.Random(42),
    )

    healthy_failures = int(healthy_events[0].value)
    degraded_failures = int(degraded_events[0].value)

    assert degraded_failures > healthy_failures


def test_storage_generator_is_reproducible_with_same_seed() -> None:
    topology, states, timestamp = make_states()

    first = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=75.0,
        rng=random.Random(42),
    )

    second = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=75.0,
        rng=random.Random(42),
    )

    first_values = [
        (
            event.signal_name,
            event.value,
            event.quality,
        )
        for event in first
    ]

    second_values = [
        (
            event.signal_name,
            event.value,
            event.quality,
        )
        for event in second
    ]

    assert first_values == second_values


def test_catchup_generator_is_reproducible_with_same_seed() -> None:
    topology, states, timestamp = make_states()

    first = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=100,
        rng=random.Random(42),
    )

    second = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=100,
        rng=random.Random(42),
    )

    assert first[0].value == second[0].value
    assert first[0].quality == second[0].quality


def test_generated_quality_is_never_quarantined() -> None:
    topology, states, timestamp = make_states()

    storage_events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=75.0,
        rng=random.Random(42),
    )

    catchup_events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=20,
        rng=random.Random(42),
    )

    allowed_qualities = {
        TelemetryQuality.HIGH,
        TelemetryQuality.MEDIUM,
        TelemetryQuality.LOW,
    }

    assert all(
        event.quality in allowed_qualities
        for event in storage_events + catchup_events
    )


def test_generated_timestamps_are_preserved() -> None:
    topology, states, timestamp = make_states()

    storage_events = generate_storage_telemetry(
        topology=topology,
        node_state=states["catchup_storage_1"],
        timestamp=timestamp,
        storage_utilization=75.0,
        rng=random.Random(42),
    )

    catchup_events = generate_catchup_service_telemetry(
        topology=topology,
        node_state=states["catchup_service_1"],
        timestamp=timestamp,
        recording_attempts=20,
        rng=random.Random(42),
    )

    assert all(
        event.observed_at == timestamp
        and event.ingested_at == timestamp
        for event in storage_events + catchup_events
    )


def test_poisson_sampler_returns_zero_for_zero_rate() -> None:
    result = sample_poisson(
        rate=0.0,
        rng=random.Random(42),
    )

    assert result == 0


def test_poisson_sampler_rejects_negative_rate() -> None:
    with pytest.raises(ValueError):
        sample_poisson(
            rate=-1.0,
            rng=random.Random(42),
        )


def test_poisson_sampler_returns_non_negative_integer() -> None:
    result = sample_poisson(
        rate=3.0,
        rng=random.Random(42),
    )

    assert isinstance(result, int)
    assert result >= 0


def test_poisson_sampler_is_reproducible() -> None:
    first = sample_poisson(
        rate=3.0,
        rng=random.Random(42),
    )

    second = sample_poisson(
        rate=3.0,
        rng=random.Random(42),
    )

    assert first == second


def test_poisson_mean_is_close_to_requested_rate() -> None:
    rate = 4.0
    rng = random.Random(42)

    samples = [
        sample_poisson(
            rate=rate,
            rng=rng,
        )
        for _ in range(5000)
    ]

    observed_mean = sum(samples) / len(samples)

    assert abs(observed_mean - rate) < 0.25
