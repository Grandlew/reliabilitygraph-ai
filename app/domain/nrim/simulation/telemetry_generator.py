from __future__ import annotations

import math
import random
from datetime import datetime

from ..telemetry import (
    BaselineStatus,
    CanonicalTelemetryEvent,
    GoldenSignal,
    TelemetryQuality,
    TelemetrySignalType,
)
from .models import (
    DeploymentTopology,
    HiddenNodeState,
    SimulationNodeType,
)


def _quality_from_rng(
    rng: random.Random,
) -> TelemetryQuality:
    draw = rng.random()

    if draw < 0.70:
        return TelemetryQuality.HIGH
    if draw < 0.90:
        return TelemetryQuality.MEDIUM
    return TelemetryQuality.LOW


def sample_poisson(
    rng: random.Random,
    rate: float,
) -> int:
    """Sample a Poisson count using Knuth's exact algorithm.

    This implementation is intended for the simulator's small event
    rates and preserves reproducibility through the supplied Random
    instance.
    """
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("Poisson rate must be finite and non-negative")

    if rate == 0.0:
        return 0

    threshold = math.exp(-rate)
    product = 1.0
    count = 0

    while product > threshold:
        count += 1
        product *= rng.random()

    return count - 1


def apply_telemetry_missingness(
    events: list[CanonicalTelemetryEvent],
    *,
    missing_probability: float,
    rng: random.Random,
) -> list[CanonicalTelemetryEvent]:
    """Drop observations independently of fault state and labels."""
    if not 0.0 <= missing_probability <= 1.0:
        raise ValueError(
            "missing_probability must be between zero and one"
        )

    return [
        event
        for event in events
        if rng.random() >= missing_probability
    ]


def generate_storage_telemetry(
    *,
    topology: DeploymentTopology,
    node_state: HiddenNodeState,
    timestamp: datetime,
    storage_utilization: float,
    rng: random.Random,
) -> list[CanonicalTelemetryEvent]:
    events: list[CanonicalTelemetryEvent] = []

    effective_utilization = (
        storage_utilization
        / max(
            0.10,
            node_state.latent_capacity_factor,
        )
    )
    observed_utilization = max(
        0.0,
        min(
            100.0,
            effective_utilization
            + rng.gauss(0.0, 0.35),
        ),
    )

    base_latency_ms = 8.0

    observed_latency = max(
        0.1,
        base_latency_ms
        * node_state.latent_latency_factor
        + rng.gauss(0.0, 1.5),
    )

    io_error_count = sample_poisson(
        rng,
        0.1 * node_state.latent_error_factor,
    )

    common = {
        "deployment_id": topology.topology_id,
        "component_node_id": node_state.node_id,
        "observed_at": timestamp,
        "ingested_at": timestamp,
        "signal_type": TelemetrySignalType.METRIC,
        "collection_source": "nrim_simulator",
        "quality": _quality_from_rng(rng),
        "baseline_status": BaselineStatus.UNKNOWN,
    }

    events.append(
        CanonicalTelemetryEvent(
            **common,
            signal_name="system.disk.utilization",
            value=round(observed_utilization, 3),
            unit="percent",
            golden_signal=GoldenSignal.SATURATION,
        )
    )

    events.append(
        CanonicalTelemetryEvent(
            **common,
            signal_name="system.disk.io_latency",
            value=round(observed_latency, 3),
            unit="milliseconds",
            golden_signal=GoldenSignal.LATENCY,
        )
    )

    events.append(
        CanonicalTelemetryEvent(
            **common,
            signal_name="system.disk.io_errors",
            value=io_error_count,
            unit="errors_per_interval",
            golden_signal=GoldenSignal.ERRORS,
        )
    )

    return events


def generate_catchup_service_telemetry(
    *,
    topology: DeploymentTopology,
    node_state: HiddenNodeState,
    timestamp: datetime,
    recording_attempts: int,
    active_sessions: float | None = None,
    rng: random.Random,
) -> list[CanonicalTelemetryEvent]:
    failure_probability = min(
        0.95,
        0.002 * node_state.latent_error_factor,
    )

    failures = sum(
        1
        for _ in range(recording_attempts)
        if rng.random() < failure_probability
    )

    events = [
        CanonicalTelemetryEvent(
            deployment_id=topology.topology_id,
            component_node_id=node_state.node_id,
            service_node_ids=[node_state.node_id],
            observed_at=timestamp,
            ingested_at=timestamp,
            signal_type=TelemetrySignalType.METRIC,
            signal_name=(
                "iptv.catchup.recording_failures"
            ),
            value=failures,
            unit="failures_per_interval",
            collection_source="nrim_simulator",
            quality=_quality_from_rng(rng),
            golden_signal=GoldenSignal.ERRORS,
            baseline_status=BaselineStatus.UNKNOWN,
        )
    ]
    if active_sessions is not None:
        if active_sessions < 0.0:
            raise ValueError(
                "active_sessions must be non-negative"
            )
        health_availability = max(
            0.05,
            min(
                1.0,
                1.0
                - 0.85
                * (
                    1.0
                    - 1.0
                    / max(
                        1.0,
                        node_state.latent_error_factor,
                    )
                ),
            ),
        )
        observed_sessions = max(
            0.0,
            active_sessions
            * health_availability
            + rng.gauss(
                0.0,
                max(0.25, active_sessions * 0.02),
            ),
        )
        events.append(
            CanonicalTelemetryEvent(
                deployment_id=topology.topology_id,
                component_node_id=node_state.node_id,
                service_node_ids=[node_state.node_id],
                observed_at=timestamp,
                ingested_at=timestamp,
                signal_type=TelemetrySignalType.METRIC,
                signal_name="iptv.session.active_count",
                value=round(observed_sessions, 3),
                unit="sessions",
                collection_source="nrim_simulator",
                quality=_quality_from_rng(rng),
                golden_signal=GoldenSignal.TRAFFIC,
                baseline_status=BaselineStatus.UNKNOWN,
            )
        )
        observed_availability = max(
            0.0,
            min(
                100.0,
                100.0 * health_availability
                + rng.gauss(0.0, 0.35),
            ),
        )
        events.append(
            CanonicalTelemetryEvent(
                deployment_id=topology.topology_id,
                component_node_id=node_state.node_id,
                service_node_ids=[node_state.node_id],
                observed_at=timestamp,
                ingested_at=timestamp,
                signal_type=TelemetrySignalType.METRIC,
                signal_name=(
                    "iptv.catchup.service_availability"
                ),
                value=round(
                    observed_availability,
                    3,
                ),
                unit="percent",
                collection_source="nrim_simulator",
                quality=_quality_from_rng(rng),
                golden_signal=GoldenSignal.TRAFFIC,
                baseline_status=BaselineStatus.UNKNOWN,
            )
        )
    return events
