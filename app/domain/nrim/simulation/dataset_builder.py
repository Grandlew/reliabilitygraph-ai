from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .fault_injector import inject_fault
from .labeler import (
    build_fault_ground_truth,
    build_healthy_ground_truth,
    root_cause_node_labels,
)
from .models import (
    DeploymentTopology,
    FailureType,
    FaultSpecification,
    HealthState,
    HiddenNodeState,
    OperatingRegime,
    ScenarioKind,
    SimulationEdgeType,
    SimulationNode,
    SimulationNodeType,
)
from .operating_regime import workload_at_time
from .propagation import propagate_fault
from .telemetry_generator import (
    apply_telemetry_missingness,
    generate_catchup_service_telemetry,
    generate_storage_telemetry,
)


@dataclass
class GeneratedScenario:
    observable: dict[str, Any]
    hidden: dict[str, Any]


def _random_stream(seed: int, name: str) -> random.Random:
    """Create a stable, independent random stream for one mechanism."""
    return random.Random(f"nrim-simulator:{seed}:{name}")


def sample_injection_time(
    *,
    start_time: datetime,
    end_time: datetime,
    sampling_interval_minutes: int,
    rng: random.Random,
    warmup_fraction: float = 0.25,
    cooldown_fraction: float = 0.25,
) -> datetime:
    """Choose a reproducible, schedule-aligned fault injection time."""
    if end_time <= start_time:
        raise ValueError("end_time must be after start_time")
    if sampling_interval_minutes <= 0:
        raise ValueError("sampling_interval_minutes must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= cooldown_fraction < 1.0:
        raise ValueError("cooldown_fraction must be in [0, 1)")
    if warmup_fraction + cooldown_fraction >= 1.0:
        raise ValueError("warmup and cooldown must leave an injection range")

    step = timedelta(minutes=sampling_interval_minutes)
    step_count = int((end_time - start_time) // step)
    earliest_step = max(1, math.ceil(step_count * warmup_fraction))
    latest_step = min(
        step_count - 1,
        math.floor(step_count * (1.0 - cooldown_fraction)),
    )

    if earliest_step > latest_step:
        raise ValueError("scenario is too short for the injection range")

    return start_time + step * rng.randint(earliest_step, latest_step)


def sanitize_observable_message(message: str) -> str:
    """Remove explicit failure-class names from model-visible text."""
    sanitized = message

    for failure_type in FailureType:
        if failure_type == FailureType.HEALTHY:
            continue

        variants = {
            failure_type.value,
            failure_type.value.replace("_", " "),
            failure_type.value.replace(
                "_",
                " ",
            ).replace("storage io", "storage i/o"),
            failure_type.name,
        }
        for variant in variants:
            sanitized = re.sub(
                re.escape(variant),
                "component condition",
                sanitized,
                flags=re.IGNORECASE,
            )

    return sanitized


def _sanitize_observable_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_observable_message(value)
    if isinstance(value, list):
        return [_sanitize_observable_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_observable_value(child)
            for key, child in value.items()
        }
    return value


def _observable_event_dump(event: Any) -> dict[str, Any]:
    dumped = event.model_dump(mode="json")
    dumped["value"] = _sanitize_observable_value(dumped["value"])
    dumped["attributes"] = _sanitize_observable_value(
        dumped.get("attributes", [])
    )
    return dumped


def _nodes_of_type(
    topology: DeploymentTopology,
    node_type: SimulationNodeType,
) -> list[SimulationNode]:
    return sorted(
        (
            node
            for node in topology.nodes
            if node.node_type == node_type
        ),
        key=lambda node: node.node_id,
    )


def initialize_states(
    *,
    topology: DeploymentTopology,
    timestamp: datetime,
) -> dict[str, HiddenNodeState]:
    return {
        node.node_id: HiddenNodeState(
            timestamp=timestamp,
            node_id=node.node_id,
            health_state=HealthState.HEALTHY,
        )
        for node in topology.nodes
    }


def build_scenario(
    *,
    scenario_id: str,
    topology: DeploymentTopology,
    regime: OperatingRegime,
    start_time: datetime,
    fault: FaultSpecification | None,
    randomize_injection_time: bool = False,
) -> GeneratedScenario:
    workload_rng = _random_stream(
        regime.random_seed,
        "workload",
    )
    telemetry_rng = _random_stream(
        regime.random_seed,
        "telemetry",
    )
    missingness_rng = _random_stream(
        regime.random_seed,
        "missingness",
    )
    injection_rng = _random_stream(
        regime.random_seed,
        "injection",
    )

    states = initialize_states(
        topology=topology,
        timestamp=start_time,
    )

    telemetry_events = []
    propagation_records = []

    storage_nodes = _nodes_of_type(
        topology,
        SimulationNodeType.CATCHUP_STORAGE,
    )
    catchup_service_nodes = _nodes_of_type(
        topology,
        SimulationNodeType.CATCHUP_SERVICE,
    )
    storage_utilization = {
        node.node_id: 55.0
        for node in storage_nodes
    }

    service_by_storage = {
        edge.target_node_id: edge.source_node_id
        for edge in topology.edges
        if edge.edge_type == SimulationEdgeType.STORES_ON
    }
    incident_onset_time = None
    fault_recovery_time = None
    observable_impact_times: list[datetime] = []
    affected_service_ids: set[str] = set()

    current_time = start_time
    step = timedelta(
        minutes=regime.sampling_interval_minutes
    )
    end_time = start_time + timedelta(
        hours=regime.duration_hours
    )

    scenario_fault = fault
    if fault is not None and randomize_injection_time:
        scenario_fault = fault.model_copy(
            update={
                "injection_time": sample_injection_time(
                    start_time=start_time,
                    end_time=end_time,
                    sampling_interval_minutes=(
                        regime.sampling_interval_minutes
                    ),
                    rng=injection_rng,
                )
            }
        )

    if (
        scenario_fault is not None
        and not start_time <= scenario_fault.injection_time <= end_time
    ):
        raise ValueError(
            "fault injection_time must fall within the scenario"
        )

    fault_injected = False
    fault_active = False
    configured_duration_hours = (
        float(
            scenario_fault.parameters.get(
                "fault_duration_hours"
            )
        )
        if (
            scenario_fault is not None
            and scenario_fault.parameters.get(
                "fault_duration_hours"
            )
            is not None
        )
        else None
    )

    while current_time <= end_time:
        workload = workload_at_time(
            timestamp=current_time,
            room_count=topology.room_count,
            regime=regime,
            rng=workload_rng,
        )

        if (
            scenario_fault is not None
            and not fault_injected
            and current_time >= scenario_fault.injection_time
        ):
            states = inject_fault(
                topology=topology,
                fault=scenario_fault,
                states=states,
            )

            states, new_records = propagate_fault(
                topology=topology,
                fault=scenario_fault,
                states=states,
                timestamp=current_time,
            )

            propagation_records.extend(new_records)
            fault_injected = True
            fault_active = True

        if (
            scenario_fault is not None
            and fault_active
            and configured_duration_hours is not None
            and current_time
            >= scenario_fault.injection_time
            + timedelta(hours=configured_duration_hours)
        ):
            states = initialize_states(
                topology=topology,
                timestamp=current_time,
            )
            fault_active = False
            fault_recovery_time = current_time

        channels_per_storage = (
            regime.catchup_recording_channels
            / len(storage_nodes)
        )

        for storage_node in storage_nodes:
            storage_state = states[storage_node.node_id]
            growth_multiplier = 1.0

            if scenario_fault is not None and fault_active:
                if (
                    scenario_fault.failure_type
                    == FailureType.STORAGE_CAPACITY_SATURATION
                    and scenario_fault.target_node_id
                    == storage_node.node_id
                ):
                    growth_multiplier = float(
                        scenario_fault.parameters.get(
                            "capacity_growth_multiplier",
                            2.5,
                        )
                    )
                elif (
                    scenario_fault.failure_type
                    == FailureType.CLEANUP_JOB_FAILURE
                    and scenario_fault.target_node_id
                    == service_by_storage.get(storage_node.node_id)
                ):
                    growth_multiplier = float(
                        scenario_fault.parameters.get(
                            "storage_growth_multiplier",
                            1.8,
                        )
                    )

            growth_gb = recorded_storage_growth_gb(
                channel_count=channels_per_storage,
                average_bitrate_mbps=(
                    regime.average_bitrate_mbps
                ),
                interval_seconds=step.total_seconds(),
            )
            capacity_tb = float(
                storage_node.capacity["usable_capacity_tb"]
            )
            storage_utilization[storage_node.node_id] += (
                utilization_increment_percent(
                    growth_gb=growth_gb * growth_multiplier,
                    usable_capacity_tb=capacity_tb,
                )
            )

            storage_events = generate_storage_telemetry(
                topology=topology,
                node_state=storage_state,
                timestamp=current_time,
                storage_utilization=storage_utilization[
                    storage_node.node_id
                ],
                rng=telemetry_rng,
            )
            telemetry_events.extend(
                apply_telemetry_missingness(
                    storage_events,
                    missing_probability=(
                        regime.telemetry_missing_probability
                    ),
                    rng=missingness_rng,
                )
            )

        attempts_per_service = max(
            1,
            int(
                regime.catchup_recording_channels
                / len(catchup_service_nodes)
                / 6
            ),
        )

        for service_node in catchup_service_nodes:
            catchup_state = states[service_node.node_id]
            expected_sessions = (
                float(
                    workload[
                        "active_catchup_sessions"
                    ]
                )
                / len(catchup_service_nodes)
            )
            catchup_events = generate_catchup_service_telemetry(
                topology=topology,
                node_state=catchup_state,
                timestamp=current_time,
                recording_attempts=attempts_per_service,
                active_sessions=expected_sessions,
                rng=telemetry_rng,
            )
            telemetry_events.extend(
                apply_telemetry_missingness(
                    catchup_events,
                    missing_probability=(
                        regime.telemetry_missing_probability
                    ),
                    rng=missingness_rng,
                )
            )

            current_failures = int(
                next(
                    event.value
                    for event in catchup_events
                    if event.signal_name
                    == "iptv.catchup.recording_failures"
                )
            )
            observed_sessions = float(
                next(
                    event.value
                    for event in catchup_events
                    if event.signal_name
                    == "iptv.session.active_count"
                )
            )

            fault_attributable_impact = (
                scenario_fault is not None
                and fault_active
                and catchup_state.health_state
                != HealthState.HEALTHY
                and (
                    current_failures > 0
                    or observed_sessions
                    < expected_sessions * 0.85
                )
            )

            if fault_attributable_impact:
                affected_service_ids.add(service_node.node_id)
                observable_impact_times.append(current_time)

                if incident_onset_time is None:
                    incident_onset_time = current_time

        current_time += step

    node_ids = [
        node.node_id
        for node in topology.nodes
    ]

    if scenario_fault is None:
        ground_truth = build_healthy_ground_truth(
            scenario_id=scenario_id,
            random_seed=regime.random_seed,
        )
    else:
        ground_truth = build_fault_ground_truth(
            scenario_id=scenario_id,
            fault=scenario_fault,
            propagation_records=propagation_records,
            affected_service_node_ids=list(
                affected_service_ids
            ),
            incident_onset_time=incident_onset_time,
            recovery_time=fault_recovery_time,
            observable_impact_times=observable_impact_times,
            random_seed=regime.random_seed,
        )

    labels = {
        "root_cause_node": root_cause_node_labels(
            node_ids=node_ids,
            ground_truth=ground_truth,
        ),
        "failure_type": ground_truth.failure_type.value,
        "affected_services": {
            node_id: int(
                node_id
                in ground_truth.affected_service_node_ids
            )
            for node_id in node_ids
        },
    }

    observable = {
        "schema_version": "0.1.0",
        "scenario_id": scenario_id,
        "topology": topology.model_dump(mode="json"),
        "telemetry": [
            _observable_event_dump(event)
            for event in telemetry_events
        ],
        "labels": labels,
    }

    hidden = {
        "schema_version": "0.1.0",
        "scenario_id": scenario_id,
        "ground_truth": ground_truth.model_dump(
            mode="json"
        ),
        "final_hidden_states": {
            node_id: state.model_dump(mode="json")
            for node_id, state in states.items()
        },
        "propagation_records": [
            record.model_dump(mode="json")
            for record in propagation_records
        ],
        "fault": (
            scenario_fault.model_dump(mode="json")
            if scenario_fault is not None
            else None
        ),
    }

    return GeneratedScenario(
        observable=observable,
        hidden=hidden,
    )


def save_scenario(
    *,
    scenario: GeneratedScenario,
    output_dir: Path,
    scenario_id: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    observable_path = (
        output_dir
        / f"{scenario_id}.observable.json"
    )
    hidden_path = (
        output_dir
        / f"{scenario_id}.hidden.json"
    )

    with observable_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scenario.observable,
            file,
            indent=2,
        )

    with hidden_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scenario.hidden,
            file,
            indent=2,
        )


def recorded_storage_growth_gb(
    *,
    channel_count: float,
    average_bitrate_mbps: float,
    interval_seconds: float,
) -> float:
    if channel_count <= 0:
        raise ValueError(
            "channel_count must be positive."
        )

    if average_bitrate_mbps <= 0:
        raise ValueError(
            "average_bitrate_mbps must be positive."
        )

    if interval_seconds <= 0:
        raise ValueError(
            "interval_seconds must be positive."
        )

    total_megabits = (
        channel_count
        * average_bitrate_mbps
        * interval_seconds
    )

    total_megabytes = total_megabits / 8.0
    total_gigabytes = total_megabytes / 1000.0

    return total_gigabytes


def utilization_increment_percent(
    *,
    growth_gb: float,
    usable_capacity_tb: float,
) -> float:
    if usable_capacity_tb <= 0:
        raise ValueError(
            "usable_capacity_tb must be positive."
        )

    usable_capacity_gb = usable_capacity_tb * 1000.0

    return growth_gb / usable_capacity_gb * 100.0
