from __future__ import annotations

import json
import random
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
    FaultSpecification,
    HealthState,
    HiddenNodeState,
    OperatingRegime,
    ScenarioKind,
)
from .operating_regime import workload_at_time
from .propagation import propagate_fault
from .telemetry_generator import (
    generate_catchup_service_telemetry,
    generate_storage_telemetry,
)


@dataclass
class GeneratedScenario:
    observable: dict[str, Any]
    hidden: dict[str, Any]


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
) -> GeneratedScenario:
    rng = random.Random(regime.random_seed)

    states = initialize_states(
        topology=topology,
        timestamp=start_time,
    )

    telemetry_events = []
    propagation_records = []

    storage_utilization = 55.0
    incident_onset_time = None
    affected_service_ids: set[str] = set()

    current_time = start_time
    step = timedelta(
        minutes=regime.sampling_interval_minutes
    )
    end_time = start_time + timedelta(
        hours=regime.duration_hours
    )

    fault_injected = False

    while current_time <= end_time:
        workload = workload_at_time(
            timestamp=current_time,
            room_count=topology.room_count,
            regime=regime,
            rng=rng,
        )

        if (
            fault is not None
            and not fault_injected
            and current_time >= fault.injection_time
        ):
            states = inject_fault(
                topology=topology,
                fault=fault,
                states=states,
            )

            states, new_records = propagate_fault(
                topology=topology,
                fault=fault,
                states=states,
                timestamp=current_time,
            )

            propagation_records.extend(new_records)
            fault_injected = True

        storage_state = states["catchup_storage_1"]
        catchup_state = states["catchup_service_1"]

        base_growth = (
            regime.catchup_recording_channels
            * regime.average_bitrate_mbps
            * step.total_seconds()
            / 8.0
            / 1_000_000.0
        )

        growth_multiplier = (
            2.5
            if fault is not None
            and fault_injected
            and fault.failure_type.value
            in {
                "storage_capacity_saturation",
                "cleanup_job_failure",
            }
            else 1.0
        )

        storage_utilization += (
            base_growth * growth_multiplier
        )

        telemetry_events.extend(
            generate_storage_telemetry(
                topology=topology,
                node_state=storage_state,
                timestamp=current_time,
                storage_utilization=storage_utilization,
                rng=rng,
            )
        )

        recording_attempts = max(
            1,
            int(regime.catchup_recording_channels / 6),
        )

        catchup_events = (
            generate_catchup_service_telemetry(
                topology=topology,
                node_state=catchup_state,
                timestamp=current_time,
                recording_attempts=recording_attempts,
                rng=rng,
            )
        )

        telemetry_events.extend(catchup_events)

        current_failures = int(
            catchup_events[0].value
        )

        if current_failures > 0:
            affected_service_ids.add(
                "catchup_service_1"
            )

            if incident_onset_time is None:
                incident_onset_time = current_time

        current_time += step

    node_ids = [
        node.node_id
        for node in topology.nodes
    ]

    if fault is None:
        ground_truth = build_healthy_ground_truth(
            scenario_id=scenario_id,
            random_seed=regime.random_seed,
        )
    else:
        ground_truth = build_fault_ground_truth(
            scenario_id=scenario_id,
            fault=fault,
            propagation_records=propagation_records,
            affected_service_node_ids=list(
                affected_service_ids
            ),
            incident_onset_time=incident_onset_time,
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
            event.model_dump(mode="json")
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
            fault.model_dump(mode="json")
            if fault is not None
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
    channel_count: int,
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
