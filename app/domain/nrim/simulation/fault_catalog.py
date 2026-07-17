from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import FailureType, SimulationNodeType


@dataclass(frozen=True)
class FaultDefinition:
    failure_type: FailureType
    valid_target_types: tuple[SimulationNodeType, ...]
    default_parameters: dict[str, Any]
    expected_service_types: tuple[
        SimulationNodeType,
        ...
    ]
    description: str


FAULT_CATALOG: dict[FailureType, FaultDefinition] = {
    FailureType.STORAGE_CAPACITY_SATURATION: FaultDefinition(
        failure_type=FailureType.STORAGE_CAPACITY_SATURATION,
        valid_target_types=(
            SimulationNodeType.CATCHUP_STORAGE,
        ),
        default_parameters={
            "capacity_growth_multiplier": 2.5,
            "cleanup_effectiveness": 0.0,
            "critical_utilization": 0.95,
        },
        expected_service_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        description=(
            "Storage consumption accelerates until usable "
            "capacity becomes operationally unsafe."
        ),
    ),
    FailureType.STORAGE_IO_DEGRADATION: FaultDefinition(
        failure_type=FailureType.STORAGE_IO_DEGRADATION,
        valid_target_types=(
            SimulationNodeType.CATCHUP_STORAGE,
        ),
        default_parameters={
            "latency_multiplier": 4.0,
            "write_success_factor": 0.60,
            "io_error_rate": 0.02,
        },
        expected_service_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        description=(
            "Storage write latency and errors increase while "
            "capacity may remain available."
        ),
    ),
    FailureType.CLEANUP_JOB_FAILURE: FaultDefinition(
        failure_type=FailureType.CLEANUP_JOB_FAILURE,
        valid_target_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        default_parameters={
            "cleanup_success_probability": 0.0,
            "storage_growth_multiplier": 1.8,
        },
        expected_service_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        description=(
            "Expired recordings are no longer removed, "
            "accelerating storage growth."
        ),
    ),
    FailureType.CATCHUP_WORKER_FAILURE: FaultDefinition(
        failure_type=FailureType.CATCHUP_WORKER_FAILURE,
        valid_target_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        default_parameters={
            "worker_availability": 0.45,
            "restart_probability": 0.08,
        },
        expected_service_types=(
            SimulationNodeType.CATCHUP_SERVICE,
        ),
        description=(
            "Recording workers become unstable independently "
            "of storage health."
        ),
    ),
}
